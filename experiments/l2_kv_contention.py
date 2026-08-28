#!/usr/bin/env python3
"""L2-E3:H1 真机锚点——KV 搬运 × NCCL 训练共置争抢。

三场景(B=KV 块搬运代理 = NCCL send/recv 大块;A=训练代理 = all_reduce 循环):
  kv_only      : B 单独跑 → KV 块延迟基线
  colocated    : A+B 并发 → KV 块延迟恶化(对照 L1 的 ~2×/16× 结构)
  colocated_cap: A 降频(sleep pacing,代理源侧限速)→ KV 恢复 + A 变慢(对照 H2)
运行:torchrun --nproc_per_node=2 experiments/l2_kv_contention.py [kv_only|colocated|colocated_cap]
产出:/root/autodl-tmp/l2results/e3_<mode>.json(rank1 的 KV 块延迟分布)
"""
import json
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist
from datetime import timedelta

KV_BYTES = 128 * 1024 * 1024      # 128MB/块(KV 搬运代理)
KV_BLOCKS = 40
TRAIN_ITERS = 2000
TRAIN_ELEMS = 32 * 1024 * 1024    # 128MB fp32 all_reduce(训练梯度代理)
TRAIN_SLEEP = 0.004               # colocated_cap:限速 pacing(秒/迭代)


def kv_loop(rank, log):
    """rank0 send → rank1 recv,KV 块延迟在 rank1 计。"""
    t = torch.empty(KV_BYTES // 4, dtype=torch.float32, device=f"cuda:{rank}")
    for i in range(KV_BLOCKS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if rank == 0:
            dist.send(t, dst=1)
        else:
            dist.recv(t, src=0)
            torch.cuda.synchronize()
            log.append(time.perf_counter() - t0)


def train_loop(rank, group):
    """训练代理:走独立 NCCL 组 g——与 KV 线程(默认组 P2P)并发安全
    (同一 PG 跨线程并发 collective 不安全,可能死锁而非争抢)。"""
    x = torch.ones(TRAIN_ELEMS, device=f"cuda:{rank}")
    for _ in range(TRAIN_ITERS):
        dist.all_reduce(x, group=group)
        if os.environ.get("MODE") == "colocated_cap":
            time.sleep(TRAIN_SLEEP)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "kv_only"
    os.environ["MODE"] = mode
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    g = dist.new_group(ranks=[0, 1])
    dist.barrier()

    kv_log = []
    if mode == "kv_only":
        kv_loop(rank, kv_log)
    else:
        if rank == 0:
            import threading
            th = threading.Thread(target=kv_loop, args=(rank, kv_log))
            th.start()
            train_loop(rank, g)
            th.join()
        else:
            # rank1:KV 接收计时为主线程,训练为子线程(对称)
            import threading
            th = threading.Thread(target=train_loop, args=(rank, g))
            th.start()
            kv_loop(rank, kv_log)
            th.join()

    dist.barrier()
    if rank == 1:
        if not kv_log:
            raise RuntimeError("KV 块日志为空:场景未按预期执行")
        ms = [x * 1000 for x in kv_log]
        out = {"mode": mode, "blocks": len(ms),
               "p50_ms": round(statistics.median(ms), 3),
               "p95_ms": round(sorted(ms)[int(0.95 * len(ms))], 3),
               "mean_ms": round(statistics.mean(ms), 3)}
        os.makedirs("/root/autodl-tmp/l2results", exist_ok=True)
        with open(f"/root/autodl-tmp/l2results/e3_{mode}.json", "w") as f:
            json.dump(out, f, indent=2)
        print("E3 done:", out)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
