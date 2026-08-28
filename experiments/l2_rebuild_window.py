#!/usr/bin/env python3
"""L2-E2:NCCL communicator 重建窗口标定(回填仿真 A2 占位常数 50ms)。

在 2×GPU 上循环 N 次:destroy_process_group → init_process_group →
首次 all_reduce,分段计时(init / 首次集合通信)。
运行(AutoDL,开机后):torchrun --nproc_per_node=2 experiments/l2_rebuild_window.py
产出:/root/autodl-tmp/l2results/e2_rebuild.json
判定:重建窗口 = init+首 collective 总耗时分布(中位/95 分位)。
"""
import json
import os
import statistics
import time

import torch
import torch.distributed as dist


def main():
    rank = int(os.environ["RANK"])
    results = []
    for i in range(20):
        t0 = time.perf_counter()
        dist.init_process_group("nccl", timeout=__import__("datetime").timedelta(seconds=120))
        t_init = time.perf_counter() - t0
        torch.cuda.set_device(rank)
        x = torch.ones(64 * 1024 * 1024, device=f"cuda:{rank}")  # 64M fp32 = 256MB
        dist.barrier()
        t1 = time.perf_counter()
        dist.all_reduce(x)
        torch.cuda.synchronize()
        t_coll = time.perf_counter() - t1
        dist.destroy_process_group()
        if rank == 0:
            results.append({"iter": i, "init_s": round(t_init, 6),
                            "first_coll_s": round(t_coll, 6)})
            print(f"[{i}] init={t_init * 1000:.1f}ms first_coll={t_coll * 1000:.1f}ms",
                  flush=True)
    if rank == 0:
        os.makedirs("/root/autodl-tmp/l2results", exist_ok=True)
        inits = [r["init_s"] for r in results]
        colls = [r["first_coll_s"] for r in results]
        out = {"n": len(results),
               "init_median_ms": round(statistics.median(inits) * 1000, 2),
               "init_p95_ms": round(sorted(inits)[int(0.95 * len(inits))] * 1000, 2),
               "first_coll_median_ms": round(statistics.median(colls) * 1000, 2),
               "raw": results}
        with open("/root/autodl-tmp/l2results/e2_rebuild.json", "w") as f:
            json.dump(out, f, indent=2)
        print("E2 done:", {k: v for k, v in out.items() if k != "raw"})


if __name__ == "__main__":
    main()
