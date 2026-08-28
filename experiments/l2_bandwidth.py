#!/usr/bin/env python3
"""L2-E1:带宽常数复测(单卡 D2D + 双卡 all-reduce 曲线,1MB→1GB)。

repo 自带(不依赖项目三旧数据盘——该实例已释放)。对齐项目三 Phase 1
已归档常数:单卡 D2D ≈1526 GB/s,双卡 all-reduce 饱和 ≈30.3 GB/s(≈50×)。
运行:torchrun --nproc_per_node=2 experiments/l2_bandwidth.py
产出:/root/autodl-tmp/l2results/e1_bandwidth.json
判定:同型实例常数与归档值同量级(±20%);显著偏离 → 记录环境差异再解读。
"""
import json
import os
import time

import torch
import torch.distributed as dist

SIZES_MB = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
WARMUP, ITERS = 3, 10


def bench(fn, bytes_moved, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return bytes_moved / dt / 1e9  # GB/s


def main():
    dist.init_process_group("nccl")
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    out = {"d2d_GBs": {}, "allreduce_GBs": {}}

    # 单卡 D2D(rank0):同卡显存拷贝,带宽上界锚点
    if rank == 0:
        for mb in SIZES_MB:
            n = mb * 1024 * 1024 // 4
            a = torch.ones(n, device="cuda:0")
            b = torch.empty(n, device="cuda:0")
            # copy 读写各 n*4 字节 → 搬运量 2×
            gbs = bench(lambda: b.copy_(a), 2 * n * 4)
            out["d2d_GBs"][mb] = round(gbs, 1)
            print(f"D2D {mb}MB: {gbs:.0f} GB/s", flush=True)
            del a, b

    # 双卡 all-reduce:环带宽锚点(代数带宽 = 2(N-1)/N × 数据量 / 时间,N=2)
    for mb in SIZES_MB:
        n = mb * 1024 * 1024 // 4
        x = torch.ones(n, device=f"cuda:{rank}")
        dist.barrier()
        gbs = bench(lambda: dist.all_reduce(x), 2 * n * 4)  # N=2 → 2(N-1)/N=1,搬运 2×数据量(读写)
        if rank == 0:
            out["allreduce_GBs"][mb] = round(gbs, 1)
            print(f"allreduce {mb}MB: {gbs:.0f} GB/s", flush=True)
        del x

    if rank == 0:
        os.makedirs("/root/autodl-tmp/l2results", exist_ok=True)
        with open("/root/autodl-tmp/l2results/e1_bandwidth.json", "w") as f:
            json.dump(out, f, indent=2)
        peak_d2d = max(out["d2d_GBs"].values())
        peak_ar = max(out["allreduce_GBs"].values())
        print(f"E1 done: D2D peak {peak_d2d} GB/s, allreduce peak {peak_ar} GB/s, "
              f"ratio {peak_d2d / peak_ar:.0f}x (归档锚点: 1526 / 30.3 ≈ 50x)")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
