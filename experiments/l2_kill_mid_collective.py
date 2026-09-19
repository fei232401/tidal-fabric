#!/usr/bin/env python3
"""L2-E4:竞态#1 实证——缩容撞 collective(all-reduce 中途 kill 一个 rank)。

父进程拉起两个 worker(rank0/1,NCCL all_reduce 循环,timeout 设 30s 便于观测,
生产默认 600s——如实记录两者)。随机 t∈[2,5]s SIGKILL rank1;
rank0 记录:kill → NCCL 异常抛出的时长与错误类型(预期:ProcessGroupNCCL
timeout / connection reset,而非静默挂死)。

v2 关键修复(相对 zcode 初版):
  1. deadline 真生效:后台 reader 线程喂 queue + 主线程 queue.get(timeout) 轮询。
     原版阻塞 readline() 若 rank0 一直不抛异常会永久卡死,deadline 形同虚设;
  2. kill→异常延迟 = 绝对 wall-clock 相减(worker 异常时刻的 time.time() 绝对值
     − 主进程 kill 时刻的 time.time()),消除 worker fork/import 造成的 t0 偏差;
  3. 子进程注入 NCCL_DEBUG=INFO,拿到错误路径证据(注:INFO 日志同步写入会轻微
     影响时序,属观测者效应;纯时序测量可改 NCCL_DEBUG=WARN)。

运行:python experiments/l2_kill_mid_collective.py(trials=3)
产出:/root/autodl-tmp/l2results/e4_kill.json
判定:观测到 rank0 抛 ProcessGroupNCCL timeout/connection reset(而非静默挂死),
     并记录 kill→异常 时长分布。
"""
import json
import os
import queue
import random
import signal
import subprocess
import sys
import threading
import time

WORKER = r'''
import os, time, torch, torch.distributed as dist
from datetime import timedelta
rank = int(os.environ["RANK"])
dist.init_process_group("nccl", timeout=timedelta(seconds=30))
torch.cuda.set_device(rank)
x = torch.ones(32*1024*1024, device=f"cuda:{rank}")
t0 = time.time()
try:
    while True:
        dist.all_reduce(x)
        if rank == 0 and int(time.time()-t0) % 5 == 0:
            print(f"alive {time.time()-t0:.1f}s", flush=True)
        time.sleep(0.001)
except Exception as e:
    # 绝对 wall-clock(同机与主进程一致),主进程据此算 kill→异常延迟
    print(f"RANK0_EXCEPTION|{time.time():.3f}|{type(e).__name__}|{str(e)[:200]}", flush=True)
    raise
'''


def main():
    random.seed(42)
    trials = []
    for i in range(3):
        env = dict(os.environ, MASTER_ADDR="127.0.0.1",
                   MASTER_PORT=str(29500 + i), WORLD_SIZE="2",
                   NCCL_DEBUG="INFO")
        t_start = time.time()
        procs = {r: subprocess.Popen(
            [sys.executable, "-c", WORKER], env=dict(env, RANK=str(r)),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for r in (0, 1)}
        time.sleep(random.uniform(2, 5))
        procs[1].send_signal(signal.SIGKILL)
        t_kill = time.time()

        q = queue.Queue()
        threading.Thread(target=lambda: [q.put(l) for l in procs[0].stdout],
                         daemon=True).start()
        exc_line = None
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                line = q.get(timeout=0.2)
            except queue.Empty:
                if procs[0].poll() is not None:
                    break
                continue
            if "RANK0_EXCEPTION" in line:
                parts = line.strip().split("|")
                exc_line = {"kill_to_exception_s": round(float(parts[1]) - t_kill, 3),
                            "type": parts[2], "msg": parts[3][:120]}
                break

        for p in procs.values():
            p.kill()
        trials.append({"trial": i,
                       "kill_at_s": round(t_kill - t_start, 3),
                       "rank0_exception": exc_line})
        print(f"trial {i}: {trials[-1]}", flush=True)

    os.makedirs("/root/autodl-tmp/l2results", exist_ok=True)
    with open("/root/autodl-tmp/l2results/e4_kill.json", "w") as f:
        json.dump(trials, f, indent=2)
    print("E4 done → /root/autodl-tmp/l2results/e4_kill.json")


if __name__ == "__main__":
    main()
