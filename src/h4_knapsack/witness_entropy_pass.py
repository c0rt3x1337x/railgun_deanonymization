import argparse
import csv
import ctypes
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "Transactions"
LIB = HERE / "libwitness_kernel.so"
LARGE_ETH = 100

_G = {}


def bucketize(value_wei, step_wei):
    return (value_wei + step_wei // 2) // step_wei


def _wlabel(wd):
    return "all" if wd == 0 else f"{wd}d"


def _load_lib():
    lib = ctypes.CDLL(str(LIB))
    lib.witness_enumerate.argtypes = [
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int32,
        ctypes.c_int64,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.witness_enumerate.restype = ctypes.c_int32
    return lib


def _init(pool_pt, pool_note, pool_bucket, pool_addr_ids, n_addrs,
          step_wei, window_ns, large_t_b, cap, max_pool_slice, max_T_b):
    _G["lib"] = _load_lib()
    _G["pt"] = pool_pt
    _G["note"] = pool_note
    _G["bkt"] = pool_bucket
    _G["addr_ids"] = pool_addr_ids
    _G["n_addrs"] = n_addrs
    _G["step"] = step_wei
    _G["wns"] = window_ns
    _G["large_t_b"] = large_t_b
    _G["cap"] = cap
    _G["pool_n"] = len(pool_pt)
    words = max_T_b // 64 + 1
    n_alloc = max_pool_slice + 1
    _G["bitset_buf"] = np.zeros(int(n_alloc) * int(words), dtype=np.uint64)
    _G["bitset_buf_ptr"] = _G["bitset_buf"].ctypes.data_as(
        ctypes.POINTER(ctypes.c_uint64))
    _G["item_counts"] = np.zeros(max_pool_slice, dtype=np.int32)
    _G["item_counts_ptr"] = _G["item_counts"].ctypes.data_as(
        ctypes.POINTER(ctypes.c_int32))
    _G["dfs_stack"] = np.zeros(max_pool_slice, dtype=np.int32)
    _G["dfs_stack_ptr"] = _G["dfs_stack"].ctypes.data_as(
        ctypes.POINTER(ctypes.c_int32))
    _G["pool_b_scratch"] = np.zeros(max_pool_slice, dtype=np.int64)
    _G["addr_id_scratch"] = np.zeros(max_pool_slice, dtype=np.int32)
    _G["addr_counts"] = np.zeros(n_addrs, dtype=np.int32)
    _G["addr_counts_ptr"] = _G["addr_counts"].ctypes.data_as(
        ctypes.POINTER(ctypes.c_int32))
    _G["addr_seen_in"] = np.zeros(n_addrs, dtype=np.int32)
    _G["addr_seen_in_ptr"] = _G["addr_seen_in"].ctypes.data_as(
        ctypes.POINTER(ctypes.c_int32))


def _process_chunk(args):
    idxs, tgt_t, tgt_T_list, direction = args
    lib = _G["lib"]
    pt = _G["pt"]; note = _G["note"]; bkt = _G["bkt"]
    addr_ids_global = _G["addr_ids"]
    n_addrs = _G["n_addrs"]
    step = _G["step"]; wns = _G["wns"]; large_t_b = _G["large_t_b"]
    cap = _G["cap"]; pool_n = _G["pool_n"]
    bitset_ptr = _G["bitset_buf_ptr"]
    item_counts = _G["item_counts"]
    item_counts_ptr = _G["item_counts_ptr"]
    dfs_stack_ptr = _G["dfs_stack_ptr"]
    pool_b_scratch = _G["pool_b_scratch"]
    addr_id_scratch = _G["addr_id_scratch"]
    addr_counts = _G["addr_counts"]
    addr_counts_ptr = _G["addr_counts_ptr"]
    addr_seen_in_ptr = _G["addr_seen_in_ptr"]
    rows = []
    for k, ti in enumerate(idxs):
        st = int(tgt_t[k]); T = tgt_T_list[k]
        T_b = bucketize(T, step)
        if direction == "1tok":
            lo = int(np.searchsorted(pt, st, side="right"))
            hi = pool_n if wns is None else int(
                np.searchsorted(pt, st + wns, side="right"))
        else:
            hi = int(np.searchsorted(pt, st, side="left"))
            lo = 0 if wns is None else int(
                np.searchsorted(pt, st - wns, side="left"))

        pool_local_n = 0
        pool_addr_local_set = set()
        pool_raw = 0
        for j in range(lo, hi):
            if note[j] > T:
                continue
            pool_raw += 1
            b = int(bkt[j])
            if b <= 0 or b > T_b:
                continue
            if pool_local_n >= len(pool_b_scratch):
                break
            pool_b_scratch[pool_local_n] = b
            addr_id_scratch[pool_local_n] = addr_ids_global[j]
            pool_addr_local_set.add(int(addr_ids_global[j]))
            pool_local_n += 1
        pool_addr_n = len(pool_addr_local_set)
        if pool_raw == 0 or T_b == 0 or pool_local_n == 0 or T_b > large_t_b:
            rows.append({
                "ti": int(ti), "T": T, "T_b": int(T_b),
                "pool_raw": pool_raw, "pool_bucketed": pool_local_n,
                "pool_addr_n": pool_addr_n,
                "K_sampled": 0, "item_n": 0, "addr_n": 0,
                "H_knap_item": 0.0, "H_knap_addr": 0.0,
                "H_naive_item": math.log2(max(pool_raw, 1)) if pool_raw else 0,
                "H_naive_addr": math.log2(max(pool_addr_n, 1)) if pool_addr_n else 0,
                "saturated": False, "skipped": True,
            })
            continue

        pool_b_view = pool_b_scratch[:pool_local_n]
        pool_b_ptr = pool_b_view.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        addr_id_view = addr_id_scratch[:pool_local_n]
        addr_id_ptr = addr_id_view.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        K_sampled = lib.witness_enumerate(
            pool_b_ptr, ctypes.c_int32(pool_local_n),
            ctypes.c_int64(int(T_b)), ctypes.c_int32(int(cap)),
            bitset_ptr, item_counts_ptr, dfs_stack_ptr,
            addr_id_ptr, ctypes.c_int32(int(n_addrs)),
            addr_counts_ptr, addr_seen_in_ptr,
        )

        if K_sampled == 0:
            rows.append({
                "ti": int(ti), "T": T, "T_b": int(T_b),
                "pool_raw": pool_raw, "pool_bucketed": pool_local_n,
                "pool_addr_n": pool_addr_n,
                "K_sampled": 0, "item_n": 0, "addr_n": 0,
                "H_knap_item": 0.0, "H_knap_addr": 0.0,
                "H_naive_item": math.log2(max(pool_raw, 1)),
                "H_naive_addr": math.log2(max(pool_addr_n, 1)),
                "saturated": False, "skipped": False,
            })
            continue

        def shannon_from_array(arr):
            tot = int(arr.sum())
            if tot == 0:
                return 0.0, 0
            nz = arr[arr > 0]
            p = nz.astype(np.float64) / tot
            H = float(-(p * np.log2(p)).sum())
            return H, int(len(nz))

        H_addr, n_addr = shannon_from_array(addr_counts)
        H_item, n_item = shannon_from_array(item_counts[:pool_local_n])

        rows.append({
            "ti": int(ti), "T": T, "T_b": int(T_b),
            "pool_raw": pool_raw, "pool_bucketed": pool_local_n,
            "pool_addr_n": pool_addr_n,
            "K_sampled": int(K_sampled), "item_n": int(n_item), "addr_n": int(n_addr),
            "H_knap_item": H_item, "H_knap_addr": H_addr,
            "H_naive_item": math.log2(max(pool_raw, 1)),
            "H_naive_addr": math.log2(max(pool_addr_n, 1)),
            "saturated": (K_sampled >= cap),
            "skipped": False,
        })
    return rows


def run(direction, step_wei, window_days, workers, cap, out_path):
    eth_label = f"{step_wei / 1e18:g}"
    wlabel = _wlabel(window_days)
    large_t_b = LARGE_ETH * 10 ** 18 // step_wei
    window_ns = None if window_days == 0 else int(window_days * 86400 * 1e9)
    if not LIB.exists():
        sys.exit(f"missing {LIB.name} -- build first: "
                 f"gcc -O3 -march=native -shared -fPIC "
                 f"-o {LIB.name} witness_kernel.c")

    sh = pd.read_csv(DATA / "eth_shield_aggregated.csv")
    un = pd.read_csv(DATA / "eth_unshields_aggregated.csv")
    print(f"loaded shields={len(sh):,} unshields={len(un):,}", flush=True)

    if direction == "1tok":
        pool_df = un; pool_time_col = "first_time_ns"
        tgt_df = sh; tgt_time_col = "last_time_ns"
    else:
        pool_df = sh; pool_time_col = "last_time_ns"
        tgt_df = un; tgt_time_col = "first_time_ns"

    p_t = pool_df[pool_time_col].astype("int64").to_numpy()
    order = np.argsort(p_t, kind="mergesort")
    pool_pt = p_t[order]
    pool_note = [int(v) for v in pool_df["note_wei"].to_numpy()[order]]
    pool_bucket = np.fromiter(
        (bucketize(v, step_wei) for v in pool_note),
        dtype="int64", count=len(pool_note))
    pool_addr_str = pool_df["address"].astype(str).str.lower().str.strip(
        ).to_numpy()[order]
    unique_addrs, pool_addr_ids = np.unique(pool_addr_str, return_inverse=True)
    pool_addr_ids = pool_addr_ids.astype(np.int32)
    n_addrs = int(len(unique_addrs))
    print(f"global distinct addresses in pool = {n_addrs:,}", flush=True)

    t_t = tgt_df[tgt_time_col].astype("int64").to_numpy()
    t_note = [int(v) for v in tgt_df["note_wei"].to_numpy()]
    n = len(tgt_df)

    if workers <= 0:
        workers = os.cpu_count() or 1
    chunk = max(64, (n + workers * 8 - 1) // (workers * 8))
    tasks = []
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        idxs = list(range(s, e))
        tasks.append((idxs, t_t[s:e], [t_note[i] for i in idxs], direction))

    if window_days == 0:
        max_pool_slice = min(len(pool_pt), 50000)
    elif window_days >= 180:
        max_pool_slice = min(len(pool_pt), 15000)
    elif window_days >= 30:
        max_pool_slice = min(len(pool_pt), 5000)
    elif window_days >= 7:
        max_pool_slice = min(len(pool_pt), 2000)
    else:
        max_pool_slice = min(len(pool_pt), 1000)
    max_T_b = int(large_t_b) + 1
    bytes_per_worker = (max_pool_slice + 1) * (max_T_b // 64 + 1) * 8
    print(f"direction={direction} window={wlabel} step={eth_label} ETH "
          f"workers={workers} cap={cap} n={n:,} chunks={len(tasks)} "
          f"(~{chunk}/chunk)", flush=True)
    print(f"buffer/worker = {bytes_per_worker/1e6:.1f} MB  "
          f"(max_pool_slice={max_pool_slice}, max_T_b={max_T_b:,})", flush=True)

    import multiprocessing as mp
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    FIELDS = ["ti", "T", "T_b", "pool_raw", "pool_bucketed", "pool_addr_n",
              "K_sampled", "item_n", "addr_n", "H_knap_item", "H_knap_addr",
              "H_naive_item", "H_naive_addr", "saturated", "skipped"]
    fh = open(out_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=FIELDS)
    writer.writeheader(); fh.flush()
    n_rows = 0
    done = 0
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=workers, initializer=_init,
                  initargs=(pool_pt, pool_note, pool_bucket,
                            pool_addr_ids, n_addrs,
                            step_wei, window_ns, large_t_b, cap,
                            max_pool_slice, max_T_b)) as pool:
        for res in pool.imap_unordered(_process_chunk, tasks):
            for r in res:
                writer.writerow(r)
                n_rows += 1
            fh.flush()
            done += 1
            if done % 5 == 0 or done == len(tasks):
                el = time.perf_counter() - t0
                rate = done / el if el > 0 else 0
                eta = (len(tasks) - done) / rate if rate > 0 else 0
                print(f"  [{done:>4}/{len(tasks)}] ({100*done/len(tasks):5.1f}%) "
                      f"rows={n_rows:,} elapsed={el:7.1f}s ETA={eta:7.1f}s",
                      flush=True)
    fh.close()
    elapsed = time.perf_counter() - t0
    df = pd.read_csv(out_path)
    print(f"\nwrote {out_path} ({len(df):,} rows) in {elapsed:.1f}s")

    valid = df[~df["skipped"]]
    linked = valid[valid["K_sampled"] >= 1]
    print(f"  total           : {len(df):,}")
    print(f"  skipped (none/large/oversize): {int(df['skipped'].sum()):,}")
    print(f"  linked (K>=1)   : {len(linked):,}")
    print(f"  saturated (K==cap): {int(linked['saturated'].sum()):,}")
    if len(linked):
        dH_addr = (linked['H_naive_addr'] - linked['H_knap_addr']).clip(lower=0)
        print(f"  median dH_addr  : {dH_addr.median():.3f} bits")
        print(f"  mean   dH_addr  : {dH_addr.mean():.3f} bits")
        print(f"  fully deanon (H_knap_addr==0): "
              f"{int((linked['H_knap_addr']==0).sum()):,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", choices=["1tok", "kto1"], required=True)
    ap.add_argument("--step-wei", type=int, required=True)
    ap.add_argument("--window-days", type=int, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cap", type=int, default=500)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    run(args.direction, args.step_wei, args.window_days,
        args.workers, args.cap, args.out)
