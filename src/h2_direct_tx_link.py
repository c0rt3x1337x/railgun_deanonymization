import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import pandas as pd
import numpy as np

import os
ROOT = Path(os.environ.get("RAILGUN_ROOT",
                            Path(__file__).resolve().parents[1]))
TX_DIR = Path(os.environ.get("RAILGUN_TX_DIR", ROOT / "data" / "transactions"))

PROTOCOL_ADDRS = frozenset({
    "0xfa7093cdd9ee6932b4eb2c9e1cde7ce00b1fa4b9",
    "0xac9f360ae85469b27aeddeafc579ef2d052ad405",
    "0x4025ee6512dbbda97049bcf5aa5d38c54af6be8a",
    "0xe8a8b458bcd1ececc6b6b58f80929b29ccecff40",
    "0x22af4edbea3de885dda8f0a0653e6209e44e5b84",
    "0xc3f2c8f9d5f0705de706b1302b7a039e1e11ac88",
    "0x0000000000000000000000000000000000000000",
})


def _norm(s):
    return s.astype(str).str.lower().str.strip()


sh = pd.read_csv(TX_DIR / "eth_shield.csv")
un = pd.read_csv(TX_DIR / "eth_unshields.csv")
sw = pd.read_csv(TX_DIR / "eth-swaps.csv")
sh["time"] = pd.to_datetime(sh["time"], utc=True, errors="coerce")
un["time"] = pd.to_datetime(un["time"], utc=True, errors="coerce")
sh["fr_l"] = _norm(sh["from_address"])
un["to_l"] = _norm(un["to_address"])
sh["th_l"] = _norm(sh["tx_hash"])
un["th_l"] = _norm(un["transaction_hash"])
sw_set = set(_norm(sw["tx_hash"]))
sh = sh[(sh["token_symbol"].str.upper() == "WETH") & sh["time"].notna()
        & ~sh["fr_l"].isin(PROTOCOL_ADDRS) & ~sh["th_l"].isin(sw_set)].copy()
un = un[(un["token_symbol"].str.upper() == "WETH") & un["time"].notna()
        & ~un["to_l"].isin(PROTOCOL_ADDRS) & ~un["th_l"].isin(sw_set)].copy()
un = un.reset_index(drop=True)
print(f"filtered shields={len(sh):,}  unshields={len(un):,}")

depositor_set = set(sh["fr_l"].unique())
withdraw_set = set(un["to_l"].unique())

earliest_shield = sh.groupby("fr_l")["time"].min()


def collect_pairs(path):
    out = []
    for chunk in pd.read_csv(path, chunksize=500_000,
                             usecols=["from_address", "to_address"]):
        f = _norm(chunk["from_address"])
        t = _norm(chunk["to_address"])
        m = (~f.isin(PROTOCOL_ADDRS)) & (~t.isin(PROTOCOL_ADDRS))
        m &= ((f.isin(depositor_set) & t.isin(withdraw_set))
              | (f.isin(withdraw_set) & t.isin(depositor_set)))
        sub = chunk[m].copy()
        if len(sub):
            ff = _norm(sub["from_address"])
            tt = _norm(sub["to_address"])
            d = np.where(ff.isin(depositor_set), ff, tt)
            w = np.where(tt.isin(withdraw_set), tt, ff)
            out.append(pd.DataFrame({"d": d, "w": w}))
    if not out:
        return pd.DataFrame(columns=["d", "w"])
    return pd.concat(out, ignore_index=True).drop_duplicates()


print("\ncollecting (d, w) pairs from eth_external2 + eth_erc20 ...")
pairs = pd.concat([
    collect_pairs(TX_DIR / "eth_external2.csv"),
    collect_pairs(TX_DIR / "eth_erc20.csv"),
], ignore_index=True).drop_duplicates()
print(f"  distinct (d, w) pairs: {len(pairs):,}")
print(f"  ... of which d == w  : {int((pairs['d'] == pairs['w']).sum()):,}")

linked_d_for_w = pairs.groupby("w")["d"].apply(set).to_dict()


def causal_test(row):
    w_addr = row["to_l"]
    w_time = row["time"]
    ds = linked_d_for_w.get(w_addr, ())
    for d in ds:
        es = earliest_shield.get(d)
        if pd.notna(es) and es < w_time:
            return True
    return False


print("\napplying event-level causal rule ...")
mask = un.apply(causal_test, axis=1)
n = int(mask.sum())
print(f"\nH2 (event-level causal): {n:,} unshield events "
      f"({100*n/len(un):.2f}% of {len(un):,})")

# Variant: exclude d == w (the d == w case is H1)
pairs_dne = pairs[pairs["d"] != pairs["w"]]
linked_d_for_w_dne = pairs_dne.groupby("w")["d"].apply(set).to_dict()


def causal_test_dne(row):
    w_addr = row["to_l"]
    w_time = row["time"]
    ds = linked_d_for_w_dne.get(w_addr, ())
    for d in ds:
        es = earliest_shield.get(d)
        if pd.notna(es) and es < w_time:
            return True
    return False


mask2 = un.apply(causal_test_dne, axis=1)
n2 = int(mask2.sum())
print(f"H2 (event-level causal, excluding d==w): {n2:,} "
      f"({100*n2/len(un):.2f}%)")
