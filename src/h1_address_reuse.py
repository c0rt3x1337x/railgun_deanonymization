import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import pandas as pd

import os
ROOT = Path(os.environ.get("RAILGUN_ROOT",
                            Path(__file__).resolve().parents[1]))
TX_DIR = Path(os.environ.get("RAILGUN_TX_DIR", ROOT / "data" / "transactions"))
H1_PPOI_CSV = Path(os.environ.get("RAILGUN_PPOI_CSV",
                                    ROOT / "data" / "h1_pairs_with_ppoi_tags.csv"))

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
print(f"filtered shields  W_s = {len(sh):,}")
print(f"filtered unshields W   = {len(un):,}")

earliest_sh = sh.groupby("fr_l")["time"].min()

un["earliest_sh"] = un["to_l"].map(earliest_sh)
h1_un = (un["earliest_sh"].notna() & (un["earliest_sh"] <= un["time"]))
print(f"\nH1 unshield events (causal): {int(h1_un.sum()):,}  "
      f"({100*h1_un.mean():.2f}% of W={len(un):,})")

latest_un = un.groupby("to_l")["time"].max()
sh["latest_un"] = sh["fr_l"].map(latest_un)
h1_sh = (sh["latest_un"].notna() & (sh["time"] <= sh["latest_un"]))
print(f"H1 shield events  (causal): {int(h1_sh.sum()):,}  "
      f"({100*h1_sh.mean():.2f}% of W_s={len(sh):,})")

sh_addrs = set(sh["fr_l"])
un_addrs = set(un["to_l"])
both = sh_addrs & un_addrs
print(f"\nDistinct addresses on both sides of the boundary: {len(both):,}")

if H1_PPOI_CSV.exists():
    pp = pd.read_csv(H1_PPOI_CSV)
    print(f"\nh1_pairs_with_ppoi_tags.csv  rows={len(pp):,}")
    print(f"  any_ppoi_hit = True  : {int(pp['any_ppoi_hit'].astype(bool).sum()):,}")
    print(f"  any_ppoi_hit = False : {int((~pp['any_ppoi_hit'].astype(bool)).sum()):,}")
    print(f"  distinct from_addr   : {pp['from_addr'].nunique():,}")
    pp_flagged_addrs = set(
        _norm(pp.loc[pp["any_ppoi_hit"].astype(bool), "from_addr"]))
    print(f"  distinct flagged from_addr: {len(pp_flagged_addrs):,}")
    h1_pp = h1_un & un["to_l"].isin(pp_flagged_addrs)
    h1_clean = h1_un & ~un["to_l"].isin(pp_flagged_addrs)
    print(f"\nOf the {int(h1_un.sum()):,} H1-positive unshield events:")
    print(f"  to_address is PPOI-flagged  : {int(h1_pp.sum()):,}  "
          f"({100*h1_pp.sum()/h1_un.sum():.2f}%)")
    print(f"  to_address NOT PPOI-flagged : {int(h1_clean.sum()):,}  "
          f"({100*h1_clean.sum()/h1_un.sum():.2f}%)")
else:
    print(f"\nPPOI file not found at {H1_PPOI_CSV}")
