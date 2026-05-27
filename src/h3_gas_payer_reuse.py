import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
import os
from pathlib import Path
import pandas as pd

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

# Broadcaster threshold: a gas_payer is tagged as a community broadcaster
# if it meets all three. The clustering self-broadcast set is every
# unshield whose gas_payer is NOT tagged.
BROADCASTER_N = 10   # at least this many events served
BROADCASTER_K = 5    # at least this many distinct recipients
BROADCASTER_L = 30   # at least this many days of activity lifetime


def _norm(s):
    return s.astype(str).str.lower().str.strip()


sh = pd.read_csv(TX_DIR / "eth_shield.csv")
un = pd.read_csv(TX_DIR / "eth_unshields.csv")
sw = pd.read_csv(TX_DIR / "eth-swaps.csv")
sh["_t"] = pd.to_datetime(sh["time"], utc=True, errors="coerce")
un["_t"] = pd.to_datetime(un["time"], utc=True, errors="coerce")
sh["_fr"] = _norm(sh["from_address"])
un["_to"] = _norm(un["to_address"])
un["_gp"] = _norm(un["gas_payer"])
sh["_th"] = _norm(sh["tx_hash"])
un["_th"] = _norm(un["transaction_hash"])
sw_set = set(_norm(sw["tx_hash"]))

sh = sh[(sh["token_symbol"].str.upper() == "WETH") & sh["_t"].notna()
        & ~sh["_fr"].isin(PROTOCOL_ADDRS) & ~sh["_th"].isin(sw_set)].copy()
un = un[(un["token_symbol"].str.upper() == "WETH") & un["_t"].notna()
        & ~un["_to"].isin(PROTOCOL_ADDRS) & ~un["_th"].isin(sw_set)
        & un["_gp"].str.match(r"^0x[0-9a-f]{40}$", na=False)].copy()
print(f"filtered shields  D = {len(sh):,}")
print(f"filtered unshields W = {len(un):,}")

depositor_set = set(sh["_fr"])

# Per gas_payer activity table
gp = un.groupby("_gp").agg(
    events=("_t", "size"),
    distinct_recipients=("_to", "nunique"),
    first=("_t", "min"),
    last=("_t", "max"),
).reset_index()
gp["lifetime_days"] = (gp["last"] - gp["first"]).dt.days

mask_bc = ((gp["events"] >= BROADCASTER_N)
           & (gp["distinct_recipients"] >= BROADCASTER_K)
           & (gp["lifetime_days"] >= BROADCASTER_L))
broadcaster_set = set(gp.loc[mask_bc, "_gp"])
print(f"broadcaster set (N={BROADCASTER_N}, k={BROADCASTER_K}, "
      f"L={BROADCASTER_L}) : {len(broadcaster_set):,} addresses")

# Self-broadcast unshields: gas_payer NOT in the broadcaster set
sb = un[~un["_gp"].isin(broadcaster_set)].copy()
n_sb = len(sb)
n_sb_addrs = sb["_gp"].nunique()
sb_addrs_in_d = set(sb["_gp"].unique()) & depositor_set
print(f"\nself-broadcast unshield events            : {n_sb:,}  "
      f"({100*n_sb/len(un):.2f}% of W)")
print(f"  distinct self-broadcast gas-payer addrs : {n_sb_addrs:,}")
print(f"  of which also depositors                : {len(sb_addrs_in_d):,}  "
      f"({100*len(sb_addrs_in_d)/max(1,n_sb_addrs):.1f}%)")

# Split into gas_payer == w and gas_payer != w
same = sb["_gp"] == sb["_to"]
sb_same = sb[same]
sb_diff = sb[~same]
print(f"\n  gas_payer == w_i (counted under H1)     : {len(sb_same):,} events, "
      f"{sb_same['_gp'].nunique():,} distinct gas-payers")
print(f"  gas_payer != w_i (H3 candidates)         : {len(sb_diff):,} events, "
      f"{sb_diff['_gp'].nunique():,} distinct gas-payers")

# H3 = distinct gas-payer addresses in the gp != w subset that are also depositors
h3_addrs = set(sb_diff["_gp"].unique()) & depositor_set
print(f"\nH3 distinct gas-payer addresses (gp != w AND gp in D): "
      f"{len(h3_addrs):,}")

# Event count: unshields whose gas_payer falls in the H3 address set
h3_events = sb_diff[sb_diff["_gp"].isin(h3_addrs)]
print(f"H3 unshield events                        : {len(h3_events):,}  "
      f"({100*len(h3_events)/len(un):.2f}% of W)")

# Robustness across alternative thresholds
print(f"\nrobustness across (N, k, L) thresholds:")
print(f"  {'(N, k, L)':>14} | {'broadcasters':>13} | {'sb addrs':>8} | "
      f"{'in D':>5} | {'pct':>5}")
for N, k, L in [(5, 2, 7), (10, 5, 30), (25, 5, 90), (50, 2, 180)]:
    m = ((gp["events"] >= N) & (gp["distinct_recipients"] >= k)
         & (gp["lifetime_days"] >= L))
    bset = set(gp.loc[m, "_gp"])
    nb_addrs = set(gp["_gp"]) - bset
    in_d = nb_addrs & depositor_set
    pct = 100 * len(in_d) / max(1, len(nb_addrs))
    print(f"  {f'({N}, {k}, {L})':>14} | {len(bset):>13,} | "
          f"{len(nb_addrs):>8,} | {len(in_d):>5,} | {pct:>4.1f}%")
