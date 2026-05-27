import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

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


FP_LEN = 9


def fp_string(amount):
    """First FP_LEN most significant non-zero fractional digits."""
    try:
        s = f"{float(amount):.18f}"
        frac = s.split(".")[1] if "." in s else "0" * 18
        stripped = frac.lstrip("0")
        if len(stripped) < FP_LEN:
            return None
        return stripped[:FP_LEN]
    except (ValueError, TypeError):
        return None


sh = pd.read_csv(TX_DIR / "eth_shield.csv")
un = pd.read_csv(TX_DIR / "eth_unshields.csv")
sw = pd.read_csv(TX_DIR / "eth-swaps.csv")
sh["_t"] = pd.to_datetime(sh["time"], utc=True, errors="coerce")
un["_t"] = pd.to_datetime(un["time"], utc=True, errors="coerce")
sh["_fr"] = _norm(sh["from_address"])
un["_to"] = _norm(un["to_address"])
sh["_th"] = _norm(sh["tx_hash"])
un["_th"] = _norm(un["transaction_hash"])
sw_set = set(_norm(sw["tx_hash"]))

sh = sh[(sh["token_symbol"].str.upper() == "WETH") & sh["_t"].notna()
        & ~sh["_fr"].isin(PROTOCOL_ADDRS) & ~sh["_th"].isin(sw_set)].copy()
un = un[(un["token_symbol"].str.upper() == "WETH") & un["_t"].notna()
        & ~un["_to"].isin(PROTOCOL_ADDRS) & ~un["_th"].isin(sw_set)].copy()
sh = sh.reset_index(drop=True)
un = un.reset_index(drop=True)
print(f"filtered shields  D = {len(sh):,}")
print(f"filtered unshields W = {len(un):,}")

# Fingerprint: first FP_LEN most significant non-zero fractional digits
sh["fp"] = sh["deposit_amount"].apply(fp_string)
un["fp"] = un["amount"].apply(fp_string)

# Drop entries with no usable fingerprint
sh = sh[sh["fp"].notna()].copy()
un = un[un["fp"].notna()].copy()
print(f"shields with usable fingerprint    : {len(sh):,}")
print(f"unshields with usable fingerprint  : {len(un):,}")

# For each unshield, count how many CAUSAL deposits share its fingerprint.
# Causal: deposit time <= unshield time.
# Implementation: for each fingerprint group, sort the deposits by time
# and use searchsorted to count how many fall before each unshield.
sh_sorted = sh.sort_values(["fp", "_t"]).reset_index(drop=True)
dep_t_by_fp = {fp: g["_t"].to_numpy() for fp, g in sh_sorted.groupby("fp")}
dep_idx_by_fp = {fp: g.index.to_numpy() for fp, g in sh_sorted.groupby("fp")}

matching_dep_count = np.zeros(len(un), dtype=np.int64)
sole_dep_idx = np.full(len(un), -1, dtype=np.int64)
for i, (fp, wt) in enumerate(zip(un["fp"].to_numpy(), un["_t"].to_numpy())):
    arr = dep_t_by_fp.get(fp)
    if arr is None:
        continue
    k = int(np.searchsorted(arr, wt, side="right"))
    matching_dep_count[i] = k
    if k == 1:
        sole_dep_idx[i] = int(dep_idx_by_fp[fp][0])
un = un.assign(n_dep=matching_dep_count, sole_dep=sole_dep_idx)

n_with_match = int((un["n_dep"] >= 1).sum())
n_unique = int((un["n_dep"] == 1).sum())
print(f"\nunshields with at least one causal frac3 match  : {n_with_match:,}")
print(f"unshields uniquely narrowed to ONE deposit      : {n_unique:,}  "
      f"({100*n_unique/len(un):.2f}% of W)")

# Build the pair table for the uniquely-narrowed subset
h5_un = un[un["n_dep"] == 1].copy()
h5_un["dep_addr"] = sh_sorted.loc[h5_un["sole_dep"], "_fr"].to_numpy()
h5_un["dep_t"] = sh_sorted.loc[h5_un["sole_dep"], "_t"].to_numpy()
h5_un["dep_amt"] = sh_sorted.loc[h5_un["sole_dep"], "deposit_amount"].to_numpy()

h1_overlap = (h5_un["dep_addr"] == h5_un["_to"]).sum()
print(f"  overlap with H1 (dep_addr == wit_addr)         : "
      f"{int(h1_overlap):,}")
print(f"  H5 \\ H1 (novel after address-reuse dedup)      : "
      f"{int(n_unique - h1_overlap):,}")
pairs = h5_un[["_to", "_t", "amount", "dep_addr", "dep_t", "dep_amt",
                "fp"]].rename(columns={"_to": "wit_addr", "_t": "wit_t",
                                        "amount": "wit_amt"})

# Persist the H5 pair set so compute_unions can join it back
out_path = ROOT / "results" / "h5_pairs.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
pairs.to_csv(out_path, index=False)
print(f"\nwrote {out_path}")

# Statistical validation: Kolmogorov-Smirnov test on the leading 3-digit
# fingerprints (continuous-ish on 0-999) -- compares deposit vs
# withdraw distributions. The shorter prefix is used here so the KS
# test has enough bin density to be meaningful.
dep_fp_arr = np.array([int(s[:3]) for s in sh["fp"].to_numpy()])
wit_fp_arr = np.array([int(s[:3]) for s in un["fp"].to_numpy()])
ks_stat, ks_p = stats.ks_2samp(dep_fp_arr, wit_fp_arr)
print(f"\nKolmogorov-Smirnov 2-sample test on full frac3 distributions:")
print(f"  KS statistic = {ks_stat:.6f}")
print(f"  p-value      = {ks_p:.3e}")
print("  reject H0 (fingerprints independent of side) at p < 0.001"
      if ks_p < 1e-3 else "  cannot reject H0 at p < 0.001")
