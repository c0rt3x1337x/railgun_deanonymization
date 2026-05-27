import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import os
ROOT = Path(os.environ.get("RAILGUN_ROOT",
                            Path(__file__).resolve().parents[1]))
TX_DIR = Path(os.environ.get("RAILGUN_TX_DIR", ROOT / "data" / "transactions"))
H4_OUT = Path(os.environ.get("RAILGUN_AGG_DIR", ROOT / "data" / "aggregated"))
WIT_DIR = Path(os.environ.get("RAILGUN_WITNESS_DIR", ROOT / "results" / "witness"))

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


print("loading filtered shields / unshields ...")
sh = pd.read_csv(TX_DIR / "eth_shield.csv")
un = pd.read_csv(TX_DIR / "eth_unshields.csv")
sw = pd.read_csv(TX_DIR / "eth-swaps.csv")
sh["time"] = pd.to_datetime(sh["time"], utc=True, errors="coerce")
un["time"] = pd.to_datetime(un["time"], utc=True, errors="coerce")
sh["fr_l"] = _norm(sh["from_address"])
un["to_l"] = _norm(un["to_address"])
un["gp_l"] = _norm(un["gas_payer"])
sh["th_l"] = _norm(sh["tx_hash"])
un["th_l"] = _norm(un["transaction_hash"])
sw_set = set(_norm(sw["tx_hash"]))

sh = sh[(sh["token_symbol"].str.upper() == "WETH") & sh["time"].notna()
        & ~sh["fr_l"].isin(PROTOCOL_ADDRS) & ~sh["th_l"].isin(sw_set)].copy()
un = un[(un["token_symbol"].str.upper() == "WETH") & un["time"].notna()
        & ~un["to_l"].isin(PROTOCOL_ADDRS) & ~un["th_l"].isin(sw_set)
        & un["gp_l"].str.match(r"^0x[0-9a-f]{40}$", na=False)].copy()
un = un.reset_index(drop=True)
sh = sh.reset_index(drop=True)
N_un = len(un)
print(f"  filtered shields  = {len(sh):,}")
print(f"  filtered unshields= {N_un:,}")
depositor_set = set(sh["fr_l"].unique())

earliest_shield_by_addr = sh.groupby("fr_l")["time"].min()
un["earliest_sh_for_to"] = un["to_l"].map(earliest_shield_by_addr)
un["earliest_sh_for_gp"] = un["gp_l"].map(earliest_shield_by_addr)

# H1
print("\nH1 (address reuse, causal)")
h1 = (un["earliest_sh_for_to"].notna()
      & (un["earliest_sh_for_to"] <= un["time"]))
print(f"  events: {int(h1.sum()):>6,}  ({100*h1.mean():5.2f}%)")

# H2
print("\nH2 (event-level causal: edge(d,w) AND d.earliest_shield < w.time)")
withdraw_addrs = set(un["to_l"].unique())
pairs = []
for p in [TX_DIR / "eth_external2.csv", TX_DIR / "eth_erc20.csv"]:
    if not p.exists():
        continue
    for chunk in pd.read_csv(p, chunksize=500_000,
                             usecols=["from_address", "to_address"]):
        f = _norm(chunk["from_address"])
        t = _norm(chunk["to_address"])
        m = (f.isin(depositor_set) & t.isin(withdraw_addrs)) | \
            (f.isin(withdraw_addrs) & t.isin(depositor_set))
        sub = chunk[m].copy()
        if len(sub):
            ff = _norm(sub["from_address"])
            tt = _norm(sub["to_address"])
            d = np.where(ff.isin(depositor_set), ff, tt)
            w = np.where(tt.isin(withdraw_addrs), tt, ff)
            pairs.append(pd.DataFrame({"d": d, "w": w}))
pairs = pd.concat(pairs, ignore_index=True).drop_duplicates()
earliest_shield = sh.groupby("fr_l")["time"].min()
linked_d_for_w = pairs.groupby("w")["d"].apply(set).to_dict()


def _h2_causal(row):
    ds = linked_d_for_w.get(row["to_l"], ())
    for d in ds:
        es = earliest_shield.get(d)
        if pd.notna(es) and es < row["time"]:
            return True
    return False


h2 = un.apply(_h2_causal, axis=1)
print(f"  events: {int(h2.sum()):>6,}  ({100*h2.mean():5.2f}%)")

# H3
print("\nH3 (gas payer reuse, causal, kmeans-broadcaster excluded)")
un["amount"] = pd.to_numeric(un["amount"], errors="coerce").fillna(0.0)
un["output_notes"] = pd.to_numeric(un["output_notes"], errors="coerce").fillna(0.0)
un["is_self"] = (un["gp_l"] == un["to_l"]).astype(int)
gp_g = un.groupby("gp_l").agg(
    n_events=("transaction_hash", "count"),
    n_unique_recipients=("to_l", "nunique"),
    total_eth_volume=("amount", "sum"),
    mean_output_notes=("output_notes", "mean"),
    self_broadcast_rate=("is_self", "mean"),
    first_seen=("time", "min"),
    last_seen=("time", "max"),
)
gp_g["lifetime_days"] = ((gp_g["last_seen"] - gp_g["first_seen"])
                         .dt.total_seconds() / 86400.0)
gp_g = gp_g.drop(columns=["first_seen", "last_seen"]).reset_index()
feat = np.column_stack([
    np.log1p(gp_g["n_events"].values),
    np.log1p(gp_g["n_unique_recipients"].values),
    np.log1p(gp_g["total_eth_volume"].values.astype(float)),
    gp_g["mean_output_notes"].values.astype(float),
    gp_g["self_broadcast_rate"].values.astype(float),
    np.log1p(gp_g["lifetime_days"].values.astype(float)),
])
fs = StandardScaler().fit_transform(feat)
km = KMeans(n_clusters=3, random_state=42, n_init=20).fit(fs)
gp_g["cluster"] = km.labels_
bc_id = int(gp_g.groupby("cluster")["n_unique_recipients"].median().idxmax())
relayer_set = set(gp_g.loc[gp_g["cluster"] == bc_id, "gp_l"].tolist())
print(f"  broadcaster cluster size = {len(relayer_set):,}")

h3 = (un["gp_l"].isin(depositor_set)
      & (un["gp_l"] != un["to_l"])
      & ~un["gp_l"].isin(relayer_set)
      & un["earliest_sh_for_gp"].notna()
      & (un["earliest_sh_for_gp"] <= un["time"]))
print(f"  events: {int(h3.sum()):>6,}  ({100*h3.mean():5.2f}%)")

# Map each raw unshield to its w_agg_id for H4/H5
print("\nmapping raw unshields to w_agg_id ...")
tgt = pd.read_csv(H4_OUT / "1to1_targets.csv")
tgt["w_address_l"] = _norm(tgt["w_address"])
tgt["w_first_time"] = pd.to_datetime(tgt["w_first_time"], utc=True, errors="coerce")
tgt["w_last_time"] = pd.to_datetime(tgt["w_last_time"], utc=True, errors="coerce")
tgt_by_addr = {a: g[["w_agg_id", "w_first_time", "w_last_time"]].to_numpy()
               for a, g in tgt.groupby("w_address_l")}
agg_ids = np.full(N_un, -1, dtype=np.int64)
for i, (addr, t) in enumerate(zip(un["to_l"].to_numpy(), un["time"].to_numpy())):
    arr = tgt_by_addr.get(addr)
    if arr is None:
        continue
    for wid, t0, t1 in arr:
        if t0 <= t <= t1:
            agg_ids[i] = int(wid)
            break
un["w_agg_id"] = agg_ids
mapped = (agg_ids != -1).sum()
print(f"  mapped {mapped:,}/{N_un:,} raw unshields ({100*mapped/N_un:.1f}%)")

# H4
print("\nH4 (knapsack fully deanon., kto1 union over windows x buckets)")
H4_w = set()
for p in WIT_DIR.glob("witness_kto1_*.csv"):
    if p.stat().st_size < 500:
        continue
    d = pd.read_csv(p, usecols=["ti", "K_sampled", "H_knap_addr", "skipped"])
    d = d[(~d["skipped"]) & (d["K_sampled"] >= 1) & (d["H_knap_addr"] == 0)]
    H4_w |= set(d["ti"].astype(int).tolist())
h4 = un["w_agg_id"].isin(H4_w)
print(f"  events: {int(h4.sum()):>6,}  ({100*h4.mean():5.2f}%)  "
      f"[from {len(H4_w):,} super-rows]")

# H5
print("\nH5 (amount fingerprint, all regimes, tier unique)")
H5_w = set()
for regime in ["1to1", "2to1", "3to1"]:
    p = H4_OUT / f"{regime}_targets.csv"
    if not p.exists():
        continue
    d = pd.read_csv(p, usecols=["w_agg_id", "tier"])
    H5_w |= set(d.loc[d["tier"].astype(str).str.lower() == "unique",
                      "w_agg_id"].astype(int).tolist())
for regime in ["1to2", "1to3"]:
    pt = H4_OUT / f"{regime}_targets.csv"
    pl = H4_OUT / f"{regime}_links.csv"
    if not pt.exists() or not pl.exists():
        continue
    t = pd.read_csv(pt, usecols=["s_agg_id", "tier"])
    uniq_s = set(t.loc[t["tier"].astype(str).str.lower() == "unique",
                       "s_agg_id"].astype(int).tolist())
    cols = pd.read_csv(pl, nrows=0).columns
    w_cols = [c for c in cols if c.endswith("_agg_id") and c.startswith("w_")]
    l = pd.read_csv(pl, usecols=["s_agg_id"] + w_cols)
    l = l[l["s_agg_id"].astype(int).isin(uniq_s)]
    for c in w_cols:
        H5_w |= set(l[c].dropna().astype(int).tolist())
h5 = un["w_agg_id"].isin(H5_w)
print(f"  events: {int(h5.sum()):>6,}  ({100*h5.mean():5.2f}%)  "
      f"[from {len(H5_w):,} super-rows]")

# Unions
print(f"\n========== raw-event unions (denominator = {N_un:,}) ==========")


def show(name, mask):
    print(f"  {name:<32s} {int(mask.sum()):>6,}  ({100*mask.mean():5.2f}%)")


show("H1 only", h1)
show("H2 only", h2)
show("H3 only", h3)
show("H4 only", h4)
show("H5 only", h5)
print()
show("H1 U H2", h1 | h2)
show("H1 U H2 U H3", h1 | h2 | h3)
show("H1 U H2 U H3 U H4", h1 | h2 | h3 | h4)
show("H1 U H2 U H3 U H4 U H5", h1 | h2 | h3 | h4 | h5)

print("\npairwise intersection (raw events):")
names = ["H1", "H2", "H3", "H4", "H5"]
masks = [h1, h2, h3, h4, h5]
for i, a in enumerate(names):
    for j in range(i + 1, len(names)):
        b = names[j]
        inter = int((masks[i] & masks[j]).sum())
        print(f"  |{a} n {b}| = {inter:>6,}")
