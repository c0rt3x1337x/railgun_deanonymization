# A Tattered Cloak of Invisibility: Measuring Anonymity Loss in RAILGUN on Ethereum

Reference implementation for the empirical heuristics used in the paper.

From a user's perspective, perhaps the most significant difference between
traditional banking services and widely used blockchain-based financial
systems is that, in the latter, transactions and -- either directly or
indirectly -- account balances are publicly observable. A growing number of
cryptographic solutions have therefore been proposed to "add a privacy
layer" to such systems. However, the privacy that users actually obtain
does not depend solely on the security of the underlying cryptographic
protocol: user behavior, transaction-amount patterns, and timing decisions
can substantially reduce anonymity.

This repository accompanies our study of behavioral leakage in
cryptocurrency mixers, focusing on RAILGUN on Ethereum mainnet. We
heuristically estimate the probability that a given deposit and withdrawal
transaction belong to the same user. We consider four sources of leakage:
characteristic timing patterns, proximity in the transaction graph induced
by prior public transactions, amount fingerprints that preserve distinctive
digit patterns across transaction values, and knapsack-type matches in
which groups of transaction amounts add up in revealing ways. Our results
show that even cryptographically strong privacy systems may suffer
substantial anonymity loss due to user behavior and transaction patterns.

## Heuristics implemented

- **H1** -- address reuse: an unshield whose recipient also acted as a
  depositor at an earlier time.
- **H2** -- direct on-chain tx link: an unshield whose recipient shares an
  ETH or ERC-20 edge with a depositor.
- **H3** -- gas-payer reuse: a self-broadcast unshield (gas payer is not
  a community relayer) whose gas payer is also a depositor.
- **H4** -- knapsack witness enumeration: per-target Shannon entropy over
  the multiset of candidate address-level witnesses for subset-sum linkage,
  compared against the naive log2(pool addresses) baseline.

The filtered dataset covers 47,596 WETH unshield events and 34,747 WETH
shield events on Ethereum mainnet after removing protocol-internal and
swap-internal transactions.

## Layout

```
submission/
  README.md
  src/
    h1_address_reuse.py
    h2_direct_tx_link.py
    h3_gas_payer_reuse.py
    h4_knapsack/
      witness_kernel.c
      witness_entropy_pass.py
      build.sh
      run.sh
    compute_unions.py
    figures/
      make_png.py
      make_tikz.py
  data/
    transactions/    raw Dune exports (eth_shield, eth_unshields, eth-swaps,
                     eth_external2, eth_erc20)
    aggregated/      per-address aggregated shield / unshield tables and the
                     1to1_targets index used for the knapsack super-rows
  results/
    witness/         per-cell witness-enumeration output CSVs (H4 outputs)
```

## Reproduce

1. Install Python deps: `pip install numpy pandas scikit-learn matplotlib seaborn`
2. Build the C kernel: `cd src/h4_knapsack && bash build.sh`
3. Run each heuristic from the repository root:
   - `python src/h1_address_reuse.py`
   - `python src/h2_direct_tx_link.py`
   - `python src/h3_gas_payer_reuse.py`
   - `bash src/h4_knapsack/run.sh`
   - `python src/compute_unions.py`
4. Regenerate figures (optional, outputs to `results/figures/`):
   - `python src/figures/make_png.py`
   - `python src/figures/make_tikz.py`

## Citation

TBD

## License

TBD
