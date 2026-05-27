# Data

Raw transaction-level exports of RAILGUN activity on Ethereum mainnet,
obtained via Dune Analytics queries against the protocol's public
contracts (Railgun Relay `0xfa7093cd...4b9` and the V2 RelayAdapt
`0xAc9f360a...0405`).

## transactions/

| File | Purpose |
|------|---------|
| `eth_shield.csv` | Shield (deposit) events |
| `eth_unshields.csv` | Unshield (withdrawal) events with UTXO metadata |
| `eth-swaps.csv` | Private DEX swaps via RelayAdapt |
| `eth_external2.csv` | External ETH transfers between candidate addresses |
| `eth_erc20.csv` | ERC-20 transfers between candidate addresses |

## aggregated/

| File | Purpose |
|------|---------|
| `eth_shield_aggregated.csv` | Per-address aggregated shield super-rows |
| `eth_unshields_aggregated.csv` | Per-address aggregated unshield super-rows |
| `1to1_targets.csv` | Canonical `w_agg_id` index for the unshield super-row used by H4 |

Override the default directories via environment variables:

```bash
export RAILGUN_TX_DIR=/alt/path/to/transactions
export RAILGUN_AGG_DIR=/alt/path/to/aggregated
```
