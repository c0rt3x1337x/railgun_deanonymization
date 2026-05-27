#include <stdint.h>
#include <string.h>

#define BIT(arr, i)  (((arr)[(i) >> 6] >> ((i) & 63)) & 1u)

static int32_t g_cap;
static int32_t g_found;
static int32_t* g_item_counts;
static int32_t* g_dfs_stack;
static int32_t g_depth;
static const int64_t* g_pool;
static const uint64_t* g_history;
static int64_t g_words;
static const int32_t* g_addr_ids;
static int32_t* g_addr_counts;
static int32_t* g_addr_seen_in;

static void dfs(int32_t i, int64_t residual) {
    if (g_found >= g_cap) return;
    if (residual == 0) {
        g_found++;
        for (int32_t k = 0; k < g_depth; ++k) {
            int32_t item = g_dfs_stack[k];
            g_item_counts[item]++;
            int32_t aid = g_addr_ids[item];
            if (g_addr_seen_in[aid] != g_found) {
                g_addr_seen_in[aid] = g_found;
                g_addr_counts[aid]++;
            }
        }
        return;
    }
    if (i == 0) return;
    int64_t v = g_pool[i - 1];
    const uint64_t* row = g_history + (size_t)(i - 1) * g_words;
    if (v > 0 && v <= residual && BIT(row, residual - v)) {
        g_dfs_stack[g_depth++] = i - 1;
        dfs(i - 1, residual - v);
        g_depth--;
        if (g_found >= g_cap) return;
    }
    if (BIT(row, residual)) {
        dfs(i - 1, residual);
    }
}

int32_t witness_enumerate(
    const int64_t* pool_b,
    int32_t n,
    int64_t T_b,
    int32_t cap,
    uint64_t* bitset_buf,
    int32_t* item_counts,
    int32_t* dfs_stack,
    const int32_t* addr_ids,
    int32_t n_addrs,
    int32_t* addr_counts,
    int32_t* addr_seen_in
) {
    if (n == 0 || cap <= 0 || T_b < 0) return 0;

    const int64_t words = (T_b / 64) + 1;
    memset(bitset_buf, 0, (size_t)(n + 1) * (size_t)words * sizeof(uint64_t));
    bitset_buf[0] = 1ULL;

    const int64_t top_word = T_b / 64;
    const int64_t top_bit  = T_b & 63;
    const uint64_t top_mask = (top_bit < 63)
        ? ((1ULL << (top_bit + 1)) - 1ULL)
        : (uint64_t)~0ULL;

    /* Forward DP: dp[i+1] = dp[i] | (dp[i] << v_i), masked to [0, T_b]. */
    for (int32_t i = 0; i < n; ++i) {
        uint64_t* prev = bitset_buf + (size_t)i * (size_t)words;
        uint64_t* curr = bitset_buf + (size_t)(i + 1) * (size_t)words;
        memcpy(curr, prev, (size_t)words * sizeof(uint64_t));
        int64_t v = pool_b[i];
        if (v <= 0 || v > T_b) continue;
        int64_t word_shift = v >> 6;
        int64_t bit_shift  = v & 63;
        if (bit_shift == 0) {
            for (int64_t j = words - 1; j >= word_shift; --j) {
                curr[j] |= prev[j - word_shift];
            }
        } else {
            int64_t cb = 64 - bit_shift;
            for (int64_t j = words - 1; j >= word_shift; --j) {
                uint64_t a = prev[j - word_shift] << bit_shift;
                uint64_t b = (j - word_shift > 0)
                    ? (prev[j - word_shift - 1] >> cb) : 0ULL;
                curr[j] |= (a | b);
            }
        }
        curr[top_word] &= top_mask;
        for (int64_t j = top_word + 1; j < words; ++j) curr[j] = 0;
    }

    uint64_t* row_n = bitset_buf + (size_t)n * (size_t)words;
    if (!BIT(row_n, T_b)) return 0;

    memset(item_counts, 0, (size_t)n * sizeof(int32_t));
    memset(addr_counts, 0, (size_t)n_addrs * sizeof(int32_t));
    memset(addr_seen_in, 0, (size_t)n_addrs * sizeof(int32_t));
    g_cap = cap;
    g_found = 0;
    g_item_counts = item_counts;
    g_dfs_stack = dfs_stack;
    g_depth = 0;
    g_pool = pool_b;
    g_history = bitset_buf;
    g_words = words;
    g_addr_ids = addr_ids;
    g_addr_counts = addr_counts;
    g_addr_seen_in = addr_seen_in;
    dfs(n, T_b);
    return g_found;
}
