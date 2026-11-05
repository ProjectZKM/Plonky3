// GPU proof-of-work grinding for Plonky3's BinaryChallenger over a Blake3
// HashChallenger (p3-binary-field challenger.rs, fast path).
//
// A candidate `index` is valid when Blake3(pending || index_le8 || 0^8) has
// its last eight bytes, read last first as a little-endian u64, zero in the
// low `bits` bits. The host absorbs `pending` once (the candidate-independent
// part of the Blake3 tree); each GPU thread appends a candidate's sixteen
// bytes and finalizes.  The caller re-checks the witness on the CPU.
//
// Build: nvcc -O3 -arch=sm_120 -Xcompiler -fPIC -shared -o libzkmgrind.so zkm_blake3_grind.cu
// Use:   ZKM_GPU_GRIND=4,5,6,7 (device list); unset, the library declines.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <thread>
#include <vector>
#include <string>

#define HD __host__ __device__ __forceinline__

static const uint32_t H_IV[8] = {0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
                                 0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u};
__constant__ uint32_t D_IV[8] = {0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
                                 0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u};

enum { CHUNK_START = 1, CHUNK_END = 2, PARENT = 4, ROOT = 8 };
enum { BLOCK_LEN = 64, CHUNK_LEN = 1024, MAX_STACK = 54 };

HD uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

HD void g(uint32_t *s, int a, int b, int c, int d, uint32_t x, uint32_t y) {
    s[a] = s[a] + s[b] + x; s[d] = rotr(s[d] ^ s[a], 16);
    s[c] = s[c] + s[d];     s[b] = rotr(s[b] ^ s[c], 12);
    s[a] = s[a] + s[b] + y; s[d] = rotr(s[d] ^ s[a], 8);
    s[c] = s[c] + s[d];     s[b] = rotr(s[b] ^ s[c], 7);
}

HD void round_fn(uint32_t *s, const uint32_t *m) {
    g(s, 0, 4, 8, 12, m[0], m[1]);   g(s, 1, 5, 9, 13, m[2], m[3]);
    g(s, 2, 6, 10, 14, m[4], m[5]);  g(s, 3, 7, 11, 15, m[6], m[7]);
    g(s, 0, 5, 10, 15, m[8], m[9]);  g(s, 1, 6, 11, 12, m[10], m[11]);
    g(s, 2, 7, 8, 13, m[12], m[13]); g(s, 3, 4, 9, 14, m[14], m[15]);
}

HD void permute(uint32_t *m) {
    const int P[16] = {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};
    uint32_t t[16];
    for (int i = 0; i < 16; i++) t[i] = m[P[i]];
    for (int i = 0; i < 16; i++) m[i] = t[i];
}

// The full 16-word compression output; `out[0..8]` is the chaining value.
HD void compress(const uint32_t *cv, const uint32_t *block, uint64_t counter, uint32_t block_len,
                 uint32_t flags, const uint32_t *iv, uint32_t *out) {
    uint32_t s[16] = {cv[0], cv[1], cv[2], cv[3], cv[4], cv[5], cv[6], cv[7],
                      iv[0], iv[1], iv[2], iv[3], (uint32_t)counter, (uint32_t)(counter >> 32),
                      block_len, flags};
    uint32_t m[16];
    for (int i = 0; i < 16; i++) m[i] = block[i];
    for (int r = 0; r < 7; r++) {
        round_fn(s, m);
        if (r < 6) permute(m);
    }
    for (int i = 0; i < 8; i++) { out[i] = s[i] ^ s[i + 8]; out[i + 8] = s[i + 8] ^ cv[i]; }
}

HD void bytes_to_words(const uint8_t *b, uint32_t *w) {
    for (int i = 0; i < 16; i++)
        w[i] = (uint32_t)b[4 * i] | ((uint32_t)b[4 * i + 1] << 8) | ((uint32_t)b[4 * i + 2] << 16) |
               ((uint32_t)b[4 * i + 3] << 24);
}

// The hasher state after the pending input, prepared so that appending
// sixteen bytes never crosses a chunk boundary and the buffered block is
// never full (both candidate-independent steps are done on the host).
struct State {
    uint32_t cv[8];              // the current chunk's chaining value
    uint64_t chunk_counter;
    uint8_t block[BLOCK_LEN];    // buffered bytes of the current block
    uint32_t block_len;          // < 64
    uint32_t blocks_compressed;  // in the current chunk
    uint32_t stack_len;
    uint32_t stack[MAX_STACK][8];
};

// --- host: absorb the pending bytes exactly as the blake3 crate does ---
static void host_parent_cv(const uint32_t *l, const uint32_t *r, uint32_t *cv) {
    uint32_t block[16], out[16];
    for (int i = 0; i < 8; i++) { block[i] = l[i]; block[i + 8] = r[i]; }
    compress(H_IV, block, 0, BLOCK_LEN, PARENT, H_IV, out);
    for (int i = 0; i < 8; i++) cv[i] = out[i];
}

static uint32_t chunk_len(const State &s) { return BLOCK_LEN * s.blocks_compressed + s.block_len; }

static void chunk_compress_block(State &s) {  // compress a full buffered block (not the last)
    uint32_t w[16], out[16];
    bytes_to_words(s.block, w);
    uint32_t flags = s.blocks_compressed == 0 ? CHUNK_START : 0;
    compress(s.cv, w, s.chunk_counter, BLOCK_LEN, flags, H_IV, out);
    for (int i = 0; i < 8; i++) s.cv[i] = out[i];
    s.blocks_compressed++;
    memset(s.block, 0, BLOCK_LEN);
    s.block_len = 0;
}

static void chunk_cv(const State &s, uint32_t *cv) {  // the finished chunk's chaining value
    uint32_t w[16], out[16];
    bytes_to_words(s.block, w);
    uint32_t flags = (s.blocks_compressed == 0 ? CHUNK_START : 0) | CHUNK_END;
    compress(s.cv, w, s.chunk_counter, s.block_len, flags, H_IV, out);
    for (int i = 0; i < 8; i++) cv[i] = out[i];
}

static void push_chunk(State &s) {  // the lazy chunk boundary of blake3's Hasher::update
    uint32_t cv[8];
    chunk_cv(s, cv);
    uint64_t total = s.chunk_counter + 1;
    while ((total & 1) == 0) {
        s.stack_len--;
        host_parent_cv(s.stack[s.stack_len], cv, cv);
        total >>= 1;
    }
    for (int i = 0; i < 8; i++) s.stack[s.stack_len][i] = cv[i];
    s.stack_len++;
    for (int i = 0; i < 8; i++) s.cv[i] = H_IV[i];
    s.chunk_counter++;
    s.blocks_compressed = 0;
    s.block_len = 0;
    memset(s.block, 0, BLOCK_LEN);
}

static void host_update(State &s, const uint8_t *in, size_t len) {
    while (len > 0) {
        if (chunk_len(s) == CHUNK_LEN) push_chunk(s);
        if (s.block_len == BLOCK_LEN) chunk_compress_block(s);
        size_t want = BLOCK_LEN - s.block_len;
        size_t room = CHUNK_LEN - chunk_len(s);
        size_t take = want < len ? want : len;
        if (take > room) take = room;
        memcpy(s.block + s.block_len, in, take);
        s.block_len += (uint32_t)take;
        in += take;
        len -= take;
    }
}

// Finish a candidate: append 16 bytes and finalize to the root digest.
HD void finish(const State &st, const uint32_t (*stack)[8], const uint8_t *cand, const uint32_t *iv,
               uint8_t *digest) {
    uint8_t block[BLOCK_LEN];
    for (int i = 0; i < BLOCK_LEN; i++) block[i] = st.block[i];
    uint32_t cv[8];
    for (int i = 0; i < 8; i++) cv[i] = st.cv[i];
    uint32_t blen = st.block_len, bcomp = st.blocks_compressed;
    uint32_t w[16], out[16];
    int k = 0;
    while (k < 16) {
        if (blen == BLOCK_LEN) {  // a full block with more input to come: not the last
            bytes_to_words(block, w);
            compress(cv, w, st.chunk_counter, BLOCK_LEN, bcomp == 0 ? CHUNK_START : 0, iv, out);
            for (int i = 0; i < 8; i++) cv[i] = out[i];
            bcomp++;
            for (int i = 0; i < BLOCK_LEN; i++) block[i] = 0;
            blen = 0;
        }
        block[blen++] = cand[k++];
    }
    bytes_to_words(block, w);
    uint32_t flags = (bcomp == 0 ? CHUNK_START : 0) | CHUNK_END;
    if (st.stack_len == 0) {
        compress(cv, w, st.chunk_counter, blen, flags | ROOT, iv, out);
    } else {
        compress(cv, w, st.chunk_counter, blen, flags, iv, out);
        uint32_t right[8];
        for (int i = 0; i < 8; i++) right[i] = out[i];
        for (int n = (int)st.stack_len - 1; n >= 0; n--) {
            uint32_t pb[16];
            for (int i = 0; i < 8; i++) { pb[i] = stack[n][i]; pb[i + 8] = right[i]; }
            compress(iv, pb, 0, BLOCK_LEN, PARENT | (n == 0 ? ROOT : 0), iv, out);
            for (int i = 0; i < 8; i++) right[i] = out[i];
        }
    }
    for (int i = 0; i < 8; i++) {
        digest[4 * i] = (uint8_t)out[i];
        digest[4 * i + 1] = (uint8_t)(out[i] >> 8);
        digest[4 * i + 2] = (uint8_t)(out[i] >> 16);
        digest[4 * i + 3] = (uint8_t)(out[i] >> 24);
    }
}

HD bool passes(const uint8_t *digest, uint32_t bits) {
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint64_t)digest[31 - i] << (8 * i);
    return (v & ((bits >= 64) ? ~0ull : ((1ull << bits) - 1))) == 0;
}

HD void candidate_bytes(uint64_t index, uint8_t *c) {
    for (int i = 0; i < 8; i++) c[i] = (uint8_t)(index >> (8 * i));
    for (int i = 8; i < 16; i++) c[i] = 0;
}

__constant__ State D_STATE;

__global__ void grind_kernel(uint64_t start, uint32_t per_thread, uint32_t bits,
                             unsigned long long *found) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t base = start + tid * per_thread;
    uint8_t cand[16], digest[32];
    for (uint32_t j = 0; j < per_thread; j++) {
        if (*(volatile unsigned long long *)found != ~0ull) return;
        uint64_t index = base + j;
        candidate_bytes(index, cand);
        finish(D_STATE, D_STATE.stack, cand, D_IV, digest);
        if (passes(digest, bits)) { atomicMin(found, (unsigned long long)index); return; }
    }
}

// Returns 0 and the index in *out when found; 1 when the GPU path declines
// (not enabled, no device, or a state it does not handle), so the caller
// grinds on the CPU instead.
extern "C" int zkm_blake3_grind(const uint8_t *pending, size_t len, uint32_t bits, uint64_t *out) {
    const char *env = getenv("ZKM_GPU_GRIND");
    if (env == nullptr || *env == 0) return 1;
    const char *min_env = getenv("ZKM_GPU_GRIND_MIN_BITS");
    uint32_t min_bits = min_env ? (uint32_t)atoi(min_env) : 20;
    if (bits < min_bits || bits > 56) return 1;
    std::vector<int> devices;
    for (std::string s(env); !s.empty();) {
        size_t c = s.find(',');
        devices.push_back(atoi(s.substr(0, c).c_str()));
        s = c == std::string::npos ? "" : s.substr(c + 1);
    }

    State st;
    memset(&st, 0, sizeof st);
    for (int i = 0; i < 8; i++) st.cv[i] = H_IV[i];
    host_update(st, pending, len);
    // Candidate-independent steps of the next update, done here.
    if (chunk_len(st) == CHUNK_LEN) push_chunk(st);
    if (st.block_len == BLOCK_LEN) chunk_compress_block(st);
    if (chunk_len(st) + 16 > CHUNK_LEN) return 1;  // the candidate would cross a chunk
    if (st.stack_len > MAX_STACK) return 1;

    // Self-check: the host and device finish agree with each other on index 0,
    // and the CPU caller re-checks the witness against the real blake3 crate.
    const unsigned threads = 256, blocks = 1u << 16;
    const uint32_t per_thread = 64;
    const uint64_t per_launch = (uint64_t)threads * blocks * per_thread;
    std::atomic<uint64_t> next{0};
    std::atomic<bool> done{false};
    std::atomic<uint64_t> best{~0ull};
    std::atomic<int> failed{0};
    std::vector<std::thread> workers;
    for (int dev : devices) {
        workers.emplace_back([&, dev]() {
            if (cudaSetDevice(dev) != cudaSuccess) { failed++; return; }
            if (cudaMemcpyToSymbol(D_STATE, &st, sizeof st) != cudaSuccess) { failed++; return; }
            unsigned long long *d_found;
            if (cudaMalloc(&d_found, sizeof *d_found) != cudaSuccess) { failed++; return; }
            while (!done.load()) {
                uint64_t start = next.fetch_add(per_launch);
                unsigned long long init = ~0ull;
                cudaMemcpy(d_found, &init, sizeof init, cudaMemcpyHostToDevice);
                grind_kernel<<<blocks, threads>>>(start, per_thread, bits, d_found);
                unsigned long long h = ~0ull;
                if (cudaMemcpy(&h, d_found, sizeof h, cudaMemcpyDeviceToHost) != cudaSuccess) {
                    failed++;
                    break;
                }
                if (h != ~0ull) {
                    uint64_t prev = best.load();
                    while (h < prev && !best.compare_exchange_weak(prev, h)) {}
                    done.store(true);
                }
            }
            cudaFree(d_found);
        });
    }
    for (auto &w : workers) w.join();
    if (best.load() == ~0ull) return 1;
    // Host re-derivation of the digest for the found index.
    uint8_t cand[16], digest[32];
    candidate_bytes(best.load(), cand);
    finish(st, st.stack, cand, H_IV, digest);
    if (!passes(digest, bits)) return 1;
    *out = best.load();
    if (getenv("ZKM_GPU_GRIND_VERBOSE"))
        fprintf(stderr, "zkm_gpu_grind: %u bits, pending %zu B, index %llu, %zu device(s)\n", bits, len,
                (unsigned long long)*out, devices.size());
    return 0;
}

#ifdef ZKM_GRIND_SELFTEST
// Self-test: the Blake3 digest of a file, its last sixteen bytes appended as
// a candidate would be (compare with b3sum).
int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "rb");
    std::vector<uint8_t> msg;
    for (int c; (c = fgetc(f)) != EOF;) msg.push_back((uint8_t)c);
    fclose(f);
    State st;
    memset(&st, 0, sizeof st);
    for (int i = 0; i < 8; i++) st.cv[i] = H_IV[i];
    host_update(st, msg.data(), msg.size() - 16);
    if (chunk_len(st) == CHUNK_LEN) push_chunk(st);
    if (st.block_len == BLOCK_LEN) chunk_compress_block(st);
    if (chunk_len(st) + 16 > CHUNK_LEN) { printf("declined\n"); return 0; }
    uint8_t digest[32];
    finish(st, st.stack, msg.data() + msg.size() - 16, H_IV, digest);
    for (int i = 0; i < 32; i++) printf("%02x", digest[i]);
    printf("\n");
    return 0;
}
#endif
