// Write-protect dirty-page tracking for guest memory: pool pages of packaged textures (CPU writes, GPU reads) are set read-only after upload, and as soon as the game writes it faults -> mark dirty, unprotect;
// before use, check the write generation of the covered pages and re-upload if newer than at upload time. NVN semantics: a texture descriptor points to pool memory, and the GPU reads the current content every time
// (measured: the game writes streamed assets into texture storage that is still being sampled, without re-registering).
// Signal handler constraint: the page-fault path uses only atomic operations and mprotect, no allocation, no locks.
#include "nvn_dirty.h"
#include "../../host/dm.h"
#include "../../host/dm_uc.h"   // DM_UC_SS: the faulting PC/LR/FP from the signal context
#include <stdatomic.h>
#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>
#include <mach/mach_time.h>
#include <mach/mach.h>   // vm_read_overwrite: reads that fail instead of faulting (process_vm_readv on Android)
#include <sched.h>

// Order of page state and mprotect: the side changing protection first sets the state to BUSY and sets the final state only after mprotect has landed; other threads wait when they see BUSY.
//   set read-only: NONE -> BUSY -> mprotect(read-only) -> PROT; unprotect: PROT -> BUSY -> mprotect(writable) -> NONE.
// Invariant: a page whose state is NONE is always writable, one whose state is PROT is always read-only. Setting read-only used to be "record PROT, then mprotect"; in between, another thread's unprotect-before-write
// (the movie decoder thread writing a frame into texture storage, a disk read into a guest buffer) could set the state back to NONE, then the mprotect landed => the page read-only with state NONE: a game thread writing it got only
// "re-execute" from the fault handler and spun forever; a host system call reading into it saw NONE in prep_write, skipped, and failed with EFAULT (reproduced by t_dirty_race).
// BUSY lasts at most one mprotect (plus a few atomic writes when setting read-only). The wait is capped at BUSY_WAIT_MS and reports red beyond it: in the fault handler it then returns "not our page" so the crash handler reports truthfully
enum { PG_NONE = 0, PG_PROT = 1, PG_BUSY = 2 };
enum { BUSY_WAIT_MS = 2000 };
// Tracked ranges = the memory pools (registered by MemoryPoolInitialize). After a pool is destroyed its live=0 slot is reused; the signal handler scans without locks.
// The table grows in blocks of 1,024 slots, up to 262,144; a block is never moved or freed, so the signal handler reads it without locks. A pool past the end goes untracked: its textures count
// as dirty and are re-uploaded at every use. The table was a fixed 256 slots (Zelda has about 300 pools at the title), then a fixed 4,096, and Breath of the Wild reaches 4,006 pools in its first
// shrine. The scans only cover slots in use, so the capacity itself costs nothing
enum { BLK_SHIFT = 10, BLK = 1 << BLK_SHIFT, NBLK = 256, MAX_RANGE = BLK * NBLK };
// Ranges are rounded to host pages (16 KB) but guest memory pools are only 4 KB aligned: adjacent pools share their first / last pages, so one address can fall in several registered ranges.
// So: "is this texture tracked" looks for a range that holds its whole span; a fault / host pre-write unprotects and stamps every range covering that page; pool destruction identifies its range by the pool's exact start mem.
// (Previously the first range containing the address was always taken: two SMBW ASTC 6x5 textures whose pool's first page is shared with a small preceding pool were judged untracked and re-uploaded on every use, about 6 MB per frame)
// span: per page, 1 + the first page of the nvn_dirty_protect call that protected it (one texture's storage); 0 when not protected by one
// last: the highest generation of any page of the range (a check of a range nothing was written in since its stamp scans no page)
// nmp: this module's mprotect calls on the range since it was registered or defragmented (nvn_dirty_defrag)
// prep: when the host last made pages of the range writable for a write of its own (prep_write; mach time)
typedef struct { uintptr_t base, end, mem; _Atomic uint8_t *state; _Atomic uint64_t *gen; _Atomic uint32_t *span; size_t cap; _Atomic int live; _Atomic uint32_t nmp; _Atomic uint64_t prep; _Atomic uint64_t last; } range_t;
static _Atomic(range_t *) g_blk[NBLK];
static inline range_t *rg(int i) { return &atomic_load_explicit(&g_blk[i >> BLK_SHIFT], memory_order_acquire)[i & (BLK - 1)]; }
static _Atomic int g_nr;
static uintptr_t g_pg, g_shift;
static _Atomic uint64_t g_now = 1;
// A page written (made writable for a write): its new generation, then the range's highest (a compare-and-swap maximum: signal-safe, and a slower thread never lowers it)
static inline void stamp_page(range_t *r, uintptr_t p) {
    uint64_t g = atomic_fetch_add(&g_now, 1) + 1; atomic_store(&r->gen[p], g);
    uint64_t o = atomic_load(&r->last); while (o < g && !atomic_compare_exchange_weak(&r->last, &o, g)) {}
}
static _Atomic uint64_t g_faults, g_preps, g_released;
static _Atomic uint64_t g_adds_total;
static _Atomic int g_live, g_live_peak;   // number of registered pools and peak (for the memory readout line)
void nvn_dirty_stat(int *live, int *peak, int *cap) { *live = atomic_load(&g_live); *peak = atomic_load(&g_live_peak); *cap = MAX_RANGE; }
static _Atomic int g_overflow_logged;
static pthread_mutex_t g_slot_mu = PTHREAD_MUTEX_INITIALIZER;   // Pool creation / destruction (guest threads, possibly concurrent) pick slots under a mutex; the page-fault handling path does not touch it
static _Atomic uint64_t g_mp_calls, g_mp_pages, g_mp_ticks;   // This module's mprotect: call count, pages covered, time spent (mach timebase, safe in the page-fault path too)
static _Atomic int g_mp_red;
static int mp(uintptr_t a, size_t len, int prot) {
    uint64_t t = mach_absolute_time(); int rc = mprotect((void *)a, len, prot);
    atomic_fetch_add(&g_mp_ticks, mach_absolute_time() - t); atomic_fetch_add(&g_mp_calls, 1); atomic_fetch_add(&g_mp_pages, len >> g_shift);
    if (rc && !atomic_exchange(&g_mp_red, 1)) LOG("[RED] dirty-page tracking: mprotect %p +%#zx failed (the process's mapping limit?)", (void *)a, len);
    return rc;
}

static mach_timebase_info_data_t g_tb;   // obtained when the first pool is registered (the fault path only multiplies and divides)
static _Atomic int g_busy_red;
// Wait for this page to leave BUSY. Returns 1 = it has left; 0 = waited past the limit (reported red once). Usable in a signal handler: only atomic reads, mach_absolute_time and sched_yield
static int busy_wait(range_t *r, uintptr_t p) {
    uint64_t t0 = 0;
    for (unsigned n = 1; atomic_load(&r->state[p]) == PG_BUSY; n++) {
        if (n & 0x3ff) continue;
        uint64_t now = mach_absolute_time(); if (!t0) t0 = now;
        if ((now - t0) * g_tb.numer / g_tb.denom > (uint64_t)BUSY_WAIT_MS * 1000000ull) {
            if (!atomic_exchange(&g_busy_red, 1)) LOG("[RED] dirty-page tracking: page %p stuck 'busy' for over %d ms (the thread changing its protection has not finished), no longer waiting", (void *)(r->base + (p << g_shift)), BUSY_WAIT_MS);
            return 0; }
        sched_yield();
    }
    return 1;
}
static int live_at(int i) { return atomic_load_explicit(&rg(i)->live, memory_order_acquire); }
// The registered range that holds all of [a, a+len) (used for a texture's tracking / protection / dirty check)
static range_t *find_span(uintptr_t a, size_t len) {
    int n = atomic_load(&g_nr);
    for (int i = 0; i < n; i++) { range_t *r = rg(i); if (live_at(i) && a >= r->base && a + len <= r->end) return r; }
    return NULL;
}
static int unprotect_page(range_t *r, uintptr_t p) {   // The caller has already changed the state from PROT to BUSY. 0 = unprotected; else it stays protected
    atomic_fetch_add(&r->nmp, 1);
    if (mp(r->base + (p << g_shift), g_pg, PROT_READ | PROT_WRITE)) { atomic_store(&r->state[p], PG_PROT); return -1; }
    stamp_page(r, p);
    atomic_store(&r->state[p], PG_NONE);
    return 0;
}
// Unprotect one page of a range: if protected, unprotect and stamp it; if another thread is changing its protection, wait for it to finish and look again -- if the other side was setting read-only, the page is PROT after the wait and still has to be unprotected
// (an unprotect-before-write never reaches the fault handler, so it must be done here). Returns 1 = this page is managed by this module; 0 = it is not, or the wait on BUSY timed out
static int release_page(range_t *r, uintptr_t p, _Atomic uint64_t *count) {
    for (;;) {
        uint8_t st = PG_PROT;
        if (atomic_compare_exchange_strong(&r->state[p], &st, PG_BUSY)) { if (unprotect_page(r, p)) return 0;   // a write let through is the only way on: not ours, the crash handler reports it
            if (count) atomic_fetch_add(count, 1); return 1; }
        if (st != PG_BUSY) return atomic_load(&r->gen[p]) != 0;   // already NONE: just unprotected by another thread (the fault happened before the unprotect) -> re-execute; pages never protected are not ours
        if (!busy_wait(r, p)) return 0;
    }
}
static void release_pages(range_t *r, uintptr_t p, uintptr_t q, _Atomic uint64_t *count);
// ---------- Write watch on a rectangle (DM_WATCH_RECT) ----------
// Diagnostic: find where a game's view rectangle lives and who writes it. DM_WATCH_RECT=<x0,y0,x1,y1> (four int32, e.g. 160,0,1440,720) scans guest
// memory once, at one present (DM_WATCH_RECT_AT, default 30), for the 16-byte little-endian pattern, and write-protects every writable page holding a
// match through the same ranges and fault path as texture tracking (a watched page is a tracked range: a game write faults, on_fault records the writer
// and lets it through as any other). The fault handler only records (signal context: atomics, plain stores and fail-instead-of-fault reads, no logging);
// dm_watch_tick, on the present thread, logs each hit once with the writer's PC and LR, the frame-pointer chain (so a hit in a generic stp/memcpy
// still names the caller a few frames up) and the values before and after the write. Each address records up to WATCH_HITS writes: after a hit is
// logged the tick re-arms the page so the next write faults again, until the ring is full (short arm). A page the tracker releases without a fault (a
// host write into it, a texture sweep over it) is re-protected by the tick. The scan reads /proc/self/maps, so it runs on Linux (Android) only;
// elsewhere the tick logs that and arms nothing. DM_WATCH_SCAN_ONLY=1 turns the watch into a read-only probe: the scan still runs and each match is
// still logged, but instead of arming, the 128 qwords before each match (0x400 bytes back) and the 16 after it are dumped as `scan:` lines, so the
// enclosing object's vtable pointer can be found offline. Nothing is write-protected, so the arming race that crashed the game twice cannot happen;
// the game continues normally afterwards.
enum { WATCH_MAX = 16, WATCH_HITS = 10, WATCH_FRAMES = 8 };
typedef struct {   // one recorded write: the writer's PC and LR, the frame-pointer chain (return addresses, [fp+8] per frame), the values before the write
    uint64_t pc, lr, fp[WATCH_FRAMES]; int nfp; int32_t old[4]; _Atomic int ready;   // ready: set (release) once the rest is written
} watch_hit_t;
typedef struct { uintptr_t addr; _Atomic int claimed, logged; watch_hit_t hits[WATCH_HITS]; } watch_t;   // claimed: next slot to claim; logged: next slot to log
static watch_t g_watch[WATCH_MAX];
static _Atomic int g_watch_n;   // watches armed; 0 = the fault path skips the check
static int g_watch_scan_only;   // DM_WATCH_SCAN_ONLY=1: dump the context around each match, never arm (read-only probe)
// Walk the frame-pointer chain from the faulting context: up to max return addresses ([fp+8] of each frame). Signal context: the reads go through
// vm_read_overwrite (fails instead of faulting on an unmapped frame), the loop is bounded, and every frame is sanity-checked (8-byte aligned, strictly
// increasing, within 64 KB of the previous) so a garbage chain stops the walk instead of crashing the game. Returns the frames found.
static int watch_fp_chain(void *uc, uint64_t *out, int max) {
    uintptr_t fp = (uintptr_t)DM_UC_SS((ucontext_t *)uc)->__fp;
    int n = 0;
    while (n < max && fp) {
        if (fp & 7) break;   // not 8-byte aligned: not a frame pointer
        uint64_t pair[2]; vm_size_t got = 0;
        if (vm_read_overwrite(mach_task_self(), (vm_address_t)fp, 16, (vm_address_t)pair, &got) != KERN_SUCCESS || got != 16) break;   // unreadable frame: stop
        uintptr_t next = (uintptr_t)pair[0];
        if (next && (next <= fp || next - fp > 64 * 1024)) break;   // the chain walks up the stack, one frame at a time
        out[n++] = pair[1];   // the caller's return address
        fp = next;
    }
    return n;
}
// A write fault on a watched address: claim a hit slot and record the writer's PC and LR, the frame-pointer chain, and the values before the write
// (the store has not landed yet). Signal context: atomics, plain stores and fail-instead-of-fault reads only; the present thread logs it (dm_watch_tick).
static void watch_fault(uintptr_t a, void *uc) {
    int n = atomic_load_explicit(&g_watch_n, memory_order_acquire);
    if (!n) return;
    for (int i = 0; i < n; i++) {
        watch_t *w = &g_watch[i];
        if (a < w->addr || a >= w->addr + sizeof w->hits[0].old) continue;
        int slot = atomic_fetch_add_explicit(&w->claimed, 1, memory_order_relaxed);
        if (slot >= WATCH_HITS) return;   // the ring is full: no more recording
        watch_hit_t *h = &w->hits[slot];
        struct __darwin_arm_thread_state64 *ss = DM_UC_SS((ucontext_t *)uc);
        h->pc = ss->__pc;
        h->lr = ss->__lr;
        h->nfp = watch_fp_chain(uc, h->fp, WATCH_FRAMES);
        memcpy(h->old, (const void *)w->addr, sizeof h->old);   // the page is read-only: reads are fine
        atomic_store_explicit(&h->ready, 1, memory_order_release);
        return;
    }
}
static int on_fault(void *addr, int is_write, void *uc) {
    if (!is_write) return 0;
    watch_fault((uintptr_t)addr, uc);
    uintptr_t a = (uintptr_t)addr; int ours = 0, n = atomic_load(&g_nr);
    for (int i = 0; i < n; i++) {   // Every range sharing the page must be released: releasing only one leaves the other range's state at "protected", and later writes no longer fault and are missed
        range_t *r = rg(i); if (!live_at(i) || a < r->base || a >= r->end) continue;
        uintptr_t pg = (a - r->base) >> g_shift; uint32_t sp = atomic_load(&r->span[pg]);
        ours |= release_page(r, pg, &g_faults);
        // The rest of the storage the same protect call covered goes with it: the texture is uploaded again after any write into it, so this costs no upload, and the
        // game's writes into it fault once instead of once a page. Released a page at a time, a texture written at every other page left its storage in alternating
        // read-only and writable pages, each a mapping of its own: Xenoblade Chronicles 3 reached the kernel's mapping limit (65,530) in its first battle, after which
        // mprotect failed, a faulting write was never let through, and the game hung (or an UnmapPhysicalMemory failed with ENOMEM)
        if (sp) { uintptr_t q = sp - 1; while (q < r->cap && q < ((r->end - r->base) >> g_shift) && atomic_load(&r->span[q]) == sp) q++; if (q > sp - 1) release_pages(r, sp - 1, q - 1, &g_faults); }
    }
    return ours;
}
// Unprotect pages p..q of a range with one mprotect per run of protected pages (a host write over a whole texture, e.g. WriteTexels, used to make one system call per 16 KB page:
// Link's Awakening, about 275 a frame). Pages another thread is changing are handled one by one after the run so far is finished, so no BUSY page is held while waiting
static void release_pages(range_t *r, uintptr_t p, uintptr_t q, _Atomic uint64_t *count) {
    uintptr_t run = 0; int open = 0;
    for (uintptr_t i = p; i <= q + 1; i++) {
        uint8_t st = PG_PROT; int took = i <= q && atomic_compare_exchange_strong(&r->state[i], &st, PG_BUSY);
        if (took) { if (!open) { run = i; open = 1; } continue; }
        if (open) {
            atomic_fetch_add(&r->nmp, 1);
            if (mp(r->base + (run << g_shift), (i - run) << g_shift, PROT_READ | PROT_WRITE)) {
                // A failed mprotect may change only part of the run; neither writable states nor a retry are safe. Also called from the fault handler.
                static const char msg[] = "[dm] [RED] dirty-page tracking: cannot make guest memory writable; stopping\n";
                (void)write(STDERR_FILENO, msg, sizeof msg - 1); _exit(1);
            }
            for (uintptr_t k = run; k < i; k++) { stamp_page(r, k); atomic_store(&r->state[k], PG_NONE); }
            if (count) atomic_fetch_add(count, i - run);
            open = 0;
        }
        if (i <= q && st == PG_BUSY) release_page(r, i, count);   // another thread is changing it: wait, then unprotect it if it ended up protected
    }
}
static void release_range(const void *addr, size_t len, _Atomic uint64_t *count) {
    uintptr_t a = (uintptr_t)addr, e = a + len; int n = atomic_load(&g_nr);
    if (!len) return;
    for (int i = 0; i < n; i++) {
        range_t *r = rg(i);
        if (!live_at(i) || e <= r->base || a >= r->end) continue;
        uintptr_t lo = a > r->base ? a : r->base, hi = e < r->end ? e : r->end;
        if (count == &g_preps) atomic_store(&r->prep, mach_absolute_time());
        release_pages(r, (lo - r->base) >> g_shift, (hi - 1 - r->base) >> g_shift, count);
    }
}
static void prep_write(void *addr, size_t len) { release_range(addr, len, &g_preps); }
// A caller that no longer needs a span watched (a packaged texture not drawn for a while, gfx_vk_tex.c vk_tex_sweep) gives up its protection: the pages are stamped as written, so
// the texture is uploaded and protected again when it is drawn. Textures stay protected until the game writes them, so the protected runs, each splitting its pool's mapping,
// would grow with every place the player has seen
void nvn_dirty_release(const void *addr, size_t len) { if (len) release_range(addr, len, &g_released); }
uint64_t nvn_dirty_released(void) { return atomic_load(&g_released); }
// Defragmenting: the pools' pages are a dma-buf's (gfx_vk_core.c zero copy), and the kernel never merges such mappings again, so every protect or unprotect boundary
// ever made stayed a mapping of its own whatever the protection was now (Xenoblade Chronicles 3: 10,600 read-only mappings with 475 runs of pages protected, 43,000 in
// all after 25 minutes of walking, toward the kernel's limit of 65,530). A range with more than DEFRAG_MP mprotect calls since the last time is mapped again in one piece:
// every page is made BUSY (a game write faulting on one waits), remap() maps the whole range read-only in one call, the runs that were writable are made writable again
// (and stamped as written, so a write that waited is the module's to let through), and the protected pages keep their protection. The mappings are then about twice the
// runs. Run on the thread that protects textures (the replay's); remap() returns 0 when it mapped the range.
// A range the host prepared a write into in the last DEFRAG_QUIET_MS is left for later: the host's own writes (a file read into a pool, which the game does for its
// streaming) are system calls the kernel fails with EFAULT on a page made read-only under them; Xenoblade Chronicles 3 then inflated a short read and crashed in zlib
enum { DEFRAG_MP = 4000, DEFRAG_QUIET_MS = 1000 };
int nvn_dirty_defrag(int (*remap)(void *mem, size_t len)) {
    int done = 0, n = atomic_load(&g_nr);
    for (int i = 0; i < n; i++) {
        range_t *r = rg(i); if (!live_at(i) || atomic_load(&r->nmp) < DEFRAG_MP || r->mem != r->base) continue;   // a pool starting mid-page shares that page: left alone
        if ((mach_absolute_time() - atomic_load(&r->prep)) * g_tb.numer / g_tb.denom < (uint64_t)DEFRAG_QUIET_MS * 1000000ull) continue;
        size_t np = (r->end - r->base) >> g_shift;
        uint8_t *was = malloc(np); if (!was) continue;
        for (size_t p = 0; p < np; p++) {   // every page BUSY; was = its state before
            for (;;) { uint8_t st = atomic_load(&r->state[p]);
                if (st == PG_BUSY) { if (!busy_wait(r, p)) break; continue; }
                if (atomic_compare_exchange_strong(&r->state[p], &st, PG_BUSY)) { was[p] = st; break; } }
        }
        int ok = remap((void *)r->base, r->end - r->base) == 0;
        for (size_t p = 0; ok && p < np; ) {   // the pages that were writable, a run at a time
            if (was[p] != PG_NONE) { p++; continue; }
            size_t q = p; while (q < np && was[q] == PG_NONE) q++;
            mp(r->base + (p << g_shift), (q - p) << g_shift, PROT_READ | PROT_WRITE); p = q;
        }
        for (size_t p = 0; p < np; p++) {
            if (!ok || was[p] == PG_PROT) { atomic_store(&r->state[p], was[p]); continue; }
            stamp_page(r, p); atomic_store(&r->state[p], PG_NONE);
        }
        free(was);
        if (ok) { atomic_store(&r->nmp, 0); done++; }
    }
    return done;
}
// Runs of protected pages over every live range: each is a mapping of its own (at most two with the writable pieces around it)
int nvn_dirty_runs(void) {
    int n = atomic_load(&g_nr), runs = 0;
    for (int i = 0; i < n; i++) {
        range_t *r = rg(i); if (!live_at(i)) continue;
        size_t np = (r->end - r->base) >> g_shift; int prev = 0;
        for (size_t p = 0; p < np; p++) { int pr = atomic_load_explicit(&r->state[p], memory_order_relaxed) != PG_NONE; runs += pr && !prev; prev = pr; }
    }
    return runs;
}

void nvn_dirty_add_range(const void *mem, size_t size) {
    static int inited;
    if (!inited) { inited = 1; g_pg = (uintptr_t)getpagesize(); g_shift = __builtin_ctzl(g_pg); mach_timebase_info(&g_tb); dm_fault_hook_register(on_fault); dm_guest_write_prep_register(prep_write); }
    atomic_fetch_add(&g_adds_total, 1);
    pthread_mutex_lock(&g_slot_mu);
    int n = atomic_load(&g_nr);
    int slot = -1;
    for (int i = 0; i < n; i++) if (!live_at(i)) { slot = i; break; }
    if (slot < 0) {
        if (n >= MAX_RANGE) {
            int expected = 0;
            if (atomic_compare_exchange_strong(&g_overflow_logged, &expected, 1)) LOG("[RED] dirty-page tracking ranges are full (%d), pool %p not tracked", MAX_RANGE, mem);
            pthread_mutex_unlock(&g_slot_mu); return;
        }
        slot = n;
        if (!atomic_load_explicit(&g_blk[n >> BLK_SHIFT], memory_order_acquire)) {   // published before g_nr counts the slot (the scans read g_nr first)
            range_t *b = calloc(BLK, sizeof *b); if (!b) die("dirty-page tracking range table allocation failed (block %d)", n >> BLK_SHIFT);
            atomic_store_explicit(&g_blk[n >> BLK_SHIFT], b, memory_order_release);
        }
    }
    { int lv = atomic_fetch_add(&g_live, 1) + 1, pk = atomic_load(&g_live_peak); while (lv > pk && !atomic_compare_exchange_weak(&g_live_peak, &pk, lv)) {} }
    range_t *r = rg(slot);
    uintptr_t base = (uintptr_t)mem & ~(g_pg - 1), end = ((uintptr_t)mem + size + g_pg - 1) & ~(g_pg - 1);
    size_t np = (end - base) >> g_shift;
    _Atomic uint8_t *state = r->state;
    _Atomic uint64_t *gen = r->gen;
    if (r->cap < np) {
        state = calloc(np, 1); gen = calloc(np, sizeof *gen); r->span = calloc(np, sizeof *r->span);
        if (!state || !gen || !r->span) die("dirty-page table allocation failed (%zu pages)", np);
        // The old array is not freed: the signal handler may still have a very brief stale reader; leak = 9 bytes per page.
        r->cap = np;
    } else {
        memset((void *)state, 0, np); memset((void *)gen, 0, np * sizeof *gen); memset((void *)r->span, 0, np * sizeof *r->span);
    }
    r->base = base; r->end = end; r->mem = (uintptr_t)mem; r->state = state; r->gen = gen; atomic_store(&r->nmp, 0); atomic_store(&r->prep, 0); atomic_store(&r->last, 0);
    if (slot == n) atomic_store(&g_nr, n + 1);
    atomic_store_explicit(&r->live, 1, memory_order_release);
    pthread_mutex_unlock(&g_slot_mu);
}
// Pool destruction: first restore still-protected pages to writable (the pool memory may be reused for something else once returned to the game), then mark invalid
void nvn_dirty_remove_range(const void *mem) {
    pthread_mutex_lock(&g_slot_mu);
    range_t *r = NULL; int n = atomic_load(&g_nr);
    for (int i = 0; i < n && !r; i++) if (live_at(i) && rg(i)->mem == (uintptr_t)mem) r = rg(i);   // Match by the pool's exact start: an adjacent pool sharing a page may have the same rounded base
    if (r) { prep_write((void *)r->base, r->end - r->base); atomic_store_explicit(&r->live, 0, memory_order_release); atomic_fetch_sub(&g_live, 1); }
    pthread_mutex_unlock(&g_slot_mu);
}
int nvn_dirty_tracked(const void *addr, size_t len) {
    return find_span((uintptr_t)addr, len) != NULL;
}
// The tracked range holding [addr, addr+len), as a handle a caller keeps (packaged textures: the scan over every range ran twice per texture lookup). NULL = untracked
const void *nvn_dirty_span(const void *addr, size_t len) { return len ? find_span((uintptr_t)addr, len) : NULL; }
// Set read-only: mark the pages BUSY first and mark them PROT only after mprotect has landed (for the order and the invariant see the state notes at the top of the file).
// The lookup and the whole call run under g_slot_mu: without it, the pool owning the range found could be retired and its slot taken by another pool between the
// lookup and the state accesses, and the page indices (computed from the new range's base) would run past that range's state array (heap corruption; the rect watch's
// arming hit it at a game's load, when pools churn). The fault handler never takes the mutex, and this call waits on nothing while holding it, so it cannot deadlock.
void nvn_dirty_protect(const void *addr, size_t len) {
    if (!len) return;
    pthread_mutex_lock(&g_slot_mu);
    range_t *r = find_span((uintptr_t)addr, len);
    if (!r) {
        // MotoGP 18 2048 diag: a texture whose span is not inside any single tracked
        // range is NEVER write-protected, so CPU writes to it never fault and the host
        // never learns the game wrote it. Log large spans only (a 2048 BC3 is 5.6 MB).
        if (len >= (1u << 20)) { static int n; if (n++ < 40) LOG("[dm] [vk] dirty-protect: NO tracked range holds [%p, +%#zx) (%zu MB): NOT protected, CPU writes invisible", addr, len, len >> 20); }
        pthread_mutex_unlock(&g_slot_mu);
        return;
    }
    uintptr_t a = (uintptr_t)addr, p = (a - r->base) >> g_shift, q = (a + len - 1 - r->base) >> g_shift;
    // Consecutive pages are merged into one mprotect (per-page calls reach thousands per frame on the first upload of large textures; character select page startup frame 5: 4888 calls).
    // Pages already PROT / BUSY break the consecutive run: do not protect again, and do not touch pages another thread is changing
    uintptr_t run = 0; int open = 0;
    for (uintptr_t i = p; i <= q + 1; i++) {
        uint8_t st = PG_NONE;
        int took = i <= q && atomic_compare_exchange_strong(&r->state[i], &st, PG_BUSY);
        if (took && !open) { run = i; open = 1; }
        if (!took && open) { atomic_fetch_add(&r->nmp, 1); mp(r->base + (run << g_shift), (i - run) << g_shift, PROT_READ); for (uintptr_t k = run; k < i; k++) { atomic_store(&r->span[k], (uint32_t)p + 1); atomic_store(&r->state[k], PG_PROT); } open = 0; }
    }
    pthread_mutex_unlock(&g_slot_mu);
}
uint64_t nvn_dirty_now(void) { return atomic_load(&g_now); }
int nvn_dirty_since_span(const void *span, const void *addr, size_t len, uint64_t stamp) {
    if (!len) return 0;
    range_t *r = (range_t *)span; uintptr_t a0 = (uintptr_t)addr;   // the kept handle still holds the span (a slot reused by another pool over the same memory is as good): no scan
    if (!r || !atomic_load_explicit(&r->live, memory_order_acquire) || a0 < r->base || a0 + len > r->end) r = find_span(a0, len);
    if (!r) return 1;   // When untracked, conservatively treat as dirty: better to upload more than to show stale content
    if (atomic_load(&r->last) <= stamp) return 0;
    uintptr_t a = (uintptr_t)addr, p = (a - r->base) >> g_shift, q = (a + len - 1 - r->base) >> g_shift;
    for (uintptr_t i = p; i <= q; i++) if (atomic_load(&r->gen[i]) > stamp) return 1;   // The generation of the last write before upload is <= the stamp; any later write must be > the stamp
    return 0;
}
int nvn_dirty_since(const void *addr, size_t len, uint64_t stamp) { return nvn_dirty_since_span(NULL, addr, len, stamp); }
void nvn_dirty_slot_stats(int *live, int *used_slots, uint64_t *adds_total) {
    int n = atomic_load(&g_nr), active = 0;
    for (int i = 0; i < n; i++) active += live_at(i) != 0;
    *live = active; *used_slots = n; *adds_total = atomic_load(&g_adds_total);
}
void nvn_dirty_stats(uint64_t *faults, uint64_t *preps) { *faults = atomic_load(&g_faults); *preps = atomic_load(&g_preps); }
void nvn_dirty_mp_stats(uint64_t *calls, uint64_t *pages, uint64_t *ns) {
    static mach_timebase_info_data_t tb; if (!tb.denom) mach_timebase_info(&tb);
    *calls = atomic_load(&g_mp_calls); *pages = atomic_load(&g_mp_pages); *ns = atomic_load(&g_mp_ticks) * tb.numer / tb.denom;
}

#if defined(__linux__)
// Scan every readable mapping for the 16-byte pattern (4-byte aligned: it is a struct of four int32). Reads go through vm_read_overwrite, which fails
// instead of faulting when a region is unmapped mid-scan. found[] takes the matches (at most max); wr[] says whether the match's page is writable;
// region[] the /proc/self/maps region each match sits in (the file path, a bracketed name like [stack], or "anon" for an anonymous mapping).
// Returns the matches found, or -1 when the maps could not be read. One pass, 1 MB chunks with the last 16 bytes carried over so a match straddling
// a chunk boundary is still found (matches fully inside the carried prefix were found in the previous chunk and are skipped).
static int watch_scan(const int32_t pat[4], uintptr_t *found, uint8_t *wr, char (*region)[64], int max) {
    FILE *f = fopen("/proc/self/maps", "r");
    if (!f) return -1;
    uint8_t *buf = malloc(1 << 20);
    if (!buf) { fclose(f); return -1; }
    int n = 0;
    char line[512];
    while (fgets(line, sizeof line, f) && n < max) {
        unsigned long lo, hi; char perms[5], path[64];
        int flds = sscanf(line, "%lx-%lx %4s %*lx %*31s %*lu %63s", &lo, &hi, perms, path);
        if (flds < 3) continue;
        if (perms[0] != 'r') continue;
        if (strstr(line, "[vdso]") || strstr(line, "[vvar]") || strstr(line, "[vsyscall]")) continue;
        int w = perms[1] == 'w';
        uintptr_t base = lo; size_t carry = 0;
        while (base < hi && n < max) {
            size_t room = (1 << 20) - carry, want = hi - base < room ? (size_t)(hi - base) : room;
            vm_size_t got = 0;
            if (vm_read_overwrite(mach_task_self(), (vm_address_t)base, want, (vm_address_t)(buf + carry), &got) != KERN_SUCCESS || got != want) break;   // unreadable part: skip the rest of this mapping
            size_t total = carry + want;
            uintptr_t start = base - carry;   // the address of buf[0]
            for (size_t o = 0; o + 16 <= total; o += 4) {
                if (o + 16 <= carry) continue;
                int32_t v; memcpy(&v, buf + o, 4);
                if (v != pat[0] || memcmp(buf + o, pat, 16)) continue;
                found[n] = start + o; wr[n] = (uint8_t)w; snprintf(region[n], 64, "%s", flds == 4 ? path : "anon"); n++;
                if (n >= max) break;
            }
            if (n >= max || base + want >= hi) break;
            memmove(buf, buf + total - 16, 16);   // carry the last 16 bytes (total is large here: the mapping continues, so want filled the room)
            base += want; carry = 16;
        }
    }
    free(buf); fclose(f);
    return n;
}
#endif
// Write-protect the pages holding [addr, addr+16): register each page as a tracked range when no range holds it yet, then protect the covered part.
// A range of its own keeps the watch independent of pool lifetimes; a page already tracked (a pool's) is protected through that range.
static void watch_arm(uintptr_t addr) {
    uintptr_t pg = (uintptr_t)getpagesize();
    uintptr_t first = addr & ~(pg - 1), last = (addr + 15) & ~(pg - 1);
    for (uintptr_t p = first; p <= last; p += pg) {
        uintptr_t lo = addr > p ? addr : p, hi = addr + 16 < p + pg ? addr + 16 : p + pg;
        if (!nvn_dirty_tracked((const void *)p, pg)) nvn_dirty_add_range((const void *)p, pg);
        nvn_dirty_protect((const void *)lo, hi - lo);
    }
}
// Is the page holding addr still read-only in a range tracking it (1), or writable / untracked (0)? Releases update every range over a page, so any
// range saying PROT means the page is read-only; all saying NONE means it was released. Under g_slot_mu: a slot retired and reused mid-scan would give
// the page index of another range's base, past that range's state array (the same race nvn_dirty_protect had).
static int watch_protected(uintptr_t addr) {
    uintptr_t pg = (uintptr_t)getpagesize(), p = addr & ~(pg - 1);
    pthread_mutex_lock(&g_slot_mu);
    int n = atomic_load(&g_nr), prot = 0;
    for (int i = 0; i < n; i++) {
        range_t *r = rg(i);
        if (!live_at(i) || p < r->base || p >= r->end) continue;
        if (atomic_load(&r->state[(p - r->base) >> g_shift]) == PG_PROT) { prot = 1; break; }
    }
    pthread_mutex_unlock(&g_slot_mu);
    return prot;
}
// The rectangle's current values, read safely (the object may have been freed since the hit).
static int watch_read(uintptr_t addr, int32_t out[4]) {
    vm_size_t got = 0;
    return vm_read_overwrite(mach_task_self(), (vm_address_t)addr, 16, (vm_address_t)out, &got) == KERN_SUCCESS && got == 16;
}
#if defined(__linux__)
// Scan-only mode (DM_WATCH_SCAN_ONLY=1): dump the qwords around a match — the 128 before it (0x400 bytes back), the match itself, and the 16 after
// it — one qword per line, so the enclosing object's vtable pointer (a pointer into the .rodata vtable range, up to 0x400 bytes back) can be
// identified offline. Reads go through vm_read_overwrite one qword at a time, which fails instead of faulting on an unmapped address; nothing is
// write-protected, so this path cannot hit the arming race. Runs on the present thread (dm_watch_tick), not signal context.
static void watch_dump_context(uintptr_t addr) {
    uintptr_t first = addr >= 0x400 ? addr - 0x400 : 0;   // the match is never this low, but never wrap the address
    for (uintptr_t a = first; a < addr + 16 + 128; a += 8) {
        uint64_t v; vm_size_t got = 0;
        if (vm_read_overwrite(mach_task_self(), (vm_address_t)a, 8, (vm_address_t)&v, &got) == KERN_SUCCESS && got == 8)
            LOG("scan: 0x%llx: 0x%llx", (unsigned long long)a, (unsigned long long)v);
        else
            LOG("scan: 0x%llx: <unreadable>", (unsigned long long)a);
    }
}
static void watch_scan_and_arm(const int32_t pat[4]) {
    uintptr_t found[WATCH_MAX]; uint8_t wr[WATCH_MAX]; char region[WATCH_MAX][64];
    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    int n = watch_scan(pat, found, wr, region, WATCH_MAX);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    if (n < 0) { LOG("[RED] watch: cannot read /proc/self/maps, nothing armed"); return; }
    LOG("watch: scanned guest memory in %.0f ms: %d match%s for {%d,%d,%d,%d}%s", ms, n, n == 1 ? "" : "es", pat[0], pat[1], pat[2], pat[3], n >= WATCH_MAX ? " (capped)" : "");
    if (!n) { LOG("watch: no match (the rectangle may not exist yet: retry with a later DM_WATCH_RECT_AT)"); return; }
    // Fill the table and publish it before protecting any page: a fault can only happen after the mprotect, by which time the table is visible.
    int armed = 0;
    for (int i = 0; i < n; i++) {
        LOG("watch: match %#llx for {%d,%d,%d,%d} in %s", (unsigned long long)found[i], pat[0], pat[1], pat[2], pat[3], region[i]);   // before arming: if arming fails, the address is still in the log
        if (g_watch_scan_only) { watch_dump_context(found[i]); continue; }   // read-only probe: dump the context, never arm
        if (!wr[i]) { LOG("watch: match at %#llx is in a read-only mapping, not watched", (unsigned long long)found[i]); continue; }
        if (!strncmp(region[i], "[stack", 6)) { LOG("watch: match at %#llx is on a stack page (a short-lived copy): not armed", (unsigned long long)found[i]); continue; }   // a local or an argument, gone when the function returns: it cannot catch the writer that matters
        watch_t *w = &g_watch[armed];
        w->addr = found[i];
        atomic_store_explicit(&w->claimed, 0, memory_order_relaxed);
        atomic_store_explicit(&w->logged, 0, memory_order_relaxed);
        for (int k = 0; k < WATCH_HITS; k++) atomic_store_explicit(&w->hits[k].ready, 0, memory_order_relaxed);
        armed++;
    }
    if (g_watch_scan_only) { LOG("watch: scan-only mode (DM_WATCH_SCAN_ONLY=1): %d match%s dumped, nothing armed", n, n == 1 ? "" : "es"); return; }   // g_watch_n stays 0: the fault path and the tick skip the watch entirely
    atomic_store_explicit(&g_watch_n, armed, memory_order_release);
    for (int i = 0; i < armed; i++) {
        watch_arm(g_watch[i].addr);
        if (!watch_protected(g_watch[i].addr)) LOG("[RED] watch: %#llx could not be write-protected (the tick retries it)", (unsigned long long)g_watch[i].addr);
        else LOG("watch: armed %#llx", (unsigned long long)g_watch[i].addr);
    }
    if (!armed) LOG("watch: nothing armed");
}
#endif
// Log each recorded hit once (with the values after the write, read now), re-arm the page after a logged hit so the next write faults again (until
// the ring is full), and re-protect watches whose pages the tracker released without a fault.
static void watch_report(uint64_t presents) {
    int n = atomic_load_explicit(&g_watch_n, memory_order_acquire);
    for (int i = 0; i < n; i++) {
        watch_t *w = &g_watch[i];
        int logged = atomic_load_explicit(&w->logged, memory_order_relaxed);
        int progress = 0;
        while (logged < WATCH_HITS && atomic_load_explicit(&w->hits[logged].ready, memory_order_acquire)) {   // in order: a later slot may be ready first
            watch_hit_t *h = &w->hits[logged];
            int32_t now[4];
            char after[64];
            if (watch_read(w->addr, now)) snprintf(after, sizeof after, "{%d,%d,%d,%d}", now[0], now[1], now[2], now[3]);
            else snprintf(after, sizeof after, "(unreadable now)");
            char pcs[96], lrs[96], chain[512]; size_t clen = 0; chain[0] = 0;   // PC, LR and the FP chain as module-relative offsets (a raw address outside any module)
            module_t *m = module_of((const void *)h->pc);
            if (m) snprintf(pcs, sizeof pcs, "%s+%#llx", m->name, h->pc - (uint64_t)m->base);
            else snprintf(pcs, sizeof pcs, "%#llx", (unsigned long long)h->pc);
            module_t *lm = module_of((const void *)h->lr);
            if (lm) snprintf(lrs, sizeof lrs, "%s+%#llx", lm->name, h->lr - (uint64_t)lm->base);
            else snprintf(lrs, sizeof lrs, "%#llx", (unsigned long long)h->lr);
            for (int f = 0; f < h->nfp && clen < sizeof chain - 1; f++) {
                module_t *fm = module_of((const void *)h->fp[f]);
                int wlen;
                if (fm) wlen = snprintf(chain + clen, sizeof chain - clen, "%s%s+%#llx", f ? " " : "", fm->name, h->fp[f] - (uint64_t)fm->base);
                else wlen = snprintf(chain + clen, sizeof chain - clen, "%s%#llx", f ? " " : "", (unsigned long long)h->fp[f]);
                if (wlen > 0) clen += (size_t)wlen;
            }
            LOG("watch: write to rect %#llx at present %llu (hit %d/%d): writer pc %s lr %s, values {%d,%d,%d,%d} -> %s, fp: %s", (unsigned long long)w->addr,
                (unsigned long long)presents, logged + 1, WATCH_HITS, pcs, lrs,
                h->old[0], h->old[1], h->old[2], h->old[3], after, h->nfp ? chain : "(none)");
            atomic_store_explicit(&w->logged, ++logged, memory_order_relaxed);
            progress = 1;
        }
        if (progress) {
            if (logged < WATCH_HITS) {   // a hit was logged: re-arm for the next write
                watch_arm(w->addr);
                if (!watch_protected(w->addr)) LOG("[RED] watch: %#llx could not be re-protected after hit %d", (unsigned long long)w->addr, logged);
            } else LOG("watch: %#llx recorded %d hits, done", (unsigned long long)w->addr, WATCH_HITS);   // short arm: the ring is full
            continue;
        }
        if (logged >= WATCH_HITS) continue;   // done: no re-protecting a finished watch
        // No hit yet: the tracker may have released the page without a fault (a host write, a texture sweep). Re-protect it, checked every 60 presents.
        if (presents % 60 == 0 && !watch_protected(w->addr)) {
            watch_arm(w->addr);
            LOG("watch: %#llx was writable again without a hit, re-protected at present %llu", (unsigned long long)w->addr, (unsigned long long)presents);
        }
    }
}
void dm_watch_tick(uint64_t presents) {
    static int state;   // 0 = not read yet, -1 = off, 1 = waiting for the scan present, 2 = scanned (armed, or dumped in scan-only mode)
    static uint64_t at; static int32_t pat[4];
    if (!state) {
        state = -1;
        const char *e = getenv("DM_WATCH_RECT");
        if (!e || !*e) return;
        if (sscanf(e, "%d,%d,%d,%d", &pat[0], &pat[1], &pat[2], &pat[3]) != 4) { LOG("[RED] DM_WATCH_RECT=%s: want four integers x0,y0,x1,y1", e); return; }
        const char *a = getenv("DM_WATCH_RECT_AT");
        at = a ? strtoull(a, NULL, 10) : 30;
        if (!at) at = 1;
        { const char *s = getenv("DM_WATCH_SCAN_ONLY"); g_watch_scan_only = s && atoi(s); }
        state = 1;
        LOG("watch: DM_WATCH_RECT={%d,%d,%d,%d}, scanning at present %llu%s", pat[0], pat[1], pat[2], pat[3], (unsigned long long)at, g_watch_scan_only ? " (scan-only: no arming)" : "");
    }
    if (state == 1 && presents >= at) {
        state = 2;
#if defined(__linux__)
        watch_scan_and_arm(pat);
#else
        LOG("watch: the memory scan needs Linux (/proc/self/maps); nothing armed on this platform");
#endif
    }
    if (state != 2) return;
    watch_report(presents);
}

// ---------- Box poke (DM_POKE_BOX) ----------
// A game whose menus are laid out in a box narrower than the framebuffer (MotoGP 18: a 1280-wide box centred in a 1600-wide viewport, while the race
// is fullscreen) keeps the box in a live value the code does not spell out as a constant: the size getter already returns the full width, no code site
// holds the narrow width, and no scissor or viewport is narrowed. DM_POKE_BOX=1 finds the box values in guest memory at run time and overwrites them
// with the fullscreen values at the start of every present, so a layout pass that rewrites them each frame is corrected again before the next frame
// is drawn. Four patterns, each 16 bytes (built from typed literals, never byte strings):
//   float {0.1, 0, 0.8, 1} -> {0, 0, 1, 1}            (a normalised origin/size quad, e.g. ULocalPlayer)
//   int   {160, 0, 1440, 720} -> {0, 0, 1600, 720}    (a pixel rect, max form)
//   int   {160, 0, 1280, 720} -> {0, 0, 1600, 720}    (a pixel rect, size form)
//   float {160, 0, 1440, 720} -> {0, 0, 1600, 720}    (a pixel rect as floats)
// The scan covers GUEST memory only: the guest address block (heap, alias region) and the loaded modules' images (data/BSS). Host mappings are never
// scanned, so the host's own copy of the rect (gfx_vk_exec.c g_pass_bb, a per-pass telemetry bounding box in libdmhost.so's .bss) cannot match —
// poking it would do nothing, it is recomputed from the draws every pass. Guest thread stacks are skipped too (a rect on a stack is a transient copy,
// gone when the function returns). Only writable mappings are scanned: a match in read-only memory could not be poked anyway, and is counted and
// reported separately. Guard: if the four patterns match more than POKE_MAX addresses in total, nothing is poked — the values are too common to trust
// any one of them. A poke writes only when the address still holds the box pattern (read back first): an address the game freed and reused, or one
// already poked, is left alone. Runs on the present thread (dm_poke_tick), Linux only (/proc/self/maps).
enum { POKE_NPAT = 4, POKE_MAX = 4, POKE_CAP = 16, POKE_RETRIES = 5, POKE_RETRY_EVERY = 600, POKE_FIRST = 30 };
#if defined(__linux__)
int dm_guest_as_contains(uintptr_t a);      // hle_svc.c
int dm_guest_stack_contains(uintptr_t a);   // hle_svc.c
typedef struct { const char *name; const void *orig, *repl; } poke_pat_t;
static const float poke_f0_orig[4] = { 0.1f, 0.0f, 0.8f, 1.0f };
static const float poke_f0_repl[4] = { 0.0f, 0.0f, 1.0f, 1.0f };
static const int32_t poke_i1_orig[4] = { 160, 0, 1440, 720 };
static const int32_t poke_i1_repl[4] = { 0, 0, 1600, 720 };
static const int32_t poke_i2_orig[4] = { 160, 0, 1280, 720 };
static const int32_t poke_i2_repl[4] = { 0, 0, 1600, 720 };
static const float poke_f3_orig[4] = { 160.0f, 0.0f, 1440.0f, 720.0f };
static const float poke_f3_repl[4] = { 0.0f, 0.0f, 1600.0f, 720.0f };
static const poke_pat_t poke_pats[POKE_NPAT] = {
    { "float {0.1,0,0.8,1}", poke_f0_orig, poke_f0_repl },
    { "int {160,0,1440,720}", poke_i1_orig, poke_i1_repl },
    { "int {160,0,1280,720}", poke_i2_orig, poke_i2_repl },
    { "float {160,0,1440,720}", poke_f3_orig, poke_f3_repl },
};
typedef struct { uintptr_t addr; int pat; char region[64]; } poke_hit_t;
static poke_hit_t g_poke[POKE_MAX];
static int g_poke_n;
// Scan the writable guest mappings for the four box patterns. hits[] takes the matches found (at most POKE_CAP); per[] counts every writable match
// per pattern (uncapped: the guard needs the true total), ro[] counts matches in read-only mappings, which are reported but never poked. Returns the
// total writable matches, or -1 when the maps could not be read. One pass, 1 MB chunks with the last 16 bytes carried over (as watch_scan).
static int poke_scan(poke_hit_t *hits, int per[POKE_NPAT], int ro[POKE_NPAT]) {
    FILE *f = fopen("/proc/self/maps", "r");
    if (!f) return -1;
    uint8_t *buf = malloc(1 << 20);
    if (!buf) { fclose(f); return -1; }
    uint32_t first[POKE_NPAT];
    for (int i = 0; i < POKE_NPAT; i++) memcpy(&first[i], poke_pats[i].orig, 4);   // the first word of each pattern, for the quick check
    int total = 0, stored = 0;
    char line[512];
    while (fgets(line, sizeof line, f)) {
        unsigned long lo, hi; char perms[5];
        if (sscanf(line, "%lx-%lx %4s", &lo, &hi, perms) < 3) continue;
        if (perms[0] != 'r') continue;
        if (!dm_guest_as_contains(lo) && !module_of((const void *)lo)) continue;   // a host mapping: never scanned (the host's own rect copy lives in one)
        if (dm_guest_stack_contains(lo) || dm_guest_stack_contains(hi - 1)) continue;   // a guest thread stack: transient copies only
        int w = perms[1] == 'w';
        uintptr_t base = lo; size_t carry = 0;
        while (base < hi) {
            size_t room = (1 << 20) - carry, want = hi - base < room ? (size_t)(hi - base) : room;
            vm_size_t got = 0;
            if (vm_read_overwrite(mach_task_self(), (vm_address_t)base, want, (vm_address_t)(buf + carry), &got) != KERN_SUCCESS || got != want) break;   // unreadable part: skip the rest of this mapping
            size_t tot = carry + want;
            uintptr_t start = base - carry;   // the address of buf[0]
            for (size_t o = 0; o + 16 <= tot; o += 4) {
                if (o + 16 <= carry) continue;   // fully inside the carried prefix: found in the previous chunk
                uint32_t v; memcpy(&v, buf + o, 4);
                for (int p = 0; p < POKE_NPAT; p++) {
                    if (v != first[p] || memcmp(buf + o, poke_pats[p].orig, 16)) continue;
                    if (!w) { ro[p]++; break; }   // read-only: counted, never stored
                    per[p]++; total++;
                    if (stored < POKE_CAP) {
                        poke_hit_t *h = &hits[stored++];
                        h->addr = start + o; h->pat = p;
                        module_t *m = module_of((const void *)h->addr);
                        if (m) snprintf(h->region, sizeof h->region, "%s+%#llx", m->name, h->addr - (uint64_t)m->base);
                        else snprintf(h->region, sizeof h->region, "guest-AS");
                    }
                    break;   // the patterns are disjoint: one address matches at most one
                }
            }
            if (base + want >= hi) break;
            memmove(buf, buf + tot - 16, 16);   // carry the last 16 bytes (total is large here: the mapping continues, so want filled the room)
            base += want; carry = 16;
        }
    }
    free(buf); fclose(f);
    return total;
}
// One scan attempt: log the counts, cache the matches when the guard passes. Returns 1 when armed, 0 when nothing matched, -1 when the guard tripped.
static int poke_scan_and_cache(void) {
    poke_hit_t hits[POKE_CAP]; int per[POKE_NPAT] = { 0 }, ro[POKE_NPAT] = { 0 };
    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    int total = poke_scan(hits, per, ro);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    if (total < 0) { LOG("[RED] poke: cannot read /proc/self/maps"); return 0; }   // retried like a no-match: the maps may appear later
    int nro = 0; for (int p = 0; p < POKE_NPAT; p++) nro += ro[p];
    LOG("poke: scanned guest memory in %.0f ms: %d match%s (%d/%d/%d/%d per pattern)%s", ms, total, total == 1 ? "" : "es",
        per[0], per[1], per[2], per[3], nro ? " (+ read-only matches skipped)" : "");
    if (!total) { LOG("poke: no match yet (the box values may not exist until the menu is laid out)"); return 0; }
    if (total > POKE_MAX) { LOG("poke: %d matches (> %d): poking nothing (guard)", total, POKE_MAX); return -1; }
    for (int i = 0; i < total; i++) {
        g_poke[i] = hits[i];
        LOG("poke: %s at %#llx in %s", poke_pats[hits[i].pat].name, (unsigned long long)hits[i].addr, hits[i].region);
    }
    g_poke_n = total;
    LOG("poke: armed %d address%s, poking at the start of every present", total, total == 1 ? "" : "es");
    return 1;
}
// Overwrite every cached address that still holds its box pattern. The game may rewrite the values each frame (a layout pass), so the poke runs
// every present; the read-back first means an address already poked, or freed and reused for something else, is left alone. A poke into a page the
// dirty tracker write-protected is released first (prep_write); a fault between the release and the store still goes through the tracker's fault path,
// which releases the page and lets the write through. The 16-byte store is not atomic against the game's own writes, but a torn write lasts one frame:
// the next present pokes it again. Logs each write for the first 10 presents after arming, then every 600th present only.
static void poke_apply(uint64_t presents, uint64_t armed_at) {
    int log = presents - armed_at < 10 || presents % 600 == 0;
    for (int i = 0; i < g_poke_n; i++) {
        const poke_pat_t *pat = &poke_pats[g_poke[i].pat];
        uint8_t cur[16]; vm_size_t got = 0;
        if (vm_read_overwrite(mach_task_self(), (vm_address_t)g_poke[i].addr, 16, (vm_address_t)cur, &got) != KERN_SUCCESS || got != 16) continue;   // freed or unmapped since the scan
        if (memcmp(cur, pat->orig, 16)) continue;   // already poked, or repurposed: leave it
        prep_write((void *)g_poke[i].addr, 16);
        memcpy((void *)g_poke[i].addr, pat->repl, 16);
        if (log) LOG("poke: %#llx %s -> fullscreen at present %llu", (unsigned long long)g_poke[i].addr, pat->name, (unsigned long long)presents);
    }
}
#endif
void dm_poke_tick(uint64_t presents) {
    static int state;   // 0 = not read yet, -1 = off/done, 1 = waiting for a scan, 2 = armed (poking every present)
    static int retries; static uint64_t next_scan, armed_at;
    if (!state) {
        state = -1;
        const char *e = getenv("DM_POKE_BOX");
        if (!e || !atoi(e)) return;
        state = 1; retries = 0; next_scan = POKE_FIRST;
        LOG("poke: DM_POKE_BOX=1, scanning guest memory at present %d", POKE_FIRST);
    }
    if (state == 1 && presents >= next_scan) {
#if defined(__linux__)
        int r = poke_scan_and_cache();
        if (r == 1) { state = 2; armed_at = presents; }
        else if (++retries > POKE_RETRIES) { state = -1; LOG("poke: giving up after %d scans, nothing poked", POKE_RETRIES + 1); }
        else next_scan = presents + POKE_RETRY_EVERY;
#else
        state = -1;
        LOG("poke: the memory scan needs Linux (/proc/self/maps); nothing poked on this platform");
#endif
    }
    if (state != 2) return;
#if defined(__linux__)
    poke_apply(presents, armed_at);
#endif
}
