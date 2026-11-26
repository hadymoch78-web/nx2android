// BRK hooks on guest code (DM_HOOK_1CED80C, DM_HOOK_E6E6AC, DM_HOOK_64EA18, DM_GFX_PRESET): when a static search cannot say what a
// function computes at run time, put a breakpoint on it and log the registers. Installed at load (probe_install, after the modules are
// loaded, before any guest code runs): the host writes a BRK over the original instruction, the SIGTRAP hook records the registers and
// emulates the displaced instruction in the handler, and the game runs on as if nothing happened. Six configurations instrument MotoGP 18
// (main v0, build 758a0602b2c99b4395277296bc002848): the viewport-offset function (main+0x1ced80c: the gate result, the scale factors
// (s0, s1) it gets from a virtual call, that callee's address, and the offsets it stores), the projection function (main+0xe6e6ac: the
// cached-projection flag, the 16 cached-matrix floats, the view dimensions, and the computed projection terms), the size getter
// (main+0x64ea18: the caller, the four size fields it reads off the object, and the (W, H) it returns), the graphics settings manager
// (DM_GFX_PRESET: USettingsVideoManager's apply routine, main+0x1252f8, and its console-exec helper, main+0x125e90: the manager reads
// each cvar's value from its own map and console-execs "<cvar> <value>", which is why -ini args never stick; the hook rewrites the value
// string in place before the exec, so a preset chosen in env.txt actually applies — the observe run showed this manager never fires, so
// it is kept for completeness but the live route is the sixth), the device-profile restore (DM_GFX_PRESET: main+0x177383c, the restore
// function that re-applies the previous values from the manager's map on a profile change — round 13 misread it as the applier and the
// observe run showed its map empty at boot, so the hook is kept for the restore-map visibility but injects nothing), and the per-CVar
// apply helper (DM_GFX_PRESET: main+0x36a380, the single funnel through which every CVar application at boot passes — profile and ini
// sections alike: the hook logs each (name, value, flags, mode) in observe mode, and in preset modes substitutes the value at entry by
// redirecting x2 to a static host UTF-32 buffer — see the note at h36_points).
// Each configuration is gated by its own env var and installs independently. Fixed addresses run only on the main build they were found
// in (AGENTS.md); any other build, or bytes that do not match the ones verified against the image, installs nothing. The displaced
// instructions are emulated in the handler (not run from a trampoline slot, as probe.c does): two of them are conditional branches, whose
// targets depend on where they execute. No guest function is called from any handler: the handlers run in signal context and only
// read/write guest memory.
#include "dm.h"
#include "dm_uc.h"
#include <TargetConditionals.h>
#include <signal.h>
#include <sys/ucontext.h>
#include <sys/mman.h>
#include <sys/uio.h>
#include <libkern/OSCacheControl.h>
#include <mach/mach.h>
#include <pthread.h>
#include <stdatomic.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
void crash_set_trap_hook(int (*fn)(ucontext_t *uc));
int (*crash_trap_hook(void))(ucontext_t *uc);   // the trap hook before ours, for chaining (crash.c)

#define HOOK_BUILD "758a0602b2c99b4395277296bc002848"   // MotoGP 18 main v0
#define HOOK_BRK 0xd43e0000                             // brk #0xf000, as probe.c
typedef struct { uint64_t off; uint32_t orig; } hook_point;   // a hook point: offset in main + the original word, verified against image.bin
enum { HOOK_GATE = 0, HOOK_BLR, HOOK_LDP, HOOK_STP, HOOK_N };
static const hook_point k_points[HOOK_N] = {
    { 0x1ced838, 0x360002a0 },   // tbz w0, #0, #0x1ced88c (the gate's branch)
    { 0x1ced84c, 0xd63f0100 },   // blr x8 (the scale-factor call)
    { 0x1ced850, 0x294f2668 },   // ldp w8, w9, [x19, #0x78] (W, H; s0/s1 hold the scale factors here)
    { 0x1ced884, 0x2910226a },   // stp w10, w8, [x19, #0x80] (offsetX, offsetY)
};
// The projection function (main+0xe6e6ac). The view object is x1 (the prologue moves it to x19 at 0xe6e6d8; x0 goes to x20), so the entry
// hook reads the flag and the cached matrix off x1. The entry hook sits on the mov itself, not on the function's first instruction
// (str d12, [sp, #-0x90]!), because the mov emulates as a single register copy while the store would need a stack write from signal context;
// nothing between the entry and the mov touches x0, x1 or lr, so the logged values are the entry values. The cbz at 0xe6e8f0 is the branch
// between the cached path (fall through) and the compute path (0xe6e964); the dims (w24, w9) are read there. All three paths — cached
// (b at 0xe6e960), tanf (falling through 0xe6ea80) and alternate-divisor (b at 0xe6e9fc, which SKIPS 0xe6ea80) — join at 0xe6ea84, so the
// terms are read there: m00 from [sp, #0x28], m11 from w20, the branch selector from [x19, #0x318]. One record per call, committed at the
// join; a call that exits early (null [x19, #0x328] -> the epilogue at 0xe6f03c) stages but never commits, and the next call overwrites it.
enum { E6_ENTRY = 0, E6_CBZ, E6_TERMS, E6_N };
static const hook_point e6_points[E6_N] = {
    { 0xe6e6d8, 0xaa0103f3 },   // mov x19, x1 (x1 is the view object; x0 and lr are still the entry values)
    { 0xe6e8f0, 0x340003a8 },   // cbz w8, #0xe6e964 (w8 == 0: compute path; else cached; w24/w9 hold W/H)
    { 0xe6ea84, 0x394cc268 },   // ldrb w8, [x19, #0x330] (the join point: m00 is at [sp, #0x28], m11 in w20)
};
// The size getter (main+0x64ea18, size 0x100). It reads four float fields off the object (x0, moved to x19 at 0x64ea30): [x19+0x588],
// [x19+0x58c], [x19+0x598], [x19+0x59c], plus the flags at [x19+0x570] (bit 9: the direct path, which skips the DPI adjustment; bit 8: the
// adjustment runs) and the H adjust at [x19+0x5b8], and returns (W, H) as floats in (s0, s1) — W is s9-s11 and H is s10-s12 at 0x64eaf8.
// The entry hook sits on the mov itself, not on the function's first instruction (stp d13, d12, [sp, #-0x50]!), for the same reason as the
// e6 entry hook: the mov emulates as a single register copy, and nothing between the entry and the mov touches x0 or lr, so the logged
// values are the entry values (the prologue saved x30 at [sp, #0x48] — stp x29, x30, [sp, #0x40] at 0x64ea28 — but the live lr is identical
// at the mov). The return hook sits on the first epilogue instruction (ldr x19, [sp, #0x30] at 0x64eb00): s0/s1 still hold the returned
// (W, H), and the single ret is at 0x64eb14 with both paths joining at 0x64eaf8, so every call commits. The four blr calls between the hooks
// are virtual calls on the DPI provider ([x19+0x6b8], slots 0x40/0x78/0xb8/0xf0), not re-entries into this getter; the staged flag is
// defensive. One record per call, committed at the return.
enum { S64_ENTRY = 0, S64_RET, S64_N };
static const hook_point s64_points[S64_N] = {
    { 0x64ea30, 0xaa0003f3 },   // mov x19, x0 (x0 is the object; lr is still the entry value)
    { 0x64eb00, 0xf9401bf3 },   // ldr x19, [sp, #0x30] (the epilogue: s0/s1 hold the returned W/H)
};
// The graphics settings manager (USettingsVideoManager). Its apply routine (main+0x1252f8, size 0xb98) reads each cvar's value from the
// manager's own map (an FName per cvar, converted to an FString by 0x403330) and applies it through its console-exec helper (main+0x125e90,
// called only from the apply routine: exactly 15 BL sites, one per cvar). The entry hook sits on the mov at 0x125310 (x0 is the manager
// object, lr the caller: nothing between the entry and the mov touches either), and logs each manager call so the run shows WHEN the
// manager applies (once at boot, per level load, repeatedly). The apply hook sits on the helper's first instruction (sub sp, sp, #0x90):
// x1 is the cvar name as an FString, x2 the value as an FString; the handler rewrites the value's characters in place (and its length)
// when a preset overrides that cvar, so the console command the helper builds carries the preset value. No guest function is called from
// the handler: the rewrite touches only the transient FString copy the helper is about to read, never the manager's stored map (whose
// values are FNames: poking them would need a name-table index for each preset value, which may not exist).
enum { GFX_MGR = 0, GFX_APPLY, GFX_N };
static const hook_point gfx_points[GFX_N] = {
    { 0x125310, 0xaa0003f3 },   // mov x19, x0 (x0 is the manager object; lr is still the entry value)
    { 0x125e90, 0xd10243ff },   // sub sp, sp, #0x90 (the helper's first instruction; x1 = name FString, x2 = value FString)
};
// The device-profile restore (main+0x177383c, size 0x204). It iterates the manager's name->value map ([x0+0xb8], entries 0x28 B:
// {FString Key @+0x00, FString Value @+0x10}) and, per entry, calls FindConsoleVariable + IConsoleVariable::Set(value, 0x4000000).
// Round 13 misread this as the choke point through which EVERY profile CVar passes at boot; round 15 proved it is the RESTORE
// function (it re-applies the PREVIOUS values saved in the map, and the map is empty by design at boot — see the correction note
// below). The entry hook sits on the function's first instruction (stp x28, x27, [sp, #-0x60]!): x0 is the manager,
// lr the caller, and the whole map is in memory at entry.
//
// DEVIATION FROM SHAPE #3 — READ THIS BEFORE CHANGING THE PRESET PATH. Gemini chose shape #3 (run the original, then a preset
// FindConsoleVariable+Set loop). That shape needs GUEST CALLS from the trap handler, and this framework runs the handler in signal
// context with no guest-call facility: no trampoline, no register save/restore for calls, no way to run guest code and get control
// back (probe.c's trampoline only replays a displaced instruction; it cannot call a guest function). Building one — a per-thread
// call state machine plus a return-BRK trampoline — would expose the boot path to delegate re-entrancy: Set fires CVar change
// delegates, which can re-enter this very function while a guest call is in flight, with no recovery path, and a corrupted thread
// at boot means the game never starts. So preset modes here use MAP REWRITE (shape #1 mechanics): at entry, before the original's
// loop runs, the handler rewrites the Value FStrings of the preset CVars in the map, and the engine's own loop then applies the
// preset values with the correct flags, locking, delegates and ordering. The end state is the same for every CVar the profile
// sets. What map rewrite CANNOT do: set a CVar the profile map does not contain — the observe log lists every CVar in the map,
// and the flush histogram reports which preset CVars were never seen, so the gap (if any) is visible and Gemini can decide
// whether a guest-call loop is worth it for the missing ones. Observe mode only reads; preset modes only rewrite values that
// fit (TOOLONG otherwise); a value already equal to the preset is left alone (SAME), so re-applies are harmless.
//
// CORRECTION (round 15, FINDINGS_BOOTAPPLY.md): 0x177383c is the RESTORE function, not the applier — it re-applies the PREVIOUS
// values saved in the map, and the map is empty by design at boot (the selection empties it at its start and only fills it on a
// profile change, mode 1). The real boot applier is 0x1772084 -> 0x36a380 (see h36_points below). This hook stays installed: its
// observe log is still the only view of the restore map, and its map-rewrite path is harmless on an empty map.
enum { DP_APPLY = 0, DP_N };
static const hook_point dp_points[DP_N] = {
    { 0x177383c, 0xa9ba6ffc },   // stp x28, x27, [sp, #-0x60]! (the restore routine's first instruction; x0 is the manager)
};
// The per-CVar apply helper (main+0x36a380, size 0x3b4). The REAL boot applier (round 15, FINDINGS_BOOTAPPLY.md; confirmed at
// runtime by the round-16 observe log, 380 calls, 0 unreadable): every CVar application at boot funnels through it — the
// device-profile selection (0x1772084 loop 3, flags 0x4000000, lr main+0x17728ec), the ini sections and the sg.* delegates
// (0x368bbc, flags 0x3000000 / 0x01000000, lr main+0x368d94) and the early ConsoleVariables.ini site (BLR at 0x364864, flags
// 0x5000000, lr main+0x364868 — a third call site the round-15 BL-only census missed; it never sets a preset CVar). At entry
// x1 is the CVar name and x2 the value, BOTH AS TCHAR* (NUL-terminated UTF-32 strings — NOT FStrings: the function reads
// wchars straight off x2, 0x36a3a0 `ldr w0, [x23]`, and the call sites pass Data pointers, 0x17728dc / 0x368d78 `ldr x2, [x8]`),
// w3 the flags, w4 the mode. Inside it normalizes the value, looks the CVar up (FindConsoleVariable, slot 0x90) and tail-calls
// IConsoleVariable::Set (slot 0x68) at 0x36a6dc — or registers an unknown CVar. The entry hook sits on the function's first
// instruction (sub sp, sp, #0x60). SUBSTITUTION (round 17, Gemini-approved): in preset modes the handler redirects x2 to a
// static host UTF-32 buffer holding the preset value — matched by name + flags (0x04000000 for the nine profile CVars,
// 0x01000000 for the three sg.*-delegate CVars) and restricted to lr main+0x17728ec / main+0x368d94 — so the engine's own Set
// applies the preset with the correct flags, locking, delegates and ordering. Observe mode only reads. Nothing is ever written
// to guest memory: the substitution only changes a register in the saved state.
enum { H36_APPLY = 0, H36_N };
static const hook_point h36_points[H36_N] = {
    { 0x36a380, 0xd10183ff },   // sub sp, sp, #0x60 (the helper's first instruction; x1 = name TCHAR*, x2 = value TCHAR*)
};
static uint8_t *g_base;                 // main's base (set at install)
static uint64_t g_at[HOOK_N];           // absolute hook addresses (1ced80c)
static uint64_t e6_at[E6_N];            // absolute hook addresses (e6e6ac)
static uint64_t s64_at[S64_N];          // absolute hook addresses (64ea18)
static uint64_t gfx_at[GFX_N];          // absolute hook addresses (1252f8/125e90)
static uint64_t dp_at[DP_N];            // absolute hook addresses (177383c)
static uint64_t h36_at[H36_N];          // absolute hook addresses (36a380)
static _Atomic int g_installed;         // the trap hook is chained (either configuration installed)
static int g_hooked_1c, g_hooked_e6, g_hooked_64, g_hooked_gfx, g_hooked_dp, g_hooked_36;   // which configurations installed (set at install, read by the trap hook)
static int (*g_prev)(ucontext_t *uc);   // the trap hook before ours (probe.c's, if any)

// One call of 0x1ced80c, as the flush thread logs it. The four hooks of a call run on one thread in order (gate, blr, ldp, stp: the code
// between them has no branch), so the handler stages the values per thread and commits one record at the stp; a gate-false call (the else
// branch, which never reaches the other hooks) commits at the tbz with gate 0.
typedef struct {
    _Atomic int ready;
    uint64_t seq;
    uint64_t obj, subobj, callee, lr;
    int32_t w, h, offx, offy;
    uint32_t s0bits, s1bits;
    int gate, has_wh, store_ok;
} hook_rec;
enum { HOOK_RING = 4096 };
static hook_rec g_ring[HOOK_RING];
static _Atomic uint64_t g_seq;
static __thread struct { int gate, has_wh; uint64_t obj, subobj, callee, lr; int32_t w, h; uint32_t s0bits, s1bits; } g_st;

// One call of 0xe6e6ac, as the flush thread logs it. The three hooks of a call run on one thread in order (entry, cbz, terms: nothing
// between them re-enters this function), so the handler stages the values per thread and commits one record at the join.
typedef struct {
    _Atomic int ready;
    uint64_t seq;
    uint64_t obj, arg0, lr;
    int32_t w, h;
    uint32_t m00bits, m11bits;
    uint32_t mat[16];   // the cached matrix, object order ([x1+0x850..0x88c]); the off-center terms the compute path zeroes are indices
                        // 1, 2, 3, 6, 7, 9, 13 (they land in the zeroed stack slots [sp+0x48/0x40/0x44/0x38/0x30/0x2c/0x20])
    int flag, cached, alt, has_mat;
} e6_rec;
enum { E6_RING = 4096 };
static e6_rec e6_ring[E6_RING];
static _Atomic uint64_t e6_seq;
static __thread struct { uint64_t obj, arg0, lr; int32_t w, h; uint32_t mat[16]; int flag, cached, has_mat, staged; } e6_st;

// One call of 0x64ea18, as the flush thread logs it. The two hooks of a call run on one thread in order (entry, return: the four virtual
// calls between them are on the DPI provider, not re-entries into this getter), so the handler stages the values per thread and commits one
// record at the return. The fields and the returned (W, H) are floats, kept as raw bits and printed as %g with their hex.
typedef struct {
    _Atomic int ready;
    uint64_t seq;
    uint64_t obj, lr, dpi;   // the object, the caller, the DPI provider ([obj+0x6b8], 0 when null)
    uint32_t flags;          // [obj+0x570]: bit 9 = direct path, bit 8 = DPI adjustment
    uint32_t f588, f58c, f598, f59c, f5b8;   // the size fields ([obj+0x588/+0x58c/+0x598/+0x59c]) and the H adjust (+0x5b8)
    uint32_t wb, hb;         // the returned (W, H) in s0/s1
    int has_fields;
} s64_rec;
enum { S64_RING = 4096 };
static s64_rec s64_ring[S64_RING];
static _Atomic uint64_t s64_seq;
static __thread struct { uint64_t obj, lr, dpi; uint32_t flags, f588, f58c, f598, f59c, f5b8; int has_fields, staged; } s64_st;

// One manager call or one cvar apply, as the flush thread logs it. The apply hook commits one record per cvar the helper is called with
// (15 per manager call); the entry hook commits one record per manager call. The name and the old/new values are copied out as ASCII in
// the handler (the guest strings are 32-bit wchar_t); the record is self-contained, so the flush thread never touches guest memory.
enum { GFX_KIND_MGR = 0, GFX_KIND_APPLY };
enum { GFX_ACT_SAME = 0, GFX_ACT_REWROTE, GFX_ACT_TOOLONG, GFX_ACT_KEEP, GFX_ACT_UNREADABLE };
typedef struct {
    _Atomic int ready;
    uint64_t seq;
    int kind, action;
    uint64_t obj, lr;
    char name[40];   // the cvar name ("fx.MaxCPUParticlesPerEmitter" is 28 chars)
    char oldv[16], newv[16];   // the value before / after (values are short: "100", "15", "0")
} gfx_rec;
enum { GFX_RING = 4096 };
static gfx_rec gfx_ring[GFX_RING];
static _Atomic uint64_t gfx_seq;
static __thread int gfx_busy;   // anti-recursion guard for the trap handler (defensive: it calls no guest code)

// One device-profile apply call, or one CVar in its map, as the flush thread logs it. The call record carries the map size and
// whether the displaced store emulated; each cvar record carries the name and the value before/after (in observe mode the two are
// equal). The name and the old/new values are copied out as ASCII in the handler (the guest strings are 32-bit wchar_t); the record
// is self-contained, so the flush thread never touches guest memory.
enum { DP_KIND_CALL = 0, DP_KIND_CVAR };
enum { DP_ACT_KEEP = 0, DP_ACT_SAME, DP_ACT_REWROTE, DP_ACT_TOOLONG, DP_ACT_UNREADABLE };
typedef struct {
    _Atomic int ready;
    uint64_t seq;
    int kind, action;
    uint64_t obj, lr;
    int count;      // the map's element count ([manager+0xe0]) — call records only
    int emu_ok;     // the displaced stp's store landed — call records only
    char name[40];   // the cvar name ("fx.MaxCPUParticlesPerEmitter" is 28 chars)
    char oldv[16], newv[16];   // the value before / after (values are short: "100", "15", "0")
} dp_rec;
enum { DP_RING = 4096 };
static dp_rec dp_ring[DP_RING];
static _Atomic uint64_t dp_seq;
static __thread int dp_busy;   // anti-recursion guard for the trap handler (defensive: it calls no guest code)

// One 0x36a380 call, as the flush thread logs it. The name and the value are copied out as ASCII in the handler (the guest
// strings are NUL-terminated UTF-32); the record is self-contained, so the flush thread never touches guest memory. The flags
// (w3) identify the source: 0x4000000 = device profile, 0x3000000 = ini sections, 0x5000000 = ConsoleVariables.ini,
// 0x01000000 = code (including the sg.* delegates). The sequence number is the call order — the sg.* delegate-ordering
// analysis needs it. newv is the value the engine actually applied: the preset value when the call was substituted, the
// supplied value otherwise.
enum { H36_ACT_KEEP = 0, H36_ACT_UNREADABLE, H36_ACT_SUBST, H36_ACT_SAME, H36_ACT_NOBUF };
typedef struct {
    _Atomic int ready;
    uint64_t seq;
    int action;
    uint64_t lr;
    uint32_t flags;   // w3
    uint32_t mode;    // w4
    char name[64];    // the cvar name ("fx.MaxCPUParticlesPerEmitter" is 28 chars; 64 covers the longest engine names)
    char value[64];   // the value as supplied (values are short: "100", "15", "True"; 64 covers the longest)
    char newv[64];    // the value applied (the preset value on SUBST, the supplied value otherwise)
} h36_rec;
enum { H36_RING = 4096 };
static h36_rec h36_ring[H36_RING];
static _Atomic uint64_t h36_seq;
static __thread int h36_busy;   // anti-recursion guard for the trap handler (defensive: it calls no guest code)

// The manager's 15 cvars, in the apply routine's order (each is referenced twice there: map lookup + console-exec). r.VSync and t.MaxFPS
// are recognised but never overridden: the FPS unlock (DM_PRESENT_INTERVAL=display, menu 40fps) and the display-res override depend on
// them. PoolSize, LODGroups and r.Shadow.MaxResolution are NOT in this table and cannot be injected this way: they are config structs,
// not manager cvars (the .pak route that could reach them does not load).
static const char *const gfx_cvars[15] = {
    "r.VSync", "r.ScreenPercentage", "r.PostProcessAAQuality", "r.MaxAnisotropy", "r.Streaming.MipBias",
    "r.AmbientOcclusionLevels", "r.DepthOfFieldQuality", "r.LensFlareQuality", "sg.ShadowQuality",
    "r.MotionBlurQuality", "t.MaxFPS", "r.BloomQuality", "r.TranslucencyLightingVolume",
    "fx.maxCPUParticlesPerEmitter", "r.ContactShadow",
};
// Presets (DM_GFX_PRESET=low|medium|high). A cvar not listed in a preset is left alone. "high" restates the docked [Switch] profile values,
// so it proves the write path without changing the look; "low" turns everything the preset owns down. The names must match the engine's
// real CVar names (round 17: fx.MaxCPUParticlesPerEmitter with a capital M, r.TranslucentLightingVolume — the manager-table spellings
// "fx.maxCPUParticlesPerEmitter" / "r.TranslucencyLightingVolume" never appear in any boot source, and r.ContactShadow is not in the
// preset at all: no boot source sets it). Values are short on purpose: the gfx/dp in-place rewrites cannot write a preset value longer
// than the value the game stored (the handler skips it and logs TOOLONG); the 0x36a380 substitution redirects x2 to a static host
// buffer instead, so it has no length limit.
typedef struct { const char *name, *value; } gfx_override;
static const gfx_override gfx_low[] = {
    { "r.ScreenPercentage", "66" }, { "r.PostProcessAAQuality", "2" }, { "r.MaxAnisotropy", "0" },
    { "r.Streaming.MipBias", "4" }, { "r.AmbientOcclusionLevels", "0" }, { "r.DepthOfFieldQuality", "0" },
    { "r.LensFlareQuality", "0" }, { "sg.ShadowQuality", "0" }, { "r.MotionBlurQuality", "0" },
    { "r.BloomQuality", "0" }, { "r.TranslucentLightingVolume", "0" },
    { "fx.MaxCPUParticlesPerEmitter", "5" },
};
static const gfx_override gfx_medium[] = {
    { "r.ScreenPercentage", "80" }, { "r.PostProcessAAQuality", "2" }, { "r.MaxAnisotropy", "1" },
    { "r.Streaming.MipBias", "3" }, { "r.LensFlareQuality", "0" }, { "sg.ShadowQuality", "1" },
    { "r.BloomQuality", "0" }, { "r.TranslucentLightingVolume", "0" },
    { "fx.MaxCPUParticlesPerEmitter", "10" },
};
static const gfx_override gfx_high[] = {
    { "r.ScreenPercentage", "100" }, { "r.PostProcessAAQuality", "3" }, { "r.MaxAnisotropy", "2" },
    { "r.Streaming.MipBias", "2" }, { "sg.ShadowQuality", "2" },
    { "r.TranslucentLightingVolume", "1" }, { "fx.MaxCPUParticlesPerEmitter", "15" },
};
enum { GFX_MODE_OBSERVE = 0, GFX_MODE_LOW, GFX_MODE_MEDIUM, GFX_MODE_HIGH };
static int g_gfx_mode;   // set once at install from DM_GFX_PRESET; read by the trap handler (never written after install)
static const gfx_override *gfx_preset(int mode, int *n) {
    switch (mode) {
    case GFX_MODE_LOW: *n = (int)(sizeof gfx_low / sizeof *gfx_low); return gfx_low;
    case GFX_MODE_MEDIUM: *n = (int)(sizeof gfx_medium / sizeof *gfx_medium); return gfx_medium;
    case GFX_MODE_HIGH: *n = (int)(sizeof gfx_high / sizeof *gfx_high); return gfx_high;
    default: *n = 0; return NULL;   // observe: no overrides
    }
}

// The preset values as NUL-terminated UTF-32, in static host memory. The 0x36a380 substitution redirects x2 to one of these
// buffers, so the engine's own Set reads the preset value through the redirected pointer: host and guest share the process
// address space, and 0x36a380 only reads through x2 (it never stores the pointer — Set copies the value into the CVar), so a
// host buffer is as good as a guest one. The buffers are const (never written after load), so concurrent trap threads cannot
// race on them. One buffer per distinct preset value (11 values across the three presets); a preset value with no buffer here
// is never substituted (H36_ACT_NOBUF), and install logs it — see dm_hook_install.
static const uint32_t h36_u32_0[] = { '0', 0 };
static const uint32_t h36_u32_1[] = { '1', 0 };
static const uint32_t h36_u32_2[] = { '2', 0 };
static const uint32_t h36_u32_3[] = { '3', 0 };
static const uint32_t h36_u32_4[] = { '4', 0 };
static const uint32_t h36_u32_5[] = { '5', 0 };
static const uint32_t h36_u32_10[] = { '1', '0', 0 };
static const uint32_t h36_u32_15[] = { '1', '5', 0 };
static const uint32_t h36_u32_66[] = { '6', '6', 0 };
static const uint32_t h36_u32_80[] = { '8', '0', 0 };
static const uint32_t h36_u32_100[] = { '1', '0', '0', 0 };
static const struct { const char *ascii; const uint32_t *u32; } h36_u32tab[] = {
    { "0", h36_u32_0 }, { "1", h36_u32_1 }, { "2", h36_u32_2 }, { "3", h36_u32_3 },
    { "4", h36_u32_4 }, { "5", h36_u32_5 }, { "10", h36_u32_10 }, { "15", h36_u32_15 },
    { "66", h36_u32_66 }, { "80", h36_u32_80 }, { "100", h36_u32_100 },
};
static const uint32_t *h36_u32_for(const char *ascii) {   // signal context: strcmp only, no allocation
    for (size_t i = 0; i < sizeof h36_u32tab / sizeof *h36_u32tab; i++)
        if (!strcmp(ascii, h36_u32tab[i].ascii)) return h36_u32tab[i].u32;
    return NULL;
}
// The three preset CVars the device profile never sets directly: their effective values come from the sg.* delegates, which
// set them through 0x36a380 at flags 0x01000000 (SetByCode) — r.PostProcessAAQuality '3' from sg.AntiAliasingQuality=2,
// r.LensFlareQuality '2' and r.BloomQuality '5' from sg.PostProcessQuality=2 (observe log-2b5df3b8.txt). They substitute at
// 0x01000000; the other nine substitute at 0x04000000 (device profile).
static const char *const h36_delegate[3] = {
    "r.PostProcessAAQuality", "r.LensFlareQuality", "r.BloomQuality",
};

static int hook_read(uint64_t a, void *out, size_t n) {   // a read that fails instead of faulting (vm_read_overwrite)
    vm_size_t got = 0;
    return vm_read_overwrite(mach_task_self(), (vm_address_t)a, n, (vm_address_t)out, &got) == KERN_SUCCESS && got == n;
}
static int hook_write(uint64_t a, const void *src, size_t n) {   // a write that fails instead of faulting: the stp target is game heap,
// normally writable, but a write-protected page (the dirty tracker, or the rect watchpoint in the same session) must not fault inside the
// signal handler. On other platforms the direct store stands (this diagnostic targets Android)
#ifdef __ANDROID__
    struct iovec l = { (void *)src, n }, r = { (void *)(uintptr_t)a, n };
    return process_vm_writev(getpid(), &l, 1, &r, 1, 0) == (ssize_t)n;
#else
    memcpy((void *)(uintptr_t)a, src, n);
    return 1;
#endif
}
static void hook_commit(const hook_rec *r) {   // signal context: an atomic claim and plain stores, as the watchpoint does
    uint64_t s = atomic_fetch_add_explicit(&g_seq, 1, memory_order_relaxed);
    hook_rec *d = &g_ring[s & (HOOK_RING - 1)];
    atomic_store_explicit(&d->ready, 0, memory_order_relaxed);
    d->seq = s; d->obj = r->obj; d->subobj = r->subobj; d->callee = r->callee; d->lr = r->lr;
    d->w = r->w; d->h = r->h; d->offx = r->offx; d->offy = r->offy;
    d->s0bits = r->s0bits; d->s1bits = r->s1bits;
    d->gate = r->gate; d->has_wh = r->has_wh; d->store_ok = r->store_ok;
    atomic_store_explicit(&d->ready, 1, memory_order_release);
}
static void e6_commit(const e6_rec *r) {   // signal context: an atomic claim and plain stores, as hook_commit does
    uint64_t s = atomic_fetch_add_explicit(&e6_seq, 1, memory_order_relaxed);
    e6_rec *d = &e6_ring[s & (E6_RING - 1)];
    atomic_store_explicit(&d->ready, 0, memory_order_relaxed);
    d->seq = s; d->obj = r->obj; d->arg0 = r->arg0; d->lr = r->lr;
    d->w = r->w; d->h = r->h; d->m00bits = r->m00bits; d->m11bits = r->m11bits;
    memcpy(d->mat, r->mat, sizeof d->mat);
    d->flag = r->flag; d->cached = r->cached; d->alt = r->alt; d->has_mat = r->has_mat;
    atomic_store_explicit(&d->ready, 1, memory_order_release);
}
static void s64_commit(const s64_rec *r) {   // signal context: an atomic claim and plain stores, as hook_commit does
    uint64_t s = atomic_fetch_add_explicit(&s64_seq, 1, memory_order_relaxed);
    s64_rec *d = &s64_ring[s & (S64_RING - 1)];
    atomic_store_explicit(&d->ready, 0, memory_order_relaxed);
    d->seq = s; d->obj = r->obj; d->lr = r->lr; d->dpi = r->dpi;
    d->flags = r->flags;
    d->f588 = r->f588; d->f58c = r->f58c; d->f598 = r->f598; d->f59c = r->f59c; d->f5b8 = r->f5b8;
    d->wb = r->wb; d->hb = r->hb; d->has_fields = r->has_fields;
    atomic_store_explicit(&d->ready, 1, memory_order_release);
}
static void gfx_commit(const gfx_rec *r) {   // signal context: an atomic claim and plain stores, as hook_commit does
    uint64_t s = atomic_fetch_add_explicit(&gfx_seq, 1, memory_order_relaxed);
    gfx_rec *d = &gfx_ring[s & (GFX_RING - 1)];
    atomic_store_explicit(&d->ready, 0, memory_order_relaxed);
    d->seq = s; d->kind = r->kind; d->action = r->action; d->obj = r->obj; d->lr = r->lr;
    memcpy(d->name, r->name, sizeof d->name);
    memcpy(d->oldv, r->oldv, sizeof d->oldv);
    memcpy(d->newv, r->newv, sizeof d->newv);
    atomic_store_explicit(&d->ready, 1, memory_order_release);
}
static void dp_commit(const dp_rec *r) {   // signal context: an atomic claim and plain stores, as hook_commit does
    uint64_t s = atomic_fetch_add_explicit(&dp_seq, 1, memory_order_relaxed);
    dp_rec *d = &dp_ring[s & (DP_RING - 1)];
    atomic_store_explicit(&d->ready, 0, memory_order_relaxed);
    d->seq = s; d->kind = r->kind; d->action = r->action; d->obj = r->obj; d->lr = r->lr;
    d->count = r->count; d->emu_ok = r->emu_ok;
    memcpy(d->name, r->name, sizeof d->name);
    memcpy(d->oldv, r->oldv, sizeof d->oldv);
    memcpy(d->newv, r->newv, sizeof d->newv);
    atomic_store_explicit(&d->ready, 1, memory_order_release);
}
static void h36_commit(const h36_rec *r) {   // signal context: an atomic claim and plain stores, as hook_commit does
    uint64_t s = atomic_fetch_add_explicit(&h36_seq, 1, memory_order_relaxed);
    h36_rec *d = &h36_ring[s & (H36_RING - 1)];
    atomic_store_explicit(&d->ready, 0, memory_order_relaxed);
    d->seq = s; d->action = r->action; d->lr = r->lr;
    d->flags = r->flags; d->mode = r->mode;
    memcpy(d->name, r->name, sizeof d->name);
    memcpy(d->value, r->value, sizeof d->value);
    memcpy(d->newv, r->newv, sizeof d->newv);
    atomic_store_explicit(&d->ready, 1, memory_order_release);
}
// Read a guest FString (UE4: { TCHAR* Data @0, int32 Num @8, int32 Max @12 }, TCHAR = 32-bit wchar_t) as ASCII. Returns the character
// count (excluding the NUL) or -1 when the header or the buffer cannot be read, the length is out of range, or a character is not
// printable ASCII. The buffer is always NUL-terminated. Signal context: hook_read only, no allocation.
static int gfx_read_fstr(uint64_t fs, char *out, size_t cap) {
    uint64_t hdr[2];
    if (!hook_read(fs, hdr, sizeof hdr)) return -1;
    uint64_t data = hdr[0];
    int32_t num = (int32_t)(hdr[1] & 0xffffffffu);   // Num is the low word at +8 (Max sits at +12)
    if (!data || num <= 0 || num > 64) return -1;
    if ((size_t)num > cap) return -1;   // Num counts the NUL: cap covers the characters + NUL
    uint32_t w[64];
    if (!hook_read(data, w, (size_t)num * 4)) return -1;
    for (int32_t i = 0; i < num - 1; i++) {
        if (w[i] < 0x20 || w[i] > 0x7e) return -1;
        out[i] = (char)w[i];
    }
    if (w[num - 1] != 0) return -1;   // not NUL-terminated: not a string we trust
    out[num - 1] = '\0';
    return num - 1;
}
// Read a guest TCHAR* (a NUL-terminated UTF-32 string, NOT an FString: no header, no length — the string ends at the first
// zero wchar) as ASCII. Returns the character count (excluding the NUL) or -1 when the pointer is null, the buffer cannot be
// read, no NUL appears within the cap, or a character is not printable ASCII. The buffer is always NUL-terminated. The read
// is chunked (8 wchars at a time) so a short string near a page boundary does not over-read into an unmapped page. Signal
// context: hook_read only, no allocation.
static int gfx_read_tstr(uint64_t ts, char *out, size_t cap) {
    if (!ts || cap < 2) return -1;
    if (cap > 64) cap = 64;
    size_t n = 0;
    while (n < cap) {
        size_t chunk = cap - n > 8 ? 8 : cap - n;
        uint32_t w[8];
        if (!hook_read(ts + n * 4, w, chunk * 4)) return -1;
        for (size_t i = 0; i < chunk; i++) {
            if (w[i] == 0) { out[n] = '\0'; return (int)n; }
            if (w[i] < 0x20 || w[i] > 0x7e) return -1;
            out[n++] = (char)w[i];
        }
    }
    return -1;   // no NUL within the cap: not a string we trust
}
// Rewrite a guest FString's value in place: the new wchar string + NUL into its Data buffer, then the new length into Num (the helper
// copies Num-1 characters, so a stale Num would carry the old tail). Only call when strlen(newval)+1 <= Num (checked by the caller).
// Returns 1 when both writes landed.
static int gfx_write_fstr(uint64_t fs, const char *newval) {
    uint64_t hdr[2];
    if (!hook_read(fs, hdr, sizeof hdr)) return 0;
    uint64_t data = hdr[0];
    if (!data) return 0;
    size_t len = strlen(newval);
    if (len + 1 > 16) return 0;
    uint32_t w[16];
    for (size_t i = 0; i <= len; i++) w[i] = (uint32_t)(unsigned char)newval[i];   // the NUL terminator included
    if (!hook_write(data, w, (len + 1) * 4)) return 0;
    int32_t num = (int32_t)(len + 1);
    return hook_write(fs + 8, &num, 4);
}
// The LDP/STP (signed immediate) form both displaced pair instructions use: rt/rt2, the base register, and the byte offset (imm7 * scale).
// watch_decode (probe.c) decodes the same form for stores, but it is static there and refuses loads, so the decode lives here.
static void ldstp_decode(uint32_t in, unsigned *rt, unsigned *rt2, unsigned *rn, int64_t *imm) {
    *rt = in & 31; *rt2 = (in >> 10) & 31; *rn = (in >> 5) & 31;
    int32_t imm7 = (int32_t)((in >> 15) & 0x7f); if (imm7 & 0x40) imm7 -= 0x80;
    *imm = (int64_t)imm7 * (in & (1u << 31) ? 8 : 4);
}
static uint64_t reg_get(_STRUCT_ARM_THREAD_STATE64 *ss, unsigned r) {   // a base register: 31 = sp (the XN macro of watch_decode)
    return r == 31 ? ss->__sp : r == 29 ? ss->__fp : r == 30 ? ss->__lr : ss->__x[r];
}
static uint64_t gpr_get(_STRUCT_ARM_THREAD_STATE64 *ss, unsigned r) {   // a general register: 31 = xzr (not sp, as in reg_get)
    return r == 31 ? 0 : r == 29 ? ss->__fp : r == 30 ? ss->__lr : ss->__x[r];
}
static void gpr_set(_STRUCT_ARM_THREAD_STATE64 *ss, unsigned r, uint64_t v) {   // the matching write (31 = xzr: no-op)
    if (r == 31) return; else if (r == 29) ss->__fp = v; else if (r == 30) ss->__lr = v; else ss->__x[r] = v;
}
static int e6_trap(ucontext_t *uc, int eid) {   // the 0xe6e6ac dispatch (hook_trap): 1 = handled, resume at the emulated pc
    _STRUCT_ARM_THREAD_STATE64 *ss = DM_UC_SS(uc);
    uint64_t pc = ss->__pc;
    uint32_t orig = e6_points[eid].orig;
    switch (eid) {
    case E6_ENTRY: {   // mov x19, x1: stage the view object (x1), the other argument (x0), the caller, the flag and the cached matrix
        e6_st.obj = ss->__x[1];
        e6_st.arg0 = ss->__x[0];
        e6_st.lr = ss->__lr;
        uint8_t flag = 0;
        e6_st.flag = hook_read(e6_st.obj + 0x844, &flag, 1) ? flag : 0;
        e6_st.has_mat = hook_read(e6_st.obj + 0x850, e6_st.mat, 64);
        e6_st.staged = 1;
        gpr_set(ss, orig & 31, gpr_get(ss, (orig >> 16) & 31));   // emulate the move (Rd = Rm)
        ss->__pc = pc + 4;
        break; }
    case E6_CBZ: {   // cbz/cbnz w8, target: stage the dims (w24, w9) and the path (taken: compute; fall through: cached), then branch
        unsigned rt = orig & 31;
        int32_t imm19 = (int32_t)((orig >> 5) & 0x7ffff); if (imm19 & 0x40000) imm19 -= 0x80000;
        e6_st.w = (int32_t)ss->__x[24];
        e6_st.h = (int32_t)ss->__x[9];
        uint64_t v = gpr_get(ss, rt);
        int zero = (orig & (1u << 31)) ? v == 0 : (uint32_t)v == 0;   // sf set: a 64-bit test; clear: 32-bit
        int take = (orig & (1u << 24)) ? !zero : zero;   // bit 24 set: cbnz; clear: cbz
        e6_st.cached = !take;
        ss->__pc = take ? pc + (uint64_t)imm19 * 4 : pc + 4;
        break; }
    case E6_TERMS: {   // ldrb w8, [x19, #0x330]: the join point — read the terms (m00 from [sp, #0x28], m11 from w20, the branch
        // selector from [x19, #0x318]) and commit the call; emulate the load
        unsigned rt = orig & 31, rn = (orig >> 5) & 31;
        uint64_t addr = reg_get(ss, rn) + ((orig >> 10) & 0xfff);   // LDRB unsigned offset: imm12, no scaling
        uint8_t b = 0;
        hook_read(addr, &b, 1);
        gpr_set(ss, rt, b);   // an 8-bit load zero-extends
        if (e6_st.staged) {
            e6_rec r; memset(&r, 0, sizeof r);
            r.obj = e6_st.obj; r.arg0 = e6_st.arg0; r.lr = e6_st.lr;
            r.w = e6_st.w; r.h = e6_st.h;
            r.flag = e6_st.flag; r.cached = e6_st.cached; r.has_mat = e6_st.has_mat;
            memcpy(r.mat, e6_st.mat, sizeof r.mat);
            uint32_t m00 = 0;
            hook_read(ss->__sp + 0x28, &m00, 4);
            r.m00bits = m00; r.m11bits = (uint32_t)ss->__x[20];
            uint8_t alt = 0;
            hook_read(ss->__x[19] + 0x318, &alt, 1);
            r.alt = alt;
            e6_commit(&r);
            e6_st.staged = 0;
        }
        ss->__pc = pc + 4;
        break; }
    }
    return 1;
}
static int s64_trap(ucontext_t *uc, int sid) {   // the 0x64ea18 dispatch (hook_trap): 1 = handled, resume at the emulated pc
    _STRUCT_ARM_THREAD_STATE64 *ss = DM_UC_SS(uc);
    uint64_t pc = ss->__pc;
    uint32_t orig = s64_points[sid].orig;
    switch (sid) {
    case S64_ENTRY: {   // mov x19, x0: stage the object (x0), the caller (live lr, still the entry value), the flags and the size fields
        s64_st.obj = ss->__x[0];
        s64_st.lr = ss->__lr;
        uint32_t blk[0x13];   // [obj+0x570..0x5bc]: the flags (0x570), the four size fields (0x588/0x58c/0x598/0x59c), the H adjust (0x5b8)
        s64_st.has_fields = hook_read(s64_st.obj + 0x570, blk, sizeof blk);
        if (s64_st.has_fields) {
            s64_st.flags = blk[0x00]; s64_st.f588 = blk[0x06]; s64_st.f58c = blk[0x07];
            s64_st.f598 = blk[0x0a]; s64_st.f59c = blk[0x0b]; s64_st.f5b8 = blk[0x12];
        } else s64_st.flags = s64_st.f588 = s64_st.f58c = s64_st.f598 = s64_st.f59c = s64_st.f5b8 = 0;
        s64_st.dpi = 0;   // 0 when the DPI provider is null (or unreadable)
        hook_read(s64_st.obj + 0x6b8, &s64_st.dpi, 8);
        s64_st.staged = 1;
        gpr_set(ss, orig & 31, gpr_get(ss, (orig >> 16) & 31));   // emulate the move (Rd = Rm)
        ss->__pc = pc + 4;
        break; }
    case S64_RET: {   // ldr x19, [sp, #0x30]: read the returned (W, H) from s0/s1 and commit the call; emulate the load
        unsigned rt = orig & 31, rn = (orig >> 5) & 31;
        uint64_t addr = reg_get(ss, rn) + (((orig >> 10) & 0xfff) << 3);   // LDR x, unsigned offset: imm12 * 8
        uint64_t v = 0;
        hook_read(addr, &v, 8);
        gpr_set(ss, rt, v);
        if (s64_st.staged) {
            s64_rec r; memset(&r, 0, sizeof r);
            r.obj = s64_st.obj; r.lr = s64_st.lr; r.dpi = s64_st.dpi;
            r.flags = s64_st.flags;
            r.f588 = s64_st.f588; r.f58c = s64_st.f58c; r.f598 = s64_st.f598; r.f59c = s64_st.f59c; r.f5b8 = s64_st.f5b8;
            r.has_fields = s64_st.has_fields;
            __uint128_t *fv = (__uint128_t *)DM_UC_V(uc);
            memcpy(&r.wb, &fv[0], 4); memcpy(&r.hb, &fv[1], 4);   // s0/s1 are the low halves of q0/q1
            s64_commit(&r);
            s64_st.staged = 0;
        }
        ss->__pc = pc + 4;
        break; }
    }
    return 1;
}
static int gfx_trap(ucontext_t *uc, int gid) {   // the graphics-manager dispatch (hook_trap): 1 = handled, resume at the emulated pc
    _STRUCT_ARM_THREAD_STATE64 *ss = DM_UC_SS(uc);
    uint64_t pc = ss->__pc;
    uint32_t orig = gfx_points[gid].orig;
    if (!gfx_busy) {   // the anti-recursion guard: the logic below calls no guest code, so this never trips; if it ever did, the
        gfx_busy = 1;  // displaced instruction is still emulated below and the game runs on
        switch (gid) {
        case GFX_MGR: {   // mov x19, x0: the manager call — log the object and the caller, so the run shows when the manager applies
            gfx_rec r; memset(&r, 0, sizeof r);
            r.kind = GFX_KIND_MGR; r.obj = ss->__x[0]; r.lr = ss->__lr;
            gfx_commit(&r);
            break; }
        case GFX_APPLY: {   // sub sp, sp, #0x90: the console-exec helper — read the name (x1) and the value (x2) FStrings, rewrite
            // the value in place when the preset overrides this cvar
            gfx_rec r; memset(&r, 0, sizeof r);
            r.kind = GFX_KIND_APPLY; r.obj = ss->__x[0]; r.lr = ss->__lr;
            r.action = GFX_ACT_UNREADABLE;
            int commit = 0;
            char name[40] = { 0 }, oldv[16] = { 0 };
            if (gfx_read_fstr(ss->__x[1], name, sizeof name) >= 0 &&
                gfx_read_fstr(ss->__x[2], oldv, sizeof oldv) >= 0) {
                int known = 0;
                for (int i = 0; i < 15; i++) if (!strcmp(name, gfx_cvars[i])) { known = 1; break; }
                if (known) {
                    memcpy(r.name, name, sizeof r.name);
                    memcpy(r.oldv, oldv, sizeof r.oldv);
                    memcpy(r.newv, oldv, sizeof r.newv);
                    r.action = GFX_ACT_KEEP;   // recognised, no preset entry: applied as stored
                    int n = 0;
                    const gfx_override *ov = gfx_preset(g_gfx_mode, &n);
                    for (int i = 0; i < n; i++) {
                        if (strcmp(name, ov[i].name)) continue;
                        if (!strcmp(oldv, ov[i].value)) { r.action = GFX_ACT_SAME; break; }   // already the preset value
                        // The rewrite is in place: the preset value must fit the stored buffer (Num counts the NUL). The value
                        // FString's Num is read back here (its header was already validated by gfx_read_fstr).
                        uint64_t hdr[2];
                        if (hook_read(ss->__x[2], hdr, sizeof hdr)) {
                            int32_t num = (int32_t)(hdr[1] & 0xffffffffu);
                            if ((int32_t)strlen(ov[i].value) + 1 <= num) {
                                if (gfx_write_fstr(ss->__x[2], ov[i].value)) {
                                    memcpy(r.newv, ov[i].value, strlen(ov[i].value) + 1);
                                    r.action = GFX_ACT_REWROTE;
                                } else r.action = GFX_ACT_UNREADABLE;   // the write failed: report it, do not claim success
                            } else r.action = GFX_ACT_TOOLONG;
                        }
                        break;
                    }
                    commit = 1;
                }
                // An unrecognised name is not a manager cvar (the helper is only called from the apply routine, so this should
                // not happen): nothing is recorded and nothing is rewritten.
            } else commit = 1;   // the FStrings could not be read: record the failure (obj/lr identify the call)
            if (commit) gfx_commit(&r);
            break; }
        }
        gfx_busy = 0;
    }
    // Emulate the displaced instruction (always, even when re-entered): the mov is a register copy, the sub adjusts the stack.
    if (gid == GFX_MGR) gpr_set(ss, orig & 31, gpr_get(ss, (orig >> 16) & 31));
    else {
        uint64_t imm = (orig >> 10) & 0xfff;   // SUB (immediate): imm12, shifted 12 when sh is set
        if ((orig >> 22) & 1) imm <<= 12;
        ss->__sp -= imm;
    }
    ss->__pc = pc + 4;
    return 1;
}
static int dp_trap(ucontext_t *uc, int did) {   // the device-profile apply dispatch (hook_trap): 1 = handled, resume at the emulated pc
    _STRUCT_ARM_THREAD_STATE64 *ss = DM_UC_SS(uc);
    uint64_t pc = ss->__pc;
    (void)did;   // a single point: the entry (dp_points[DP_APPLY])
    // Emulate the displaced instruction FIRST: stp x28, x27, [sp, #-0x60]! — pre-index: sp -= 0x60, store x28 at [sp], x27 at
    // [sp+8]. The decode is checked against the word (0xa9ba6ffc: rt 28, rt2 27, rn 31 = sp, imm7 -12 * 8 = -0x60, bits 24:23 = 3 =
    // pre-index writeback). The map walk below only reads x0/lr, which the store does not touch, so the order is safe; and a
    // re-entered call still gets its instruction emulated.
    uint64_t vals[2] = { ss->__x[28], ss->__x[27] };
    uint64_t newsp = ss->__sp - 0x60;
    int emu_ok = hook_write(newsp, vals, sizeof vals);
    ss->__sp = newsp;
    ss->__pc = pc + 4;
    if (dp_busy) return 1;   // re-entered (defensive: the logic below calls no guest code, so this never trips)
    dp_busy = 1;
    // The manager is x0; lr is the caller (the prologue has not run yet, so both are the entry values).
    uint64_t mgr = ss->__x[0];
    uint64_t lr = ss->__lr;
    dp_rec call; memset(&call, 0, sizeof call);
    call.kind = DP_KIND_CALL; call.obj = mgr; call.lr = lr; call.count = -1; call.emu_ok = emu_ok;
    call.action = DP_ACT_UNREADABLE;
    // The map at [manager+0xb8]: elements pointer +0x00, bitmask pointer +0x20 ([manager+0xd8], inline at manager+0xc8 when null),
    // element count +0x28 ([manager+0xe0]). The original's own loop reads exactly these three fields (0x177385c/0x1773868/0x177392c).
    uint64_t elems = 0, bits = 0;
    int32_t count = -1;
    if (mgr && hook_read(mgr + 0xb8, &elems, 8) && hook_read(mgr + 0xd8, &bits, 8) &&
        hook_read(mgr + 0xe0, &count, 4) && count >= 0 && count <= 256) {
        call.count = count;
        call.action = DP_ACT_KEEP;
        dp_commit(&call);
        if (count > 0) {
            if (!elems) {   // a nonzero count with no elements array: the layout is not what the RE says — say so, once
                dp_rec r; memset(&r, 0, sizeof r);
                r.kind = DP_KIND_CVAR; r.action = DP_ACT_UNREADABLE; r.obj = mgr; r.lr = lr;
                dp_commit(&r);
            } else {
                uint64_t bitbase = bits ? bits : mgr + 0xc8;   // the inline fallback, as the original reads it
                int nwords = (count + 31) / 32;
                uint32_t words[8];   // 256 bits at most
                if (hook_read(bitbase, words, (size_t)nwords * 4)) {
                    int n = 0;
                    const gfx_override *ov = gfx_preset(g_gfx_mode, &n);
                    for (int32_t i = 0; i < count; i++) {
                        if (!(words[i >> 5] & (1u << (i & 31)))) continue;   // a hole in the map: no live entry
                        uint64_t keyfs = elems + (uint64_t)i * 0x28;
                        uint64_t valfs = keyfs + 0x10;
                        char name[40] = { 0 }, oldv[16] = { 0 };
                        dp_rec r; memset(&r, 0, sizeof r);
                        r.kind = DP_KIND_CVAR; r.obj = mgr; r.lr = lr;
                        int klen = gfx_read_fstr(keyfs, name, sizeof name);
                        int vlen = gfx_read_fstr(valfs, oldv, sizeof oldv);
                        if (klen < 0 || vlen < 0) { r.action = DP_ACT_UNREADABLE; dp_commit(&r); continue; }
                        memcpy(r.name, name, sizeof r.name);
                        memcpy(r.oldv, oldv, sizeof r.oldv);
                        memcpy(r.newv, oldv, sizeof r.newv);
                        r.action = DP_ACT_KEEP;   // observed (or no preset entry): applied as stored
                        for (int j = 0; j < n; j++) {
                            if (strcmp(name, ov[j].name)) continue;
                            if (!strcmp(oldv, ov[j].value)) { r.action = DP_ACT_SAME; break; }   // already the preset value
                            // The rewrite is in place: the preset value must fit the stored buffer (Num = vlen+1, NUL included).
                            if ((int32_t)strlen(ov[j].value) <= vlen) {
                                if (gfx_write_fstr(valfs, ov[j].value)) {
                                    memcpy(r.newv, ov[j].value, strlen(ov[j].value) + 1);
                                    r.action = DP_ACT_REWROTE;
                                } else r.action = DP_ACT_UNREADABLE;   // the write failed: report it, do not claim success
                            } else r.action = DP_ACT_TOOLONG;
                            break;
                        }
                        dp_commit(&r);
                    }
                } else {   // the bitmask could not be read: the entries cannot be enumerated
                    dp_rec r; memset(&r, 0, sizeof r);
                    r.kind = DP_KIND_CVAR; r.action = DP_ACT_UNREADABLE; r.obj = mgr; r.lr = lr;
                    dp_commit(&r);
                }
            }
        }
        dp_busy = 0;
        return 1;
    }
    // The map header could not be read, or the count is out of range: record the failure and run on (the original still applies
    // the profile untouched — a hook that cannot see the map must never write to it).
    dp_commit(&call);
    dp_busy = 0;
    return 1;
}
static int h36_trap(ucontext_t *uc, int hid) {   // the per-CVar apply dispatch (hook_trap): 1 = handled, resume at the emulated pc
    _STRUCT_ARM_THREAD_STATE64 *ss = DM_UC_SS(uc);
    uint64_t pc = ss->__pc;
    uint32_t orig = h36_points[hid].orig;
    // Emulate the displaced instruction FIRST: sub sp, sp, #0x60 — the decode is checked against the word (0xd10183ff: sf 1,
    // op 1 = SUB, imm12 0x60, sh 0, rn 31 = sp, rd 31 = sp). The logic below only reads x1/x2/w3/w4/lr (and the substitution
    // writes x2), which the sub does not touch, so the order is safe; and a re-entered call still gets its instruction emulated.
    uint64_t imm = (orig >> 10) & 0xfff;   // SUB (immediate): imm12, shifted 12 when sh is set
    if ((orig >> 22) & 1) imm <<= 12;
    ss->__sp -= imm;
    ss->__pc = pc + 4;
    if (h36_busy) return 1;   // re-entered (defensive: the logic below calls no guest code, so this never trips)
    h36_busy = 1;
    // Read the name (x1) and the value (x2) as TCHAR* and log them with the flags (w3) and the mode (w4). In preset modes
    // (low/medium/high) a matching CVar is substituted: x2 is redirected to a static host UTF-32 buffer holding the preset
    // value, so the engine's own Set applies the preset with the correct flags, locking, delegates and ordering. Observe mode
    // only reads. Nothing is ever written to guest memory — the substitution only changes a register in the saved state.
    h36_rec r; memset(&r, 0, sizeof r);
    r.action = H36_ACT_UNREADABLE;
    r.lr = ss->__lr;
    r.flags = (uint32_t)ss->__x[3];
    r.mode = (uint32_t)ss->__x[4];
    char name[64] = { 0 }, value[64] = { 0 };
    if (gfx_read_tstr(ss->__x[1], name, sizeof name) >= 0 &&
        gfx_read_tstr(ss->__x[2], value, sizeof value) >= 0) {
        memcpy(r.name, name, sizeof r.name);
        memcpy(r.value, value, sizeof r.value);
        memcpy(r.newv, value, sizeof r.newv);
        r.action = H36_ACT_KEEP;
        // Substitution: preset modes only, and only when the name is in the active preset, the flags match the CVar's group
        // (0x04000000 device profile, 0x01000000 for the three sg.*-delegate CVars), and the caller is the profile loop (lr
        // main+0x17728ec) or the config applier (lr main+0x368d94) — the early ConsoleVariables.ini site (lr main+0x364868)
        // never sets a preset CVar and is excluded. The lr is absolute: compare the offset from the module base.
        if (g_gfx_mode != GFX_MODE_OBSERVE) {
            int n = 0;
            const gfx_override *ov = gfx_preset(g_gfx_mode, &n);
            for (int i = 0; i < n; i++) {
                if (strcmp(name, ov[i].name)) continue;
                int delegate = 0;
                for (int j = 0; j < 3; j++) if (!strcmp(name, h36_delegate[j])) { delegate = 1; break; }
                uint32_t want = delegate ? 0x01000000u : 0x04000000u;
                uint64_t lroff = r.lr - (uint64_t)g_base;
                if (r.flags != want || (lroff != 0x17728ec && lroff != 0x368d94)) break;   // wrong source: leave it
                if (!strcmp(value, ov[i].value)) { r.action = H36_ACT_SAME; break; }   // already the preset value
                const uint32_t *u32 = h36_u32_for(ov[i].value);
                if (!u32) { r.action = H36_ACT_NOBUF; break; }   // no UTF-32 buffer for this value: never substitute
                ss->__x[2] = (uint64_t)(uintptr_t)u32;   // redirect x2 to the preset value (a register write, not a guest write)
                memcpy(r.newv, ov[i].value, strlen(ov[i].value) + 1);
                r.action = H36_ACT_SUBST;
                break;
            }
        }
    }
    h36_commit(&r);
    h36_busy = 0;
    return 1;
}
static int hook_trap(ucontext_t *uc) {   // the SIGTRAP hook (crash.c): 1 = handled, resume at the emulated pc; anything else chains on
    if (!atomic_load_explicit(&g_installed, memory_order_acquire)) return g_prev ? g_prev(uc) : 0;
    uint64_t pc = DM_UC_SS(uc)->__pc;
    int id = -1;
    if (g_hooked_1c) for (int i = 0; i < HOOK_N; i++) if (pc == g_at[i]) { id = i; break; }
    if (id < 0) {   // not a 1ced80c point: try the projection, size-getter and graphics hooks, else chain on
        int eid = -1;
        if (g_hooked_e6) for (int i = 0; i < E6_N; i++) if (pc == e6_at[i]) { eid = i; break; }
        if (eid >= 0) return e6_trap(uc, eid);
        int sid = -1;
        if (g_hooked_64) for (int i = 0; i < S64_N; i++) if (pc == s64_at[i]) { sid = i; break; }
        if (sid >= 0) return s64_trap(uc, sid);
        int gid = -1;
        if (g_hooked_gfx) for (int i = 0; i < GFX_N; i++) if (pc == gfx_at[i]) { gid = i; break; }
        if (gid >= 0) return gfx_trap(uc, gid);
        int did = -1;
        if (g_hooked_dp) for (int i = 0; i < DP_N; i++) if (pc == dp_at[i]) { did = i; break; }
        if (did >= 0) return dp_trap(uc, did);
        int hid = -1;
        if (g_hooked_36) for (int i = 0; i < H36_N; i++) if (pc == h36_at[i]) { hid = i; break; }
        if (hid < 0) return g_prev ? g_prev(uc) : 0;
        return h36_trap(uc, hid);
    }
    _STRUCT_ARM_THREAD_STATE64 *ss = DM_UC_SS(uc);
    uint32_t orig = k_points[id].orig;
    switch (id) {
    case HOOK_GATE: {   // tbz w0, #0, target: log the gate result, then take the branch or fall through as the bit says
        unsigned bit = (orig >> 19) & 31;
        int32_t imm14 = (int32_t)((orig >> 5) & 0x3fff); if (imm14 & 0x2000) imm14 -= 0x4000;
        g_st.gate = (ss->__x[0] & (1ull << bit)) != 0;
        g_st.obj = ss->__x[19]; g_st.subobj = 0; g_st.callee = 0; g_st.s0bits = g_st.s1bits = 0; g_st.has_wh = 0;
        // lr is clobbered by the gate's own bl (it holds this tbz's address), so the caller comes from the stack: the prologue saved x30
        // at [sp, #0x68] (stp x29, x30, [sp, #0x60])
        g_st.lr = 0; hook_read(ss->__sp + 0x68, &g_st.lr, 8);
        if (g_st.gate) { ss->__pc = pc + 4; break; }
        hook_rec r; memset(&r, 0, sizeof r);
        r.gate = 0; r.obj = g_st.obj; r.lr = g_st.lr;
        uint32_t wh[2];
        r.has_wh = hook_read(g_st.obj + 0x78, wh, 8);
        if (r.has_wh) { r.w = (int32_t)wh[0]; r.h = (int32_t)wh[1]; }
        hook_commit(&r);
        ss->__pc = pc + (uint64_t)imm14 * 4;
        break; }
    case HOOK_BLR: {   // blr x8: log the callee (x8) and the sub-object (x0); lr = the return, pc = the target
        unsigned rn = (orig >> 5) & 31;
        g_st.callee = reg_get(ss, rn);
        g_st.subobj = ss->__x[0];
        ss->__lr = pc + 4;
        ss->__pc = g_st.callee;
        break; }
    case HOOK_LDP: {   // ldp w8, w9, [x19, #0x78]: log s0/s1 (the scale factors, just returned); emulate the load
        unsigned rt, rt2, rn; int64_t imm;
        ldstp_decode(orig, &rt, &rt2, &rn, &imm);
        uint64_t obj = reg_get(ss, rn);
        uint32_t wh[2] = { 0, 0 };
        g_st.has_wh = hook_read(obj + (uint64_t)imm, wh, 8);
        ss->__x[rt] = wh[0]; ss->__x[rt2] = wh[1];   // 32-bit loads zero-extend
        __uint128_t *v = (__uint128_t *)DM_UC_V(uc);
        memcpy(&g_st.s0bits, &v[0], 4); memcpy(&g_st.s1bits, &v[1], 4);
        g_st.obj = obj; g_st.w = (int32_t)wh[0]; g_st.h = (int32_t)wh[1];
        ss->__pc = pc + 4;
        break; }
    case HOOK_STP: {   // stp w10, w8, [x19, #0x80]: log the offsets; emulate the store; commit the call
        unsigned rt, rt2, rn; int64_t imm;
        ldstp_decode(orig, &rt, &rt2, &rn, &imm);
        uint64_t obj = reg_get(ss, rn);
        uint32_t off[2] = { (uint32_t)ss->__x[rt], (uint32_t)ss->__x[rt2] };
        int ok = hook_write(obj + (uint64_t)imm, off, 8);
        hook_rec r; memset(&r, 0, sizeof r);
        r.gate = 1; r.obj = obj; r.subobj = g_st.subobj; r.callee = g_st.callee; r.lr = g_st.lr;
        r.s0bits = g_st.s0bits; r.s1bits = g_st.s1bits; r.w = g_st.w; r.h = g_st.h; r.has_wh = g_st.has_wh;
        r.offx = (int32_t)off[0]; r.offy = (int32_t)off[1]; r.store_ok = ok;
        hook_commit(&r);
        ss->__pc = pc + 4;
        break; }
    }
    return 1;
}
static void *hook_flush(void *p) {   // drain the ring to the log: full lines for the first 50 calls, then a histogram every 10 s
    (void)p;
    uint64_t next = 0, logged = 0, total = 0, gate_false = 0, last_hist = 0;
    struct { uint64_t obj, callee; int32_t w, h, offx, offy; uint32_t s0b, s1b; uint64_t n; } hist[64];
    int nhist = 0;
    uint64_t other = 0;
    for (;;) {
        usleep(200000);
        uint64_t upto = atomic_load_explicit(&g_seq, memory_order_acquire);
        while (next < upto) {
            hook_rec *r = &g_ring[next & (HOOK_RING - 1)];
            if (!atomic_load_explicit(&r->ready, memory_order_acquire)) break;   // not published yet; retry next round
            uint64_t s1 = r->seq;
            hook_rec c = *r;
            if (r->seq != s1) break;   // torn by an overwrite; retry next round
            if (s1 != next) {
                if (s1 > next) { LOG("[RED] hook: ring overran, %llu records lost", (unsigned long long)(s1 - next)); next = s1; continue; }
                break;   // a stale slot: the record for next is not published yet
            }
            next++; total++;
            float f0, f1; memcpy(&f0, &c.s0bits, 4); memcpy(&f1, &c.s1bits, 4);
            if (logged < 50) {
                logged++;
                if (c.gate) {
                    char cb[192], lb[192];
                    crash_addr_str(cb, sizeof cb, c.callee); crash_addr_str(lb, sizeof lb, c.lr);
                    LOG("hook: call #%llu obj %#llx W %d H %d s0 %g (%08x) s1 %g (%08x) offX %d offY %d callee %s lr %s subobj %#llx%s",
                        (unsigned long long)total, (unsigned long long)c.obj, c.w, c.h, f0, c.s0bits, f1, c.s1bits,
                        c.offx, c.offy, cb, lb, (unsigned long long)c.subobj, c.store_ok ? "" : " [store skipped]");
                } else {
                    LOG("hook: call #%llu obj %#llx W %d H %d gate 0 (else branch)", (unsigned long long)total,
                        (unsigned long long)c.obj, c.w, c.h);
                }
            }
            if (!c.gate) { gate_false++; continue; }
            int k = 0;
            while (k < nhist && (hist[k].obj != c.obj || hist[k].w != c.w || hist[k].h != c.h || hist[k].offx != c.offx ||
                   hist[k].offy != c.offy || hist[k].s0b != c.s0bits || hist[k].s1b != c.s1bits || hist[k].callee != c.callee)) k++;
            if (k < nhist) hist[k].n++;
            else if (nhist < 64) { hist[nhist].obj = c.obj; hist[nhist].callee = c.callee; hist[nhist].w = c.w; hist[nhist].h = c.h;
                hist[nhist].offx = c.offx; hist[nhist].offy = c.offy; hist[nhist].s0b = c.s0bits; hist[nhist].s1b = c.s1bits; hist[nhist].n = 1; nhist++; }
            else other++;
        }
        uint64_t now = now_ns();
        if (total && now - last_hist >= 10000000000ull) {
            last_hist = now;
            LOG("hook: histogram (%llu calls, %llu gate-false):", (unsigned long long)total, (unsigned long long)gate_false);
            for (int k = 0; k < nhist; k++) {
                float f0, f1; memcpy(&f0, &hist[k].s0b, 4); memcpy(&f1, &hist[k].s1b, 4);
                char cb[192]; crash_addr_str(cb, sizeof cb, hist[k].callee);
                LOG("hook:   obj %#llx W %d H %d s0 %g s1 %g offX %d offY %d callee %s: %llu calls",
                    (unsigned long long)hist[k].obj, hist[k].w, hist[k].h, f0, f1, hist[k].offx, hist[k].offy, cb, (unsigned long long)hist[k].n);
            }
            if (other) LOG("hook:   other tuples: %llu calls", (unsigned long long)other);
        }
    }
    return NULL;
}
static void *e6_flush(void *p) {   // drain the ring to the log: full lines for the first 50 calls and the first call of each new
    (void)p;                       // histogram bucket (a view that appears later still gets its matrix printed), then a histogram every 10 s
    uint64_t next = 0, logged = 0, total = 0, ncached = 0, last_hist = 0;
    struct { uint64_t obj; int flag, cached, alt, w, h; uint32_t m00b, m11b, mat[16]; int terms_same, mat_same; uint64_t n; } hist[64];
    int nhist = 0;
    uint64_t other = 0;
    for (;;) {
        usleep(200000);
        uint64_t upto = atomic_load_explicit(&e6_seq, memory_order_acquire);
        while (next < upto) {
            e6_rec *r = &e6_ring[next & (E6_RING - 1)];
            if (!atomic_load_explicit(&r->ready, memory_order_acquire)) break;   // not published yet; retry next round
            uint64_t s1 = r->seq;
            e6_rec c = *r;
            if (r->seq != s1) break;   // torn by an overwrite; retry next round
            if (s1 != next) {
                if (s1 > next) { LOG("[RED] hook-e6: ring overran, %llu records lost", (unsigned long long)(s1 - next)); next = s1; continue; }
                break;   // a stale slot: the record for next is not published yet
            }
            next++; total++;
            if (c.cached) ncached++;
            float f00, f11; memcpy(&f00, &c.m00bits, 4); memcpy(&f11, &c.m11bits, 4);
            int k = 0;
            while (k < nhist && (hist[k].obj != c.obj || hist[k].flag != c.flag || hist[k].cached != c.cached ||
                   hist[k].alt != c.alt || hist[k].w != c.w || hist[k].h != c.h)) k++;
            int is_new = 0;
            if (k < nhist) {
                hist[k].n++;
                if (hist[k].m00b != c.m00bits || hist[k].m11b != c.m11bits) hist[k].terms_same = 0;
                if (memcmp(hist[k].mat, c.mat, sizeof hist[k].mat)) hist[k].mat_same = 0;
            } else if (nhist < 64) {
                hist[nhist].obj = c.obj; hist[nhist].flag = c.flag; hist[nhist].cached = c.cached; hist[nhist].alt = c.alt;
                hist[nhist].w = c.w; hist[nhist].h = c.h; hist[nhist].m00b = c.m00bits; hist[nhist].m11b = c.m11bits;
                memcpy(hist[nhist].mat, c.mat, sizeof hist[nhist].mat);
                hist[nhist].terms_same = 1; hist[nhist].mat_same = 1; hist[nhist].n = 1; nhist++;
                is_new = 1;
            } else other++;
            if (logged < 50 || is_new) {
                if (logged < 50) logged++;
                char lb[192];
                crash_addr_str(lb, sizeof lb, c.lr);
                LOG("hook-e6: call #%llu obj %#llx arg0 %#llx lr %s W %d H %d flag %d %s alt %d m00 %g (%08x) m11 %g (%08x)%s",
                    (unsigned long long)total, (unsigned long long)c.obj, (unsigned long long)c.arg0, lb,
                    c.w, c.h, c.flag, c.cached ? "cached" : "compute", c.alt, f00, c.m00bits, f11, c.m11bits,
                    c.has_mat ? "" : " [matrix unreadable]");
                if (c.has_mat) {
                    float m[16];
                    for (int j = 0; j < 16; j++) memcpy(&m[j], &c.mat[j], 4);
                    LOG("hook-e6:   mat [%g %g %g %g; %g %g %g %g; %g %g %g %g; %g %g %g %g]",
                        m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7],
                        m[8], m[9], m[10], m[11], m[12], m[13], m[14], m[15]);
                }
            }
        }
        uint64_t now = now_ns();
        if (total && now - last_hist >= 10000000000ull) {
            last_hist = now;
            LOG("hook-e6: histogram (%llu calls, %llu cached, %llu compute):", (unsigned long long)total,
                (unsigned long long)ncached, (unsigned long long)(total - ncached));
            for (int k = 0; k < nhist; k++) {
                float f0, f1; memcpy(&f0, &hist[k].m00b, 4); memcpy(&f1, &hist[k].m11b, 4);
                LOG("hook-e6:   obj %#llx W %d H %d flag %d %s alt %d m00 %g m11 %g%s%s: %llu calls",
                    (unsigned long long)hist[k].obj, hist[k].w, hist[k].h, hist[k].flag,
                    hist[k].cached ? "cached" : "compute", hist[k].alt, f0, f1,
                    hist[k].terms_same ? "" : " [terms vary]", hist[k].mat_same ? "" : " [matrix varies]",
                    (unsigned long long)hist[k].n);
            }
            if (other) LOG("hook-e6:   other tuples: %llu calls", (unsigned long long)other);
        }
    }
    return NULL;
}
static void *s64_flush(void *p) {   // drain the ring to the log: full lines for the first 50 calls and the first call of each new
    (void)p;                       // histogram bucket (a caller that appears later still gets its fields printed), then a histogram every 10 s
    uint64_t next = 0, logged = 0, total = 0, last_hist = 0;
    struct { uint64_t lr; uint32_t flags, f588, f58c, f598, f59c, f5b8, wb, hb; int fields_same, ret_same; uint64_t n; } hist[64];
    int nhist = 0;
    uint64_t other = 0;
    for (;;) {
        usleep(200000);
        uint64_t upto = atomic_load_explicit(&s64_seq, memory_order_acquire);
        while (next < upto) {
            s64_rec *r = &s64_ring[next & (S64_RING - 1)];
            if (!atomic_load_explicit(&r->ready, memory_order_acquire)) break;   // not published yet; retry next round
            uint64_t s1 = r->seq;
            s64_rec c = *r;
            if (r->seq != s1) break;   // torn by an overwrite; retry next round
            if (s1 != next) {
                if (s1 > next) { LOG("[RED] hook-64: ring overran, %llu records lost", (unsigned long long)(s1 - next)); next = s1; continue; }
                break;   // a stale slot: the record for next is not published yet
            }
            next++; total++;
            // The histogram keys by (caller, flags): the fields and the returned size are first-seen values with [varies] flags, so a
            // caller whose size drifts does not explode the bucket count the way keying by the float values would.
            int k = 0;
            while (k < nhist && (hist[k].lr != c.lr || hist[k].flags != c.flags)) k++;
            int is_new = 0;
            if (k < nhist) {
                hist[k].n++;
                if (hist[k].f588 != c.f588 || hist[k].f58c != c.f58c || hist[k].f598 != c.f598 ||
                    hist[k].f59c != c.f59c || hist[k].f5b8 != c.f5b8) hist[k].fields_same = 0;
                if (hist[k].wb != c.wb || hist[k].hb != c.hb) hist[k].ret_same = 0;
            } else if (nhist < 64) {
                hist[nhist].lr = c.lr; hist[nhist].flags = c.flags;
                hist[nhist].f588 = c.f588; hist[nhist].f58c = c.f58c; hist[nhist].f598 = c.f598;
                hist[nhist].f59c = c.f59c; hist[nhist].f5b8 = c.f5b8; hist[nhist].wb = c.wb; hist[nhist].hb = c.hb;
                hist[nhist].fields_same = 1; hist[nhist].ret_same = 1; hist[nhist].n = 1; nhist++;
                is_new = 1;
            } else other++;
            if (logged < 50 || is_new) {
                if (logged < 50) logged++;
                char lb[192];
                crash_addr_str(lb, sizeof lb, c.lr);
                float a, b, d, e, g, w, h;
                memcpy(&a, &c.f588, 4); memcpy(&b, &c.f58c, 4); memcpy(&d, &c.f598, 4);
                memcpy(&e, &c.f59c, 4); memcpy(&g, &c.f5b8, 4); memcpy(&w, &c.wb, 4); memcpy(&h, &c.hb, 4);
                LOG("hook-64: call #%llu obj %#llx lr %s flags %08x f588 %g (%08x) f58c %g (%08x) f598 %g (%08x) f59c %g (%08x) f5b8 %g (%08x) dpi %#llx -> W %g (%08x) H %g (%08x)%s",
                    (unsigned long long)total, (unsigned long long)c.obj, lb, c.flags,
                    a, c.f588, b, c.f58c, d, c.f598, e, c.f59c, g, c.f5b8,
                    (unsigned long long)c.dpi, w, c.wb, h, c.hb, c.has_fields ? "" : " [fields unreadable]");
            }
        }
        uint64_t now = now_ns();
        if (total && now - last_hist >= 10000000000ull) {
            last_hist = now;
            LOG("hook-64: histogram (%llu calls):", (unsigned long long)total);
            for (int k = 0; k < nhist; k++) {
                float a, b, d, e, g, w, h;
                memcpy(&a, &hist[k].f588, 4); memcpy(&b, &hist[k].f58c, 4); memcpy(&d, &hist[k].f598, 4);
                memcpy(&e, &hist[k].f59c, 4); memcpy(&g, &hist[k].f5b8, 4); memcpy(&w, &hist[k].wb, 4); memcpy(&h, &hist[k].hb, 4);
                char lb[192];
                crash_addr_str(lb, sizeof lb, hist[k].lr);
                LOG("hook-64:   lr %s flags %08x f588 %g f58c %g f598 %g f59c %g f5b8 %g -> W %g H %g%s%s: %llu calls",
                    lb, hist[k].flags, a, b, d, e, g, w, h,
                    hist[k].fields_same ? "" : " [fields vary]", hist[k].ret_same ? "" : " [ret varies]",
                    (unsigned long long)hist[k].n);
            }
            if (other) LOG("hook-64:   other tuples: %llu calls", (unsigned long long)other);
        }
    }
    return NULL;
}
static const char *gfx_act_str(int a) {
    switch (a) {
    case GFX_ACT_REWROTE: return "rewrote";
    case GFX_ACT_SAME: return "same";
    case GFX_ACT_TOOLONG: return "TOO LONG (skipped)";
    case GFX_ACT_KEEP: return "kept";
    default: return "unreadable";
    }
}
static void *gfx_flush(void *p) {   // drain the ring to the log: a full line for every manager call and every cvar apply (the manager
    (void)p;                       // applies 15 cvars per call and calls are rare, so nothing is sampled), then a histogram every 10 s
    uint64_t next = 0, total = 0, mgrs = 0, rewrote = 0, last_hist = 0;
    struct { char name[40]; char oldv[16], newv[16]; int action; uint64_t n; } hist[64];
    int nhist = 0;
    uint64_t other = 0;
    for (;;) {
        usleep(200000);
        uint64_t upto = atomic_load_explicit(&gfx_seq, memory_order_acquire);
        while (next < upto) {
            gfx_rec *r = &gfx_ring[next & (GFX_RING - 1)];
            if (!atomic_load_explicit(&r->ready, memory_order_acquire)) break;   // not published yet; retry next round
            uint64_t s1 = r->seq;
            gfx_rec c = *r;
            if (r->seq != s1) break;   // torn by an overwrite; retry next round
            if (s1 != next) {
                if (s1 > next) { LOG("[RED] hook-gfx: ring overran, %llu records lost", (unsigned long long)(s1 - next)); next = s1; continue; }
                break;   // a stale slot: the record for next is not published yet
            }
            next++; total++;
            if (c.kind == GFX_KIND_MGR) {
                mgrs++;
                char lb[192];
                crash_addr_str(lb, sizeof lb, c.lr);
                LOG("hook-gfx: manager call #%llu obj %#llx lr %s", (unsigned long long)mgrs, (unsigned long long)c.obj, lb);
                continue;
            }
            if (c.action == GFX_ACT_REWROTE) rewrote++;
            if (c.name[0]) LOG("hook-gfx: apply %s: '%s' -> '%s' [%s]", c.name, c.oldv, c.newv, gfx_act_str(c.action));
            else LOG("hook-gfx: apply: FString unreadable (obj %#llx lr %#llx)", (unsigned long long)c.obj, (unsigned long long)c.lr);
            int k = 0;
            while (k < nhist && (strcmp(hist[k].name, c.name) || strcmp(hist[k].oldv, c.oldv) ||
                   strcmp(hist[k].newv, c.newv) || hist[k].action != c.action)) k++;
            if (k < nhist) hist[k].n++;
            else if (nhist < 64) { memcpy(hist[nhist].name, c.name, sizeof hist[nhist].name);
                memcpy(hist[nhist].oldv, c.oldv, sizeof hist[nhist].oldv);
                memcpy(hist[nhist].newv, c.newv, sizeof hist[nhist].newv);
                hist[nhist].action = c.action; hist[nhist].n = 1; nhist++; }
            else other++;
        }
        uint64_t now = now_ns();
        if (total && now - last_hist >= 10000000000ull) {
            last_hist = now;
            LOG("hook-gfx: histogram (%llu manager calls, %llu applies, %llu rewritten):", (unsigned long long)mgrs,
                (unsigned long long)(total - mgrs), (unsigned long long)rewrote);
            for (int k = 0; k < nhist; k++) {
                if (hist[k].name[0]) LOG("hook-gfx:   %s: '%s' -> '%s' [%s]: %llu", hist[k].name, hist[k].oldv,
                    hist[k].newv, gfx_act_str(hist[k].action), (unsigned long long)hist[k].n);
                else LOG("hook-gfx:   [unreadable FString]: %llu", (unsigned long long)hist[k].n);
            }
            if (other) LOG("hook-gfx:   other tuples: %llu applies", (unsigned long long)other);
        }
    }
    return NULL;
}
static const char *dp_act_str(int a) {
    switch (a) {
    case DP_ACT_REWROTE: return "rewrote";
    case DP_ACT_SAME: return "same";
    case DP_ACT_TOOLONG: return "TOO LONG (skipped)";
    case DP_ACT_KEEP: return "kept";
    default: return "unreadable";
    }
}
static void *dp_flush(void *p) {   // drain the ring to the log: a full line for every apply call and every cvar in its map (calls are
    (void)p;                       // rare — boot + profile changes — so nothing is sampled), then a histogram every 10 s, including the
                                   // preset CVars never seen in the map (map rewrite cannot reach those: see the deviation note at dp_points)
    uint64_t next = 0, total = 0, calls = 0, rewrote = 0, unread = 0, last_hist = 0;
    struct { char name[40]; char oldv[16], newv[16]; int action; uint64_t n; } hist[64];
    int nhist = 0;
    uint64_t other = 0;
    int nlow = (int)(sizeof gfx_low / sizeof *gfx_low);   // gfx_low holds all 12 preset CVars (the union of the three presets)
    int seen[16] = { 0 };   // 16 is headroom over the 12 preset CVars
    for (;;) {
        usleep(200000);
        uint64_t upto = atomic_load_explicit(&dp_seq, memory_order_acquire);
        while (next < upto) {
            dp_rec *r = &dp_ring[next & (DP_RING - 1)];
            if (!atomic_load_explicit(&r->ready, memory_order_acquire)) break;   // not published yet; retry next round
            uint64_t s1 = r->seq;
            dp_rec c = *r;
            if (r->seq != s1) break;   // torn by an overwrite; retry next round
            if (s1 != next) {
                if (s1 > next) { LOG("[RED] hook-dp: ring overran, %llu records lost", (unsigned long long)(s1 - next)); next = s1; continue; }
                break;   // a stale slot: the record for next is not published yet
            }
            next++; total++;
            if (c.kind == DP_KIND_CALL) {
                calls++;
                char lb[192];
                crash_addr_str(lb, sizeof lb, c.lr);
                if (c.action == DP_ACT_UNREADABLE)
                    LOG("hook-dp: apply call #%llu obj %#llx lr %s [map unreadable]%s", (unsigned long long)calls,
                        (unsigned long long)c.obj, lb, c.emu_ok ? "" : " [stp emulation write failed]");
                else
                    LOG("hook-dp: apply call #%llu obj %#llx lr %s map %d entries%s", (unsigned long long)calls,
                        (unsigned long long)c.obj, lb, c.count, c.emu_ok ? "" : " [stp emulation write failed]");
                continue;
            }
            if (c.action == DP_ACT_REWROTE) rewrote++;
            if (c.action == DP_ACT_UNREADABLE) unread++;
            if (c.name[0]) {
                LOG("hook-dp: apply %s: '%s' -> '%s' [%s]", c.name, c.oldv, c.newv, dp_act_str(c.action));
                for (int i = 0; i < nlow; i++) if (!strcmp(c.name, gfx_low[i].name)) seen[i] = 1;
            } else LOG("hook-dp: apply: FString unreadable (obj %#llx lr %#llx)", (unsigned long long)c.obj, (unsigned long long)c.lr);
            int k = 0;
            while (k < nhist && (strcmp(hist[k].name, c.name) || strcmp(hist[k].oldv, c.oldv) ||
                   strcmp(hist[k].newv, c.newv) || hist[k].action != c.action)) k++;
            if (k < nhist) hist[k].n++;
            else if (nhist < 64) { memcpy(hist[nhist].name, c.name, sizeof hist[nhist].name);
                memcpy(hist[nhist].oldv, c.oldv, sizeof hist[nhist].oldv);
                memcpy(hist[nhist].newv, c.newv, sizeof hist[nhist].newv);
                hist[nhist].action = c.action; hist[nhist].n = 1; nhist++; }
            else other++;
        }
        uint64_t now = now_ns();
        if (total && now - last_hist >= 10000000000ull) {
            last_hist = now;
            LOG("hook-dp: histogram (%llu apply calls, %llu cvars, %llu rewritten, %llu unreadable):", (unsigned long long)calls,
                (unsigned long long)(total - calls), (unsigned long long)rewrote, (unsigned long long)unread);
            for (int k = 0; k < nhist; k++) {
                if (hist[k].name[0]) LOG("hook-dp:   %s: '%s' -> '%s' [%s]: %llu", hist[k].name, hist[k].oldv,
                    hist[k].newv, dp_act_str(hist[k].action), (unsigned long long)hist[k].n);
                else LOG("hook-dp:   [unreadable FString]: %llu", (unsigned long long)hist[k].n);
            }
            if (other) LOG("hook-dp:   other tuples: %llu cvars", (unsigned long long)other);
            int nmiss = 0;
            for (int i = 0; i < nlow; i++) if (!seen[i]) nmiss++;
            if (nmiss) {
                char miss[512] = { 0 };
                size_t off = 0;
                for (int i = 0; i < nlow; i++) if (!seen[i]) {
                    int w = snprintf(miss + off, sizeof miss - off, "%s%s", off ? ", " : "", gfx_low[i].name);
                    if (w < 0 || off + (size_t)w >= sizeof miss) break;
                    off += (size_t)w;
                }
                LOG("hook-dp:   preset CVars never seen in the profile map (%d): %s", nmiss, miss);
            }
        }
    }
    return NULL;
}
// Render a preset-CVar source mask (bit 0 = profile 0x4000000, bit 1 = ini 0x3000000, bit 2 = ConsoleVariables.ini 0x5000000,
// bit 3 = code 0x01000000, bit 4 = other flags) as "profile+ini".
static void h36_src_str(uint32_t src, char *buf, size_t cap) {
    static const struct { uint32_t bit; const char *label; } tab[5] = {
        { 1u, "profile" }, { 2u, "ini" }, { 4u, "consolevars" }, { 8u, "code" }, { 16u, "other" } };
    size_t off = 0;
    buf[0] = '\0';
    for (int j = 0; j < 5; j++) if (src & tab[j].bit) {
        int w = snprintf(buf + off, cap - off, "%s%s", off ? "+" : "", tab[j].label);
        if (w < 0 || off + (size_t)w >= cap) break;
        off += (size_t)w;
    }
}
static void *h36_flush(void *p) {   // drain the ring to the log: a full line for every apply call (boot applies are bounded —
    (void)p;                       // the profile plus the ini sections are a few hundred calls at most — so nothing is sampled),
                                   // then a histogram every 10 s, including per-preset-CVar coverage (seen count + which sources
                                   // applied it + the last value seen) and the preset CVars never seen at all (the follow-up preset
                                   // round cannot reach those through this hook: see the note at h36_points)
    uint64_t next = 0, total = 0, unread = 0, subst = 0, same = 0, nobuf = 0, last_hist = 0;
    struct { char name[64]; char value[64]; uint32_t flags, mode; uint64_t n; } hist[64];
    int nhist = 0;
    uint64_t other = 0;
    int nlow = (int)(sizeof gfx_low / sizeof *gfx_low);   // gfx_low holds all 12 preset CVars (the union of the three presets)
    // Per-preset-CVar coverage: how many times it was applied, from which sources (bit 0 = profile 0x4000000, bit 1 = ini
    // 0x3000000, bit 2 = ConsoleVariables.ini 0x5000000, bit 3 = code 0x01000000, bit 4 = other flags), and the last value +
    // flags seen. The "last value" is the value the engine applied (the preset value on a substituted call).
    struct { uint64_t n; uint32_t src; char lastv[64]; uint32_t lastflags; } seen[16];   // 16 is headroom over the 12 preset CVars
    memset(seen, 0, sizeof seen);
    for (;;) {
        usleep(200000);
        uint64_t upto = atomic_load_explicit(&h36_seq, memory_order_acquire);
        while (next < upto) {
            h36_rec *r = &h36_ring[next & (H36_RING - 1)];
            if (!atomic_load_explicit(&r->ready, memory_order_acquire)) break;   // not published yet; retry next round
            uint64_t s1 = r->seq;
            h36_rec c = *r;
            if (r->seq != s1) break;   // torn by an overwrite; retry next round
            if (s1 != next) {
                if (s1 > next) { LOG("[RED] hook-36: ring overran, %llu records lost", (unsigned long long)(s1 - next)); next = s1; continue; }
                break;   // a stale slot: the record for next is not published yet
            }
            next++; total++;
            char lb[192];
            crash_addr_str(lb, sizeof lb, c.lr);
            if (c.action == H36_ACT_UNREADABLE) {
                unread++;
                LOG("hook-36: call #%llu: TCHAR unreadable (flags %08x mode %u lr %s)", (unsigned long long)total,
                    c.flags, c.mode, lb);
                continue;
            }
            if (c.action == H36_ACT_SUBST) {
                subst++;
                LOG("hook-36: SUBST call #%llu %s: '%s' -> '%s' [flags %08x mode %u] lr %s", (unsigned long long)total,
                    c.name, c.value, c.newv, c.flags, c.mode, lb);
            } else if (c.action == H36_ACT_SAME) {
                same++;
                LOG("hook-36: call #%llu %s: '%s' [flags %08x mode %u] lr %s [same]", (unsigned long long)total,
                    c.name, c.value, c.flags, c.mode, lb);
            } else if (c.action == H36_ACT_NOBUF) {
                nobuf++;
                LOG("[RED] hook-36: call #%llu %s: no UTF-32 buffer for the preset value (flags %08x mode %u lr %s)",
                    (unsigned long long)total, c.name, c.flags, c.mode, lb);
            } else {
                LOG("hook-36: call #%llu %s: '%s' [flags %08x mode %u] lr %s", (unsigned long long)total,
                    c.name, c.value, c.flags, c.mode, lb);
            }
            const char *effv = c.action == H36_ACT_SUBST ? c.newv : c.value;   // the value the engine applied
            for (int i = 0; i < nlow; i++) if (!strcmp(c.name, gfx_low[i].name)) {
                seen[i].n++;
                seen[i].src |= c.flags == 0x4000000 ? 1u : c.flags == 0x3000000 ? 2u :
                             c.flags == 0x5000000 ? 4u : c.flags == 0x01000000 ? 8u : 16u;
                memcpy(seen[i].lastv, effv, sizeof seen[i].lastv);
                seen[i].lastflags = c.flags;
            }
            int k = 0;
            while (k < nhist && (strcmp(hist[k].name, c.name) || strcmp(hist[k].value, effv) ||
                   hist[k].flags != c.flags || hist[k].mode != c.mode)) k++;
            if (k < nhist) hist[k].n++;
            else if (nhist < 64) { memcpy(hist[nhist].name, c.name, sizeof hist[nhist].name);
                memcpy(hist[nhist].value, effv, sizeof hist[nhist].value);
                hist[nhist].flags = c.flags; hist[nhist].mode = c.mode; hist[nhist].n = 1; nhist++; }
            else other++;
        }
        uint64_t now = now_ns();
        if (total && now - last_hist >= 10000000000ull) {
            last_hist = now;
            LOG("hook-36: histogram (%llu calls, %llu unreadable, %llu substituted, %llu same, %llu no-buffer):",
                (unsigned long long)total, (unsigned long long)unread, (unsigned long long)subst,
                (unsigned long long)same, (unsigned long long)nobuf);
            for (int k = 0; k < nhist; k++)
                LOG("hook-36:   %s: '%s' [flags %08x mode %u]: %llu", hist[k].name, hist[k].value,
                    hist[k].flags, hist[k].mode, (unsigned long long)hist[k].n);
            if (other) LOG("hook-36:   other tuples: %llu calls", (unsigned long long)other);
            int nseen = 0;
            for (int i = 0; i < nlow; i++) if (seen[i].n) nseen++;
            LOG("hook-36:   preset CVars seen (%d of %d):", nseen, nlow);
            for (int i = 0; i < nlow; i++) if (seen[i].n) {
                char src[48];
                h36_src_str(seen[i].src, src, sizeof src);
                LOG("hook-36:     %s: %llux [%s] last '%s' flags %08x", gfx_low[i].name,
                    (unsigned long long)seen[i].n, src, seen[i].lastv, seen[i].lastflags);
            }
            int nmiss = nlow - nseen;
            if (nmiss) {
                char miss[512] = { 0 };
                size_t off = 0;
                for (int i = 0; i < nlow; i++) if (!seen[i].n) {
                    int w = snprintf(miss + off, sizeof miss - off, "%s%s", off ? ", " : "", gfx_low[i].name);
                    if (w < 0 || off + (size_t)w >= sizeof miss) break;
                    off += (size_t)w;
                }
                LOG("hook-36:   preset CVars never seen (%d): %s", nmiss, miss);
            }
        }
    }
    return NULL;
}
static int hook_install_points(const hook_point *pts, int n, uint64_t *at, const char *env) {   // verify the words, then write the BRKs
    for (int i = 0; i < n; i++) {                                                               // (rollback on failure, as before)
        uint32_t cur = *(uint32_t *)(g_base + pts[i].off);
        if (cur != pts[i].orig) { LOG("[RED] hook: main+0x%llx holds %08x, expected %08x; %s not installed",
                                      (unsigned long long)pts[i].off, cur, pts[i].orig, env); return 0; }
        at[i] = (uint64_t)(g_base + pts[i].off);
    }
    long ps = getpagesize();
    int written = 0;
    for (int i = 0; i < n; i++) {
        uint64_t pg = at[i] & ~(uint64_t)(ps - 1);
        if (mprotect((void *)(uintptr_t)pg, (size_t)ps, PROT_READ | PROT_WRITE)) {
            LOG("[RED] hook: cannot make main+0x%llx writable; %s not installed", (unsigned long long)pts[i].off, env);
            goto rollback;
        }
        *(uint32_t *)(uintptr_t)at[i] = HOOK_BRK;
        mprotect((void *)(uintptr_t)pg, (size_t)ps, PROT_READ | PROT_EXEC);
        sys_icache_invalidate((void *)(uintptr_t)at[i], 4);
        written++;
    }
    return 1;
rollback:   // a half-installed hook would trap into the crash path: restore every BRK already written
    for (int i = 0; i < written; i++) {
        uint64_t pg = at[i] & ~(uint64_t)(ps - 1);
        if (mprotect((void *)(uintptr_t)pg, (size_t)ps, PROT_READ | PROT_WRITE)) continue;
        *(uint32_t *)(uintptr_t)at[i] = pts[i].orig;
        mprotect((void *)(uintptr_t)pg, (size_t)ps, PROT_READ | PROT_EXEC);
        sys_icache_invalidate((void *)(uintptr_t)at[i], 4);
    }
    return 0;
}
void dm_hook_install(void) {   // called from probe_install, after the modules are loaded and before any guest code runs
    static int done;
    if (done) return; done = 1;
    int want1 = 0, want6 = 0, want4 = 0, wantg = 0;
    { const char *e = getenv("DM_HOOK_1CED80C"); want1 = e && atoi(e); }
    { const char *e = getenv("DM_HOOK_E6E6AC"); want6 = e && atoi(e); }
    { const char *e = getenv("DM_HOOK_64EA18"); want4 = e && atoi(e); }
    {   // DM_GFX_PRESET=observe|low|medium|high: the graphics-manager hooks + the device-profile restore hook + the per-CVar
        // apply hook + the preset to inject (unset = off). Parsed once here; the trap handlers only read g_gfx_mode, never the environment.
        const char *e = getenv("DM_GFX_PRESET");
        if (e && *e) {
            if (!strcmp(e, "observe")) { wantg = 1; g_gfx_mode = GFX_MODE_OBSERVE; }
            else if (!strcmp(e, "low")) { wantg = 1; g_gfx_mode = GFX_MODE_LOW; }
            else if (!strcmp(e, "medium")) { wantg = 1; g_gfx_mode = GFX_MODE_MEDIUM; }
            else if (!strcmp(e, "high")) { wantg = 1; g_gfx_mode = GFX_MODE_HIGH; }
            else LOG("[RED] hook-gfx: DM_GFX_PRESET='%s' is not observe|low|medium|high; graphics hooks not installed", e);
        }
    }
    if (!want1 && !want6 && !want4 && !wantg) return;
    if (wantg) {   // every preset value must have a UTF-32 buffer, or its substitution is skipped (H36_ACT_NOBUF) — a
        for (int m = GFX_MODE_LOW; m <= GFX_MODE_HIGH; m++) {   // static configuration error, reported once here, not per call
            int n = 0; const gfx_override *ov = gfx_preset(m, &n);
            for (int i = 0; i < n; i++) if (!h36_u32_for(ov[i].value))
                LOG("[RED] hook-36: no UTF-32 buffer for preset value '%s' (%s); it will never substitute", ov[i].value, ov[i].name);
        }
    }
    module_t *m = NULL;
    for (int i = 0; i < g_nmod; i++) if (!strcmp(g_mod[i].name, "main")) m = &g_mod[i];
    if (!m) { LOG("[RED] hook: no main module; DM_HOOK_1CED80C/DM_HOOK_E6E6AC/DM_HOOK_64EA18/DM_GFX_PRESET not installed"); return; }
    if (strcmp(m->build_id, HOOK_BUILD)) { LOG("[RED] hook: main build %s is not %s; DM_HOOK_1CED80C/DM_HOOK_E6E6AC/DM_HOOK_64EA18/DM_GFX_PRESET not installed", m->build_id, HOOK_BUILD); return; }
    g_base = m->base;
    if (want1) g_hooked_1c = hook_install_points(k_points, HOOK_N, g_at, "DM_HOOK_1CED80C");
    if (want6) g_hooked_e6 = hook_install_points(e6_points, E6_N, e6_at, "DM_HOOK_E6E6AC");
    if (want4) g_hooked_64 = hook_install_points(s64_points, S64_N, s64_at, "DM_HOOK_64EA18");
    if (wantg) g_hooked_gfx = hook_install_points(gfx_points, GFX_N, gfx_at, "DM_GFX_PRESET");
    if (wantg) g_hooked_dp = hook_install_points(dp_points, DP_N, dp_at, "DM_GFX_PRESET");
    if (wantg) g_hooked_36 = hook_install_points(h36_points, H36_N, h36_at, "DM_GFX_PRESET");
    if (!g_hooked_1c && !g_hooked_e6 && !g_hooked_64 && !g_hooked_gfx && !g_hooked_dp && !g_hooked_36) return;
    g_prev = crash_trap_hook();
    crash_set_trap_hook(hook_trap);
    atomic_store_explicit(&g_installed, 1, memory_order_release);
    pthread_t th;
    if (g_hooked_1c) { pthread_create(&th, NULL, hook_flush, NULL); pthread_detach(th);
        LOG("hook: 4 breakpoints at main+0x1ced838/0x1ced84c/0x1ced850/0x1ced884 (DM_HOOK_1CED80C)"); }
    if (g_hooked_e6) { pthread_create(&th, NULL, e6_flush, NULL); pthread_detach(th);
        LOG("hook-e6: 3 breakpoints at main+0xe6e6d8/0xe6e8f0/0xe6ea84 (DM_HOOK_E6E6AC)"); }
    if (g_hooked_64) { pthread_create(&th, NULL, s64_flush, NULL); pthread_detach(th);
        LOG("hook-64: 2 breakpoints at main+0x64ea30/0x64eb00 (DM_HOOK_64EA18)"); }
    if (g_hooked_gfx) { pthread_create(&th, NULL, gfx_flush, NULL); pthread_detach(th);
        LOG("hook-gfx: 2 breakpoints at main+0x125310/0x125e90 (DM_GFX_PRESET=%s)",
            g_gfx_mode == GFX_MODE_LOW ? "low" : g_gfx_mode == GFX_MODE_MEDIUM ? "medium" :
            g_gfx_mode == GFX_MODE_HIGH ? "high" : "observe"); }
    if (g_hooked_dp) { pthread_create(&th, NULL, dp_flush, NULL); pthread_detach(th);
        LOG("hook-dp: 1 breakpoint at main+0x177383c (DM_GFX_PRESET=%s)",
            g_gfx_mode == GFX_MODE_LOW ? "low" : g_gfx_mode == GFX_MODE_MEDIUM ? "medium" :
            g_gfx_mode == GFX_MODE_HIGH ? "high" : "observe"); }
    if (g_hooked_36) { pthread_create(&th, NULL, h36_flush, NULL); pthread_detach(th);
        LOG("hook-36: 1 breakpoint at main+0x36a380 (DM_GFX_PRESET=%s%s)",
            g_gfx_mode == GFX_MODE_LOW ? "low" : g_gfx_mode == GFX_MODE_MEDIUM ? "medium" :
            g_gfx_mode == GFX_MODE_HIGH ? "high" : "observe",
            g_gfx_mode == GFX_MODE_OBSERVE ? ", observe only" : ", substituting preset values"); }
}
