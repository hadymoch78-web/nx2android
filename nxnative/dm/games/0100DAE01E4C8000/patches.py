# Story of Seasons: Grand Bazaar (0100DAE01E4C8000): the rewriter's patches of this game's own builds. dm/tools/patch_nso.py loads every profile's patches.py and applies those of a module's NSO build ID;
# CUTS, OBS and FRAME_STEP as described there ("Registered patches"). Patches of Nintendo SDK module builds are in dm/tools/sdk_patches.py
CUTS = {
  # Story of Seasons: Grand Bazaar v1.0.0 main
  '07a019c51236763f6f70339d39f420a9': [
    # Money not reduced: SpendMoney (RVA 0x2B0D6C0, IL2CPP) computes the post-purchase balance as sub w22, w8, w19 (money - price) at main+0x2b0d88c
    # and hands (before, after) to the apply/animation routine at 0x2cd4d10. Replacing the SUB with add w22, w8, wzr (a copy) makes after == before,
    # so the purchase flow (animation, completion callback, item grant) runs unchanged while the money never decreases.
    # Verified against dump.cs: PlayerMoney field 0x2c, SpendMoney(uint money, bool isNotAnim, Action complete) RVA 0x2B0D6C0, 6 direct callers.
    (0x2b0d88c, '1601134b', '16011f0b', 'money not reduced: sub w22, w8, w19 -> add w22, w8, wzr (SpendMoney new-balance computation)'),
  ],
  # Story of Seasons: Grand Bazaar v1.5.0 main (build 2026-11-17, BID 2ef3bb19...)
  # All offsets verified against the decompressed 1.5.0 .text (text_150.bin, RVA == offset) and every
  # replacement encoding verified by disassembly. patch_nso.py asserts the original bytes before cutting,
  # so a wrong offset aborts the build instead of silently corrupting the game.
  '2ef3bb19eac3b0a5697768f8a453953f': [
    # 1. Money not reduced: SpendMoney (RVA 0x2B195D0, IL2CPP, GameController/BokuMono) computes the post-purchase balance as sub w22, w8, w19 (money - price) at main+0x2b1979c.
    # Same computation as v1.0.0, verified by disassembly: ldr w8, [x8, #0x2c] (PlayerMoney field 0x2c) immediately precedes the SUB, same registers (w22=result, w8=money, w19=price),
    # same surrounding loads; all 70 differing words in the 0x400 function window vs v1.0.0 are address/offset operands only (ADRP/LDR/BL immediates), zero logic changes.
    # Verified against dump.cs 1.5.0: SpendMoney(uint money, bool isNotAnim, Action complete) RVA 0x2B195D0, get/set_PlayerMoney 0x2B14860/0x2B14950, 6 direct BL callers.
    # Replacing the SUB with add w22, w8, wzr (a copy) makes after == before, so the purchase flow runs unchanged while the money never decreases.
    (0x2b1979c, '1601134b', '16011f0b', 'money not reduced: sub w22, w8, w19 -> add w22, w8, wzr (SpendMoney new-balance computation)'),
    # 2. Stamina/HP never decreases: DamageHP(float damage, bool isDamageBuff) RVA 0x2992CD0 computes the effective damage (after buff reduction, clamped >= 0),
    # loads the current PlayerHP (ldr s0, [x8, #0x50]) and subtracts: fsub s0, s0, s8 at main+0x2992f24, then calls set_PlayerHP. Negative costs never reach
    # this point (early exit on fcmp s8, #0.0), so this function is purely the drain path; the 5 other set_PlayerHP callers are heal/set-to-max paths.
    # Replacing the FSUB with fmov s0, s0 (a copy) makes set_PlayerHP receive the unchanged HP: the player never gets tired.
    (0x2992f24, '0038281e', '0040201e', 'stamina never decreases: fsub s0, s0, s8 -> fmov s0, s0 (DamageHP new-HP computation)'),
    # 3. Watering can never empties: UseWateringPot(int useWater) RVA 0x2BD2080 loads (max, current) with ldp w9, w8, [x19, #0x4c] and subtracts:
    # sub w8, w8, w20 at main+0x2bd20c4, then clamps to [0, max] and stores. Replacing the SUB with mov w8, w8 (a copy) keeps the current water
    # unchanged. Note: fill the can once at a water source first; an empty can stays empty.
    (0x2bd20c4, '0801144b', 'e803082a', 'water never decreases: sub w8, w8, w20 -> mov w8, w8 (UseWateringPot new-water computation)'),
    # 4-8. Items never consumed: every StorageManager.Use overload subtracts the used stack from the slot count with a SUBS whose flags drive the
    # "count reached 0 -> clear the slot" branches, so the SUBS must keep setting flags. Replacing the subtrahend with wzr (subtract 0) keeps the
    # count and the flags correct (count > 0, checked by the preceding cbz): the store path always runs and stores the unchanged count.
    # The BagItemStorageManager.Use overrides just call the base Use, so patching the base covers the player bag too.
    (0x2d1af5c, '2a01136b', '2a011f6b', 'items never consumed: subs w10, w9, w19 -> subs w10, w9, wzr (StorageManager.Use(uint, int) stack)'),
    (0x2d1b1d8, '2901136b', '29011f6b', 'items never consumed: subs w9, w9, w19 -> subs w9, w9, wzr (StorageManager.Use(int slot, int) stack)'),
    (0x2d1b244, '2a01146b', '2a011f6b', 'items never consumed: subs w10, w9, w20 -> subs w10, w9, wzr (StorageManager.TryUse stack)'),
    (0x2d243ac, '4b01096b', '4b011f6b', 'items never consumed: subs w11, w10, w9 -> subs w11, w10, wzr (StorageManager.Use(ItemData) stack, site 1)'),
    (0x2d24504, '4b01096b', '4b011f6b', 'items never consumed: subs w11, w10, w9 -> subs w11, w10, wzr (StorageManager.Use(ItemData) stack, site 2)'),
    # 9. Friendship maxes on every gain: LikeabilityManager.AddPoint(uint charaId, float point, ...) delegates to the worker at 0x289D0F0, which
    # converts the gained points to int (w26) and adds them to the current Point (w27): add w9, w27, w26 at main+0x289d9f8. The result is then
    # clamped to the story level cap (csel w10, w9, w25, lt) and stored. Replacing the ADD with mov w9, #0xFFFF (65535) makes every gain land
    # exactly on the level cap, so one talk or gift maxes that villager for the current story progress. w26 is untouched, so the result reporting
    # (increased/decreased) still works.
    (0x289d9f8, '69031a0b', 'e9ff9f52', 'friendship to cap on every gain: add w9, w27, w26 -> mov w9, #0xFFFF (LikeabilityManager worker)'),
    # 10-11. Crops/grass instantly fully grown: FieldCropEntity.IsMaxGrowthLevel() RVA 0x278E2F0 and FieldGrassEntity.IsMaxGrowthLevel() RVA 0x27AA020
    # both compare the current growth level (field 0x9c) against the max and return current >= max. Replacing the function entry with
    # mov w0, #1; ret makes every crop/grass report fully grown, so it can be harvested immediately after planting. The early return happens
    # before any stack push, so the frame is untouched. Called through virtual dispatch (no direct BL callers), so the patch takes effect wherever
    # the game checks growth through the entity.
    (0x278e2f0, 'fd7bbea9f30b00f9', '20008052c0035fd6', 'crops instantly grown: IsMaxGrowthLevel -> mov w0, #1; ret (FieldCropEntity)'),
    (0x27aa020, 'fd7bbea9f30b00f9', '20008052c0035fd6', 'grass instantly grown: IsMaxGrowthLevel -> mov w0, #1; ret (FieldGrassEntity)'),
    # 12. Windmill crafts instantly complete: WindmillCraftingManager.IsCompleteRecipe(uint craftItemId, WindmillCraftingType craftType) RVA 0x2F65630
    # scans the crafting list for the recipe. Replacing the function entry with mov w0, #1; ret makes every craft report complete, so finished goods
    # can be taken out immediately. Same early-return safety as above (before any stack push). Called through virtual dispatch (no direct BL callers).
    (0x2f65630, 'fd7bbca9f70b00f9', '20008052c0035fd6', 'windmill instantly complete: IsCompleteRecipe -> mov w0, #1; ret (WindmillCraftingManager)'),
    # 13. Fullscreen 1600x720 (no black bars, no stretch): the display-resolution selector at 0x15812F4 (Unity player, native) defaults
    # the handheld resolution to 1280x720 (mov w23, #0x500 at main+0x1581308) and only switches to 1920x1080 on the docked branches
    # (GetOperationMode == 1). The game never calls nn::oe::GetDefaultDisplayResolution (not in its imports), so the host cannot change
    # the resolution — the width must be patched here. Replacing the default width with 1600 (0x640) makes handheld render 1600x720
    # (20:9, the Redmi Turbo 3's 2712x1220 aspect), so the present fills the screen; docked stays 1920x1080. All handheld paths
    # (default, opmode==0, both-checks) read w23, so one cut covers them. Touch stays correct: the host still reports the 1280x720
    # panel and the player scales panel coords to the render size.
    (0x1581308, '17a08052', '17c88052', 'fullscreen 1600x720: handheld default width 1280 -> 1600 (display resolution selector)'),
    # NOT INCLUDED (no safe single-site patch found in this pass):
    # - Time freeze: the in-game clock does not advance through BokuMonoDateTime.Add* (all 0 direct callers) and get_TimeScale has 0 callers
    #   (the clock reads the TimeScale field directly or the advance is inlined); the per-frame advance site was not isolated. Needs deeper tracing.
    # - Cheer: no depletable "cheer" resource exists in the game (only cheer animations and the computed bazaar CheerBonus in the sales dialog).
  ],
}
