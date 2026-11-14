# Defaults chosen while building (flip any; each is one config or one module)

1. **Eyes:** two Eyes with different viewpoints in the main system (occlusion makes per-Eye credit
   non-trivial); one Eye is an ablation. `default_eyes()` in `datagen.py`.
2. **Eye output:** SCENE JSON (class, coarse region, size, mass/fragility guess, occluded, confidence).
   Metric poses come from a simulator depth/segmentation tool, not from the VLM.
3. **Arm head:** per-arm flow-matching head; MLP head is the ablation (`ArmPolicy(head=...)`).
   A joint two-arm head for handover is not implemented.
4. **Latent messages:** out of the main system; only the typed text layer exists.
5. **Task family:** one family (backpack packing) with amendments and injected faults for Track B;
   Track A uses VIKI-Bench unchanged.
6. **Base models:** Qwen3.5-4B for Eye and Brain as in the plan; support in your TRL/vLLM versions is
   unverified. A Qwen2.5-VL-based control is required for Track A.
7. **Heavy-bottom rule** ignores the explicitly "last" item (otherwise "heavy at bottom" and "lunchbox
   last" contradict each other for light lunchboxes).
8. **Failure attribution** (Eye/Brain/Arm) is rule-based and heuristic (`loop.attribute`); report it as
   such, and validate on a hand-labelled subset before using it as a result.
