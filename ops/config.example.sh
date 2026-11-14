# Copy to ops/config.sh and edit after running scripts/s0_inventory.py (step "inventory").
export L1_ID=""      # path to the L1 split you treat as in-domain (.parquet)
export L1_OOD=""
export L2_ID=""
export L2_OOD=""
export VIKI_CKPT_L2=""      # local path of the released L2 checkpoint (e.g. $MODELS/VIKI-R-7B-L2)
export VIKI_CKPT_L1=""
export BASE_MODEL="$MODELS/Qwen3.5-4B"          # Eye and Brain base (switch to the Qwen2.5-VL fallback if the gate fails)
export FALLBACK_MODEL="$MODELS/Qwen2.5-VL-3B-Instruct"
export GPU_GB=48                                # your VRAM; used for vLLM utilization defaults
