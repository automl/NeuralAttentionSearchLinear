LENGTH=$1
ROOT="${NATSL_ROOT:-/mnt/home/awinje/NAtSL}"
TOKENIZER="fla-hub/transformer-1.3B-100B"
SAVE_DIR="$ROOT/eval/ruler/datasets/ruler/$(basename $TOKENIZER)-$LENGTH"
echo Preparing $(basename $SAVE_DIR)
uv run python scripts/data/prepare.py --save_dir=$SAVE_DIR \
    --benchmark synthetic \
    --task niah_single_1 \
    --tokenizer_path $TOKENIZER \
    --tokenizer_type hf \
    --max_seq_length $LENGTH \
    --model_template_type base \
    --num_samples 500

