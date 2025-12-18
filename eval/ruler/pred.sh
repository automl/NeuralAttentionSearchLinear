TFPP_SMALL=~/NAtSL/models/tfpp_small
GATED_DELTANET_SMALL=~/NAtSL/models/gated_deltanet_small
MAMBA2_SMALL=~/NAtSL/models/mamba2_small
HYBRID_DELTANET_SMALL=~/NAtSL/models/gated_deltanet_hybrid_small

MODEL=$1
LENGTH=$2
BENCHMARK=ruler
BATCH_SIZE=$((256/($LENGTH/1024)))  # ca. 40-80 GB
BASENAME=$(basename $MODEL)

ROOT="${NATSL_ROOT:-/mnt/home/awinje/NAtSL}"
SAVE_DIR=$ROOT/eval/ruler/predictions/$BASENAME/$BENCHMARK-$LENGTH
DATA_DIR=$ROOT/eval/ruler/datasets/$BENCHMARK/transformer-1.3B-100B-$LENGTH

echo Getting predictions for $BASENAME-$LENGTH on $BENCHMARK-$LENGTH

# Skip if predictions exist (e.g. when iterating over all models/lengths again, we don't overwrite - saves time)
if [ -d "$SAVE_DIR" ]; then
  echo "Prediction already exists at: $SAVE_DIR"
  exit 0
fi

python $ROOT/eval/ruler/RULER/scripts/pred/call_api.py --server_type="hf" \
    --data_dir="$DATA_DIR" \
    --save_dir="$SAVE_DIR" \
    --benchmark="synthetic" \
    --task="niah_single_1" \
    --batch_size=$BATCH_SIZE \
    --model_name_or_path=$MODEL

# Delete predictions on error (just so we don't have empty prediction folders etc.)
if [ ! $? -eq 0 ]; then
	echo "Cleaning up..."
	rm -r $SAVE_DIR
fi

