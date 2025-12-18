ROOT="${NATSL_ROOT:-/mnt/home/awinje/NAtSL}"
BENCHMARK=ruler

MODEL=$1
LENGTH=$2

BASENAME=$(basename $MODEL)

echo Getting evaluations for $BASENAME on $BENCHMARK-$LENGTH

python "$ROOT/eval/ruler/RULER/scripts/eval/evaluate.py" \
    --data_dir="$ROOTeval/ruler/predictions/$BASENAME/$BENCHMARK-$LENGTH" \
    --benchmark="synthetic" \

