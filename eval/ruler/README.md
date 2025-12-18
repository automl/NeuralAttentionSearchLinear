# RULER benchmark (w/ HF)

## How to use

### Clone RULER and install NEMO

RULER (install here; in eval/ruler/:
```
git clone git@github.com:NVIDIA/RULER.git
```

NEMO:
```
pip install nemo-toolkit[all]==2.5.3
```

Export `$NATSL_ROOT` (root to this repo):
```
export NATSL_ROOT="/mnt/home/$USER/project/NAtSL"
```

### Prepare RULER dataset with fla tokenizer

Run prepare.sh, e.g.:
```
for length in 1024 2048 4096 8192 16384 32768 65536 131072; do prepare.sh $length; done
```
(prepares data for all lengths 1K-131K)

### Generate predictions

Run pred.sh for each model and length:
```
MODEL=/path/to/hf-weights
LENGTH=1024  # e.g.
pred.sh $MODEL $LENGTH
```

### Get evaluations

Run eval.sh:
```
eval.sh $MODEL $LENGTH  # do for each model/length
```

### Plotting

Just run plotting.py
```
python plotting.py
```
Reads all evaluated models/lengths, plots them in image plot.png.

