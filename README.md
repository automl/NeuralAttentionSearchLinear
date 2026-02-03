# Neural Attention Search Linear


This repository introduces a framework that searches for the optimal token-level hybrid attention models.
Currently, we allow searching for the optimal Gated DeltaNet-Softmax Transformer models.



usage: 
create a new conda environment with: 
```angular2html
conda create -n NAtSL python=3.11
conda activate NAtSL
pip install -e .
```
We use the [flame](https://github.com/fla-org/flame.git) package to train the model:
```
cd experiments
git clone https://github.com/fla-org/flame.git && cd flame
pip install -e flame/
cd ..
```

Then train the model:
```
cd experiments
sbatch slurm_train_model.sh
```

We could then evaluate the pre-trained model with `experiments/harness.py`
