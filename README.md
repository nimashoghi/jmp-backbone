# JMP - From Molecules to Materials: Pre-training Large Generalizable Models for Atomic Property Prediction

This is the refined code implementation of the ICLR 2024 poster paper [From Molecules to Materials: Pre-training Large Generalizable Models for Atomic Property Prediction](https://openreview.net/forum?id=PfPnugdxup)

## Installation

```bash
conda create -n jmp-peft python=3.11 -y
conda activate jmp-peft

# Install PyTorch
conda install -y -c conda-forge -c pytorch -c nvidia -c pyg \
    "pytorch=2.2.*" torchvision torchaudio pytorch-cuda=12.1 \
    pyg pyg pytorch-scatter pytorch-sparse pytorch-cluster \
    "numpy<2" matplotlib seaborn sympy pandas numba scikit-learn plotly nbformat ipykernel ipywidgets tqdm pyyaml networkx \
    pytorch-lightning torchmetrics lightning \
    einops wandb \
    cloudpickle pydantic \
    frozendict wrapt varname typing-extensions lovely-tensors lovely-numpy requests pytest nbval

# Rich for better terminal output
pip install rich lmdb ase pymatgen matbench-discovery beartype jaxtyping e3nn
```

And then clone this repo and run:

```bash
pip install -e .
```