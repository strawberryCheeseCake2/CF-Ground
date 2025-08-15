# Conda environment

- Little package
```
conda env create -p ./conda_env -f environment.yml
conda activate ./conda_env
```

- total package
```
conda env create -p ./stage_env2 -f environment.full.yml
conda activate ./stage_env2
```