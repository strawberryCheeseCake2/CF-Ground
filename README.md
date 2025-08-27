# Run Grounding

## GUI Actor

## :rescue_worker_helmet: Installation

1. Create a conda environment: 
```bash
conda create -n gui_actor python=3.10
conda activate gui_actor
```

2. Install the dependencies: 
- NVIDIA GPU
```bash
pip install -r requirements_gui_agent.txt
```
- MAC OS
```bash
pip install -r requirements_gui_agent_mac.txt
```

## zonui

1. Create a conda environment: 
```bash
conda create -n zonui python=3.10
conda activate zonui
```

2. Install the dependencies:
- NVIDIA GPU
```bash
pip install -r requirements_zonui.txt
```


## :minidisc: Data Preparation
1. Download the processed data from [here](https://huggingface.co/datasets/cckevinn/GUI-Actor-Data).
2. Modify the paths in the [data_config.yaml](./data/data_config.yaml) file to point to the downloaded data.
