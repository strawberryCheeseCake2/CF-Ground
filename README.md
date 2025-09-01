# Run Grounding

## GUI Actor

## :rescue_worker_helmet: Installation

1. Clone this repo to your local machine: 
```bash
git clone git@github.com:strawberryCheeseCake2/CF-Ground.git
cd CF-Ground
```

2. Create a conda environment: 
```bash
conda create -n gui_actor python=3.10
conda activate gui_actor
```

3. Install the dependencies: 
- NVIDIA GPU
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_gui_agent.txt
```
- MAC OS
```bash
pip install -r requirements_gui_agent_mac.txt
```

## :minidisc: Data Preparation
1. Download the processed data from [here](https://huggingface.co/datasets/OS-Copilot/ScreenSpot-v2/tree/main).
2. Move files at ./data folder
