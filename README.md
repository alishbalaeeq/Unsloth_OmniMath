# Fine-Tuning Phi-4 on a Mathematical Dataset with Unsloth

This project fine-tunes the `unsloth/phi-4` model on the Omni-MATH dataset to enhance its mathematical problem-solving capabilities. It leverages the Unsloth library to achieve significantly faster training and reduced memory usage, making it possible to fine-tune powerful models on consumer-grade hardware.

## Overview

The core of this project is to take a powerful, pre-trained language model and specialize it for a specific domain—in this case, mathematics. By using 4-bit quantization and LoRA (Low-Rank Adaptation) through Unsloth, we can efficiently adapt the model without the need for extensive computational resources.

The project is structured into modular Python scripts for clarity and maintainability:
- **`data.py`**: Handles loading and preprocessing of the dataset.
- **`model.py`**: Manages the loading of the base model and tokenizer, and applies the PEFT (Parameter-Efficient Fine-Tuning) configuration.
- **`trainer.py`**: Configures the training arguments and sets up the `SFTTrainer`.
- **`training.py`**: The main script that orchestrates the entire fine-tuning process.

## Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.8+
- An NVIDIA GPU with CUDA installed
- Conda for environment management

## Getting Started

### 1. Clone the Repository
```bash
git clone <your-repo-link>
cd <your-repo-directory>
```

### 2. Create and Activate Conda Environment
It is highly recommended to use a dedicated Conda environment to manage dependencies.
```bash
conda create --name unsloth_env python=3.11
conda activate unsloth_env
```

### 3. Install Dependencies
This project uses a specific set of libraries for compatibility and performance. The following command, powered by Unsloth, will install the correct versions of PyTorch, Transformers, and other necessary packages for a CUDA 11.8 environment.
```bash
pip install "unsloth[cu118-torch2.3]=v2024.6"
```
This single command handles the installation of:
- `unsloth`
- `torch` (GPU version)
- `transformers`
- `datasets`
- `trl`
- `xformers` (for optimized attention)
- `bitsandbytes` (for quantization)

## Usage

To start the fine-tuning process, simply run the main training script:
```bash
python Llamacpp/code/training.py
```
The script will handle everything from loading the data and model to running the training loop and saving the output to the `outputs/` directory.

## Configuration Details

The fine-tuning process is configured across the different modules:

### Model Configuration (`model.py`)
- **Base Model**: `unsloth/phi-4`
- **Quantization**: 4-bit (`load_in_4bit = True`)
- **Max Sequence Length**: `2048`
- **LoRA (Low-Rank Adaptation)**:
    - **Rank (`r`)**: `16`
    - **`lora_alpha`**: `16`
    - **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

### Dataset (`data.py`)
- The training uses the `omni_math_phi.json` dataset, which is expected to be in the `dataset/` directory.
- The data is formatted using Unsloth's `standardize_sharegpt` function, which prepares it for instruction fine-tuning.

### Training Hyperparameters (`trainer.py`)
- **Trainer**: `trl.SFTTrainer`
- **Batch Size**: `2` (per device)
- **Gradient Accumulation**: `4` steps
- **Learning Rate**: `2e-4`
- **Optimizer**: `adamw_8bit`
- **LR Scheduler**: `linear`
- **Precision**: `fp16` or `bf16` (auto-detected)

## Project Structure
```
.
├── Llamacpp/
│   ├── code/
│   │   ├── data.py
│   │   ├── model.py
│   │   ├── trainer.py
│   │   └── training.py
│   ├── dataset/
│   │   └── omni_math_phi.json
│   ├── outputs/
│   └── README.md
└── ...
```

## Dataset

The training uses the `omni_math_phi.json` dataset, which contains mathematical problems and solutions in a conversation format. Each entry includes:
- A mathematical problem posed by the user
- A detailed solution provided by the assistant

Example from the dataset:
