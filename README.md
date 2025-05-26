# FGG-music-code

[Demo](https://huajianduzhuo-code.github.io/FGG-diffusion-music/) | [Paper](https://arxiv.org/abs/2410.08435)

This is the code repository of the paper:

> Tingyu Zhu, Haoyu Liu, Ziyu Wang, Zhimin Jiang, and Zeyu Zheng. "Efficient Fine-Grained Guidance for Diffusion Model Based Symbolic Music Generation." ICML 2025.


## Description

A deep learning model for generating musical accompaniments and melodies based on chord progressions. This project implements a diffusion model that can generate either combined melody and accompaniment or separate accompaniment based on chord conditions.

## Features

- Two training (and generation) modes:
  - **Combined Mode**: Generates both melody and accompaniment together, conditioned on chord progressions
  - **Separate Mode**: Generates accompaniment only, conditioned on both chord progressions and melody
- Data augmentation through pitch shifting
- Support for both training from scratch and resuming from checkpoints
- Configurable training parameters

## Prerequisites

- Python 3.x
- Required packages (see `requirements.txt`):
  - numpy
  - tqdm
  - torch
  - Other dependencies listed in requirements.txt

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
### 0. Downloading Data and Pre-trained Checkpoints

The data and pre-trained checkpoints can be downloaded and added to the repository. These are located in the following directories:

- `data/pop909_data`: Contains training data (We use the same dataset files as [whole-song-gen](https://github.com/ZZWaang/whole-song-gen))
- `results/`: Contains the model checkpoints

Download them using the corresponding links provided in the `download_link.txt` files located in each directory.

For example, you can find the download link for the POP909 dataset in `data/pop909_data/download_link.txt`.


### 1. Data Generation



First, generate the training and test datasets using `train_test_data_generation.py`. You can choose between two data formats:

```bash
# For separate melody and accompaniment (default)
python data/train_test_data_generation.py

# For combined melody and accompaniment
python data/train_test_data_generation.py --combine_melody_acc
```

The script will generate:
- Training data slices
- Test data slices
- Data will be saved in `data/train_test_slices/` directory

### 2. Model Training

Train the model using `train.py`. The script supports various training options:

```bash
# Basic training with default settings
python train.py --output_dir results

# Training with specific options
python train.py \
    --output_dir results \
    --data_format separate_melody_accompaniment \
    --uniform_pitch_shift \
    --null_cond_weight 0.5 \
    --load_chkpt_from [checkpoint_path]
```

#### Training Options

- `--output_dir`: Directory to store model checkpoints and logs (default: 'results')
- `--data_format`: Choose between 'separate_melody_accompaniment' or 'combine_melody_accompaniment'
- `--uniform_pitch_shift`: Apply pitch shift uniformly (default: random)
- `--debug`: Enable debug mode
- `--load_chkpt_from`: Path to load existing checkpoint for resuming training
- `--null_cond_weight`: Weight parameter for null condition in classifier-free guidance (default: 0.5)

### 3. Model Generation

After training the model, you can use it to generate new musical pieces. The repository includes two Jupyter notebooks for generation:

1. `generation.ipynb`: Basic generation notebook that demonstrates how to:
   - Load a trained model
   - Generate new musical pieces using chord conditions
   - Save the generated results into .mid and .wav files

2. `generation_style.ipynb`: Advanced generation notebook that includes:
   - Style-based generation capabilities
   - Additional control over the generation process

To use the generation notebooks:

1. Make sure you have Jupyter installed:
```bash
pip install jupyter
```

2. Start Jupyter:
```bash
jupyter notebook
```

1. Open either `generation.ipynb` or `generation_style.ipynb` in your browser

4. Follow the notebook instructions to:
   - Load your trained model
   - Set generation parameters
   - Generate and save new musical pieces

Note: Make sure to use the appropriate model path based on your training mode (combined or separate melody/accompaniment).

## Project Structure

```
FGG-music-code/
├── data/                           # Data processing and dataset management
│   ├── train_test_data_generation.py  # Script for generating training/test data
│   ├── dataset_loading.py            # Dataset loading utilities
│   ├── train_test_slices/           # Generated training and test data
│   ├── pop909_data/                 # POP909 dataset files
│   ├── prepare_training_pianoroll/  # Piano roll preparation utilities
│   └── song_analysis_utils/         # Music analysis utilities
│
├── model/                          # Model architecture and components
│   ├── architecture/               # Neural network architectures
│   ├── latent_diffusion.py        # Latent diffusion model implementation
│   ├── model_sdf.py               # Score-based diffusion model
│   └── sampler_sdf.py             # Sampling utilities
│
├── train/                         # Training related code
│   └── train_params.py           # Training parameters and configurations
│
├── generation_utils/              # Utilities for music generation
│
├── generated_samples/            # Directory for storing generated music
│
├── results/                      # Model checkpoints and training logs
│
├── train.py                      # Main training script
├── generation.ipynb              # Basic generation notebook
├── generation_style.ipynb        # Advanced style-based generation notebook
└── requirements.txt              # Project dependencies
```