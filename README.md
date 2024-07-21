# Setup Instructions

This README provides detailed setup instructions for setting up the environment necessary for running our experiments.

## Requirements

Ensure the following requirements are met:

- Python version: 3.6.5
- CUDA Toolkit 11.1
- cuDNN v8.2.1.32
- PyTorch 1.8.1

To install PyTorch with GPU support, run the following command:

```python
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```
# Project Training and Testing Guide

This guide explains how to use the scripts for training and testing within the project.

## Data Augmentation

To enhance the training process, you can perform data augmentation using the following command:

```python
python data_augmentation.py --input_prefix v1/Urdu/Urdu --output_prefix v1/Urdu/Urdu --factor 6
```

## Training

To train the model, you can use the following command:

```python
python main.py --lr 0.001 --train_lang Urdu --batch_size 64 --epochs 52 --fusion gated --pretrained_fop_path /path/to/Urdu_checkpoint.pth.tar
```
## Score Polarization Strategy

For the obtained L2 scores, refer to the following for Score Polarization:

```python
python score_polarization.py path/to/face_images path/to/audio_files score_folder1 score_folder2
```
