# Landscape Image Colorization

A deep learning project built with Keras and TensorFlow that automatically colorizes grayscale landscape images using a U-Net style Convolutional Neural Network (CNN) architecture.

## Overview

This repository contains the training pipeline and model loading utilities for an image-to-image translation model. The network takes single-channel grayscale images resized to 128x128 pixels and predicts their corresponding three-channel RGB color representations.

## Project Structure

- `main.py`: Handles data loading, preprocessing, model architecture definition, training, and validation.
- `load_model.py`: Demonstrates how to load the saved Keras model (`color_model.keras`) for inference.
- `requirments.txt`: Lists the Python package dependencies required to run the project.
- `landscape Images/`: Expected root directory for the dataset, containing `gray/` and `color/` subdirectories.

## Prerequisites and Installation

Ensure you have Python 3.9+ installed. Install the required dependencies using pip:

```bash
pip install -r requirments.txt
```

### Dependencies

- TensorFlow (`2.21.0`)
- NumPy (`2.5.3`)
- scikit-learn (`1.9.1`)
- Pillow (`12.3.0`)

## Dataset Preparation

The training script expects the dataset to be organized in the following directory structure relative to `main.py`:

```text
landscape Images/
├── gray/
│   ├── image_001.jpg
│   └── ...
└── color/
    ├── image_001.jpg
    └── ...
```

Filenames in both `gray/` and `color/` must match alphabetically so that input and target pairs align properly during sorting.

## Model Architecture

The model uses an encoder-decoder architecture with skip connections (U-Net variant):

- **Input**: 128x128x1 grayscale image normalized to `[0, 1]`.
- **Encoder**: Four convolutional blocks with increasing filter sizes (32, 64, 128, 256). Each block consists of a 3x3 `Conv2D` layer with ReLU activation, `BatchNormalization`, `SpatialDropout2D` (rate 0.2), and `MaxPool2D` (except the bottleneck block).
- **Decoder**: Three upsampling blocks using `Conv2DTranspose` layers (128, 64, 32 filters) concatenated with corresponding feature maps from the encoder via skip connections.
- **Output**: A 1x1 `Conv2D` layer producing a 128x128x3 RGB output image.

## Training

To train the model from scratch, run:

```bash
python main.py
```

### Training Configuration

- **Train/Test Split**: 80% training, 20% testing (`random_state=1`), with 10% of the training set reserved for validation.
- **Optimizer**: Adam
- **Loss Function**: Huber Loss
- **Batch Size**: 64
- **Epochs**: Up to 350
- **Callbacks**:
  - `EarlyStopping`: Monitors `val_loss` with a patience of 10 epochs and restores the best weights.
  - `ReduceLROnPlateau`: Reduces the learning rate by a factor of 0.5 if `val_loss` plateaus for 5 epochs.

*Note: Uncomment the `model.save('color_model.keras')` line at the bottom of `main.py` (and add the closing quote) to persist the trained weights to disk.*

## Loading the Trained Model

Once the model is saved as `color_model.keras`, you can load it into your inference pipeline as shown in `load_model.py`:

```python
import keras

model = keras.saving.load_model('color_model.keras')
```