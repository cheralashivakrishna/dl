# KMNIST Classification using Neural Networks

## Overview

This project implements a neural network to classify images from the Kuzushiji-MNIST (KMNIST) dataset. The model is trained using different configurations of hidden layers, activation functions, optimizers, and weight decay values to determine the best performing architecture.

## Dataset

The KMNIST dataset is a drop-in replacement for the MNIST dataset, containing 70,000 grayscale images of handwritten Japanese characters:

- **Training set**: 60,000 images
- **Test set**: 10,000 images

The dataset is loaded using `tensorflow_datasets` and preprocessed by normalizing the images and applying one-hot encoding to the labels.

## Model Architecture

The neural network is implemented using `TensorFlow` and `Keras`. The architecture consists of:

- Input layer: 784 neurons (28x28 flattened image)
- Hidden layers: Configurable number of layers with varying units and activation functions
- Output layer: 10 neurons (softmax activation for classification)

## Training Configurations

The model was trained with multiple configurations:

| Hidden Layers  | Activation | Optimizer | Weight Decay | Best Validation Accuracy | Test Accuracy |
| -------------- | ---------- | --------- | ------------ | ------------------------ | ------------- |
| [128, 64]      | ReLU       | Adam      | 0.0005       | 0.9495                   | 0.8834        |
| [128, 128, 64] | ReLU       | SGD       | 0.0000       | 0.8363                   | 0.7053        |
| [256, 128, 64] | Sigmoid    | RMSprop   | 0.0005       | 0.9157                   | 0.8323        |
| [128, 64, 32]  | ReLU       | Nesterov  | 0.0005       | 0.9308                   | 0.8603        |

The best performing model:

- **Hidden Layers**: `[128, 64]`
- **Activation**: `ReLU`
- **Optimizer**: `Adam`
- **Weight Decay**: `0.0005`
- **Validation Accuracy**: `94.95%`
- **Test Accuracy**: `88.34%`

## Results

The best model achieved a test accuracy of **88.34%**. A confusion matrix was generated to analyze model performance across different classes.

## Usage

To run the model, ensure you have the required dependencies installed:

```bash
pip install tensorflow tensorflow-datasets numpy matplotlib seaborn pandas scikit-learn
```

Then, execute the script:

```bash
python train_kmnist.py
```

## Author

**Cherala Shiva Krishna **

