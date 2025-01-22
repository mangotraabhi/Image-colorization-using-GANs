# Image Colorization using GANs

This project involves the colorization of grayscale images using Generative Adversarial Networks (GANs). The implemented model consists of a Generator and a Discriminator that work together to produce high-quality colorized images.

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Training Procedure](#training-procedure)
5. [Results](#results)
6. [Dependencies](#dependencies)
7. [Usage](#usage)
8. [References](#references)

## Introduction

Image colorization is a process of converting grayscale images into colored ones. GANs are particularly effective for this task because they can learn complex mappings from grayscale to color while preserving fine image details. This project uses a U-Net-based generator and a PatchGAN discriminator for image colorization.

## Dataset

The dataset used for this project consists of pairs of grayscale and color images. The images were preprocessed as follows:

- **Size:** All images were resized to 256x256 pixels.
- **Normalization:** Pixel values were normalized to the range [0, 1].

### Directory Structure

- `/content/landscape Images/color`: Contains color images.
- `/content/landscape Images/gray`: Contains grayscale images.

## Model Architecture

### Generator

The generator is based on the U-Net architecture with the following components:

- **Downsampling:** Sequential convolutional layers with LeakyReLU activation and optional batch normalization.
- **Upsampling:** Transposed convolutional layers with ReLU activation, batch normalization, and optional dropout.
- **Skip Connections:** Connections between corresponding downsampling and upsampling layers to preserve spatial information.

### Discriminator

The discriminator is a PatchGAN that classifies whether each patch in the input image pair is real or fake. It consists of multiple downsampling layers and convolutional layers with LeakyReLU activation.

## Training Procedure

### Loss Functions

- **Generator Loss:** Combines Binary Crossentropy loss and L1 loss to encourage similarity between generated and target images.
- **Discriminator Loss:** Binary Crossentropy loss to distinguish real and fake images.

### Optimizers

Adam optimizer with a learning rate of 2e-4 and beta_1 of 0.5 was used for both the generator and discriminator.

### Hyperparameters

- **Image Size:** 256x256
- **Batch Size:** 64
- **Epochs:** 50
- **Lambda (L1 loss weight):** 100

### Training Process

1. Load and preprocess the dataset.
2. Train the generator and discriminator in an adversarial manner.
3. Use the generator to produce colorized images and evaluate results visually.

## Results

### Loss Graphs

- Generator and discriminator loss values were plotted over iterations to monitor training progress.

### Image Samples

- Examples of colorized images are displayed, including the input grayscale image, ground truth, and the generated image.

## Dependencies

This project requires the following libraries:

- Python
- TensorFlow
- NumPy
- OpenCV
- Matplotlib
- Seaborn
- tqdm

## Usage

1. Clone the repository and navigate to the project directory.
2. Ensure the dataset is placed in the appropriate directory structure.
3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the training script:

   ```bash
   python train.py
   ```

5. Evaluate the model and visualize results:

   ```bash
   python evaluate.py
   ```

## References

1. [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
2. [Pix2Pix GAN](https://arxiv.org/abs/1611.07004)
