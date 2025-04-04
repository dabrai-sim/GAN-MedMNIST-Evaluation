# Evaluating GANs on MedMNIST with TensorBoard Visualization

This repository provides a complete framework for evaluating a Generative Adversarial Network (GAN) trained on the MedMNIST dataset. The project uses two popular evaluation metrics—**Inception Score (IS)** and **Fréchet Inception Distance (FID)**—and leverages **TensorBoard** to visualize generated images and metric trends.

## Overview

- **Project Objective:**  
  Evaluate the performance of a GAN using quantitative metrics (IS and FID) and visualize results using TensorBoard.
  
- **Dataset:**  
  MedMNIST – a collection of curated medical images for benchmarking machine learning models in healthcare.

- **Key Features:**
  - Generate synthetic MedMNIST images with a GAN.
  - Preprocess images for metric computation.
  - Log images and evaluation metrics using TensorBoard.
  - Compute and display IS and FID.
    
## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/medmnist-gan-evaluation.git
   cd medmnist-gan-evaluation
   ```

2. **Install dependencies using Conda:**

   ```bash
   conda create -n medmnist-gan python=3.9
   conda activate medmnist-gan
   conda install -c conda-forge tensorflow torch torchvision torchmetrics tensorboard
   ```

   Alternatively, use `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Evaluate the GAN:**

   In your Python script or interactive session, import and run the evaluation function:

   ```python
   from evaluate_medmnist_gan import evaluate_medmnist_gan
   # Assume 'generator' is your trained GAN generator model
   is_score, fid_score = evaluate_medmnist_gan(generator, num_images=100, latent_dim=100, device='cuda')
   ```

2. **Launch TensorBoard:**

   After running the evaluation, start TensorBoard to view the generated images and metrics:

   ```bash
   tensorboard --logdir=logs_medmnist
   ```
   Then open your browser and go to [http://localhost:6006](http://localhost:6006).


### Results & Insights
- **Inception Score:** Provides a quantitative measure of image quality.
- **FID Score:** Reflects the similarity between synthetic and real image distributions.
  
Visualizing these metrics on TensorBoard allowed for a detailed analysis of model performance and guided further improvements.

## Conclusion
This project demonstrates a robust framework for evaluating GANs on MedMNIST, combining quantitative metrics with qualitative visualizations. The integration of TensorBoard facilitates efficient monitoring and iterative model refinement, which is essential for advancing generative modeling in medical imaging.

 
