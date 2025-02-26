# Hyperspectral Image Classification using 3D CNN

This repository contains an implementation of a 3D Convolutional Neural Network (3D CNN) for hyperspectral image classification. The model processes hyperspectral data cubes, extracting spatial-spectral features for accurate classification.

## Features
- Supports multiple hyperspectral datasets (e.g., Pavia University, Salinas, Indian Pines, etc.)
- Uses a **3D CNN model** with cross-channel attention layers
- Implements **PCA-based dimensionality reduction**
- Custom data loader and preprocessing pipeline
- Training and evaluation scripts with performance metrics
- Generates classification maps for visualization

## Project Structure
```
├── config.py           # Configuration settings
├── data_loader.py      # Data loading and preprocessing
├── generate_results.py # Generates classification maps
├── main.py             # Main script to train and evaluate the model
├── metrics.py          # Computes accuracy, confusion matrix, etc.
├── model.py            # 3D CNN model architecture
├── train.py            # Training pipeline
├── utils.py            # Utility functions
├── results/            # Output directory for results
└── README.md           # Project documentation
```

## Installation
To run this project, install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
### 1. Configure Parameters
Modify `config.py` to set dataset paths, training parameters, and model settings.

### 2. Train the Model
Run the following command to train the model:
```bash
python main.py
```

### 3. Generate Classification Maps
After training, generate classification results using:
```bash
python generate_results.py
```

## Datasets
The model supports multiple hyperspectral datasets, including:
- **Pavia University (PU)**
- **Indian Pines (IP)**
- **Salinas (SV)**
- **Botswana (BO)**

Ensure dataset `.mat` files are placed in the `attached_assets` directory.

## Results
Results, including classification maps and accuracy metrics, are saved in the `results/` directory.

## Citation
If you use this work in your research, please cite appropriately.

## License
This project is licensed under the MIT License.

