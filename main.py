import torch
from config import Config
from data_loader import prepare_data
from model import PResNet
from train import train_model

def main():
    # Initialize configuration
    config = Config()
    print(f"\n{'='*50}")
    print(f"Starting training process for {config.DATASET} dataset")
    print(f"{'='*50}\n")
    print(f"Device being used: {config.DEVICE}")

    # Prepare data
    print(f"\nPreparing {config.DATASET} dataset...")
    train_loader, test_loader, val_loader, num_classes, n_bands = prepare_data(config)
    print(f"Dataset prepared successfully:")
    print(f"Number of classes: {num_classes}")
    print(f"Number of bands: {n_bands}")

    # Calculate avgpoosize based on spatial size
    avgpoosize = {
        5: 1, 7: 1,
        9: 2, 11: 2,
        13: 3, 15: 3,
        17: 4, 19: 4
    }.get(config.SPATIAL_SIZE, 3)  # default to 3 if size not in mapping
    print(f"Using avgpoosize: {avgpoosize} for spatial size: {config.SPATIAL_SIZE}")

    # Initialize model
    print("\nInitializing model...")
    model = PResNet(
        depth=config.DEPTH,
        alpha=config.ALPHA,
        num_classes=num_classes,
        n_bands=n_bands,
        avgpoosize=avgpoosize,
        inplanes=config.INPLANES,
        bottleneck=config.BOTTLENECK
    ).to(config.DEVICE)
    print("Model initialized successfully")


    # Print model configuration
    print("\nModel Configuration:")
    print(f"Depth: {config.DEPTH}")
    print(f"Alpha: {config.ALPHA}")
    print(f"Inplanes: {config.INPLANES}")
    print(f"Bottleneck: {config.BOTTLENECK}")

    # Train and evaluate model
    print("\nStarting training...")
    classification, confusion, results = train_model(
        model, train_loader, test_loader, val_loader, config
    )

    # Print results
    print("\nTraining completed!")
    print("\nClassification Report:")
    print(classification)
    print("\nConfusion Matrix:")
    print(confusion)
    print("\nResults [OA, AA, Kappa, Class-wise accuracies]:")
    print(results)

if __name__ == '__main__':
    main()
