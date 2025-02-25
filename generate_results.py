import os
import numpy as np
import torch
import spectral
import scipy.io as sio
from model import PResNet
from config import Config
from data_loader import loadData, padWithZeros
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
import torch.serialization

def generate_classification_map(model, data, labels, config):
    """Generate classification map for the entire image"""
    height, width = labels.shape
    PATCH_SIZE = config.SPATIAL_SIZE
    outputs = np.zeros((height, width))

    # Pad the image
    padded_data = padWithZeros(data, PATCH_SIZE // 2)

    # Create classification map
    print("Generating classification map...")
    for i in range(height):
        if i % 20 == 0:
            print(f'Processing row {i}/{height}')
        for j in range(width):
            if int(labels[i, j]) == 0:
                continue

            # Extract and process patch
            patch = padded_data[i:i + PATCH_SIZE, j:j + PATCH_SIZE, :]
            patch = patch.reshape(1, patch.shape[0], patch.shape[1], patch.shape[2])
            patch = torch.FloatTensor(patch.transpose(0, 3, 1, 2)).to(config.DEVICE)

            # Get prediction
            with torch.no_grad():
                prediction = model(patch)
                prediction = np.argmax(prediction.cpu().numpy(), axis=1)
                outputs[i][j] = prediction + 1

    return outputs

def save_results(predictions, ground_truth, dataset_name, save_dir='results'):
    """Save classification results and metrics"""
    os.makedirs(save_dir, exist_ok=True)

    # Save classification map
    plt.figure(figsize=(12, 4))

    # Ground Truth
    plt.subplot(131)
    plt.imshow(ground_truth)
    plt.title('Ground Truth')
    plt.axis('off')

    # Prediction
    plt.subplot(132)
    plt.imshow(predictions)
    plt.title('Classification Map')
    plt.axis('off')

    # Difference Map
    plt.subplot(133)
    diff_map = (predictions != ground_truth).astype(int)
    plt.imshow(diff_map, cmap='RdYlGn_r')
    plt.title('Difference Map\n(Red: Misclassified)')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_classification_results.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Calculate and save metrics
    mask = ground_truth != 0
    y_true = ground_truth[mask]
    y_pred = predictions[mask]

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)

    with open(os.path.join(save_dir, f'{dataset_name}_metrics.txt'), 'w') as f:
        f.write(f"Overall Accuracy: {accuracy*100:.2f}%\n\n")
        f.write("Detailed Classification Report:\n")
        f.write(report)

def main():
    # Initialize configuration
    config = Config()
    print(f"\nGenerating results for {config.DATASET} dataset")

    # Load data
    data, labels, _ = loadData(config.DATASET, config.NUM_COMPONENTS, config.DATA_PATH)

    # Load trained model
    print("\nLoading trained model...")
    model = PResNet(
        depth=config.DEPTH,
        alpha=config.ALPHA,
        num_classes=len(np.unique(labels)) - 1,
        n_bands=config.NUM_COMPONENTS,
        avgpoosize=3,  # Default for spatial size 15
        inplanes=config.INPLANES,
        bottleneck=config.BOTTLENECK
    ).to(config.DEVICE)

    try:
        # Load the checkpoint
        checkpoint = torch.load("best_model.pth.tar", map_location=config.DEVICE, weights_only=False)

        # Extract state dict from the nested structure
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Load the state dict
        model.load_state_dict(state_dict)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return

    model.eval()

    # Generate classification map
    print("\nGenerating classification map...")
    predictions = generate_classification_map(model, data, labels, config)

    # Save results
    print("\nSaving results...")
    save_results(predictions, labels, config.DATASET)
    print("\nResults saved in 'results' directory")

if __name__ == '__main__':
    main()