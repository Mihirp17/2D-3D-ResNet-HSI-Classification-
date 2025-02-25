import os
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

class HyperData(Dataset):
    def __init__(self, dataset):
        try:
            self.data = dataset[0].astype(np.float32)
            self.labels = [int(n) for n in dataset[1]]
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except Exception as e:
            print(f"Error initializing dataset: {str(e)}")
            raise


    def __getitem__(self, index):
        try:
            img = torch.from_numpy(np.asarray(self.data[index,:,:,:])).to(self.device)
            label = self.labels[index]
            return img, label

        except Exception as e:
            print(f"Error getting item at index {index}: {str(e)}")
            raise

    def __len__(self):
        return len(self.labels)

    def __labels__(self):
        return self.labels

def loadData(name, num_components=None, data_path='attached_assets'):
    """Load and preprocess hyperspectral data"""
    try:
        print(f"\nLoading {name} dataset from {data_path}")
        if name == 'SV':
            data = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
            labels = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
        elif name == 'PU':
            data = sio.loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']
            labels = sio.loadmat(os.path.join(data_path, 'PaviaU_gt.mat'))['paviaU_gt']
        elif name == 'KSC':
            data = sio.loadmat(os.path.join(data_path, 'KSC.mat'))['KSC']
            labels = sio.loadmat(os.path.join(data_path, 'KSC_gt.mat'))['KSC_gt']
        elif name == 'IP':
            data = sio.loadmat(os.path.join(data_path, 'Indian_pines.mat'))['indian_pines']
            labels = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
        elif name == 'PC':
            data = sio.loadmat(os.path.join(data_path, 'Pavia.mat'))['pavia']
            labels = sio.loadmat(os.path.join(data_path, 'Pavia_gt.mat'))['pavia_gt']
        elif name == 'LO':
            data = sio.loadmat(os.path.join(data_path, 'Loukia.mat'))['ori_data']
            labels = sio.loadmat(os.path.join(data_path, 'Loukia_gt.mat'))['map']
        elif name == 'BO':
            data = sio.loadmat(os.path.join(data_path, 'Botswana.mat'))['Botswana']
            labels = sio.loadmat(os.path.join(data_path, 'Botswana_gt.mat'))['Botswana_gt']
        elif name == 'HR':
            data = sio.loadmat(os.path.join(data_path, 'hermiston2007.mat'))['HypeRvieW']
            labels = sio.loadmat(os.path.join(data_path, 'Hermiston_gt.mat'))['gt5clasesHermiston']
        else:
            raise ValueError(f"Dataset {name} not supported")

        print(f"Raw data shape: {data.shape}, labels shape: {labels.shape}")

        # Reshape and preprocess
        shapeor = data.shape
        data = data.reshape(-1, data.shape[-1])

        if num_components is not None:
            print(f"Reducing dimensions with PCA to {num_components} components...")
            pca = PCA(n_components=num_components)
            data = pca.fit_transform(data)
            print(f"Explained variance ratio: {sum(pca.explained_variance_ratio_):.4f}")
            shapeor = list(shapeor)
            shapeor[-1] = num_components

        print("Applying StandardScaler...")
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        data = data.reshape(shapeor)

        print(f"Unique labels: {np.unique(labels)}")
        num_class = len(np.unique(labels)) - 1
        print(f"Found {num_class + 1} classes (including background)")


        return data, labels, num_class

    except Exception as e:
        print(f"Error in loadData: {str(e)}")
        raise

def prepare_data(config):
    """Prepare data loaders for training"""
    try:
        # Load and preprocess data
        data, labels, num_classes = loadData(
            config.DATASET, 
            config.NUM_COMPONENTS, 
            data_path=config.DATA_PATH
        )

        print("\nCreating image cubes...")
        pixels, labels = createImageCubes(data, labels, windowSize=config.SPATIAL_SIZE)
        print(f"Created patches shape: {pixels.shape}")

        # Limit the number of samples to 25000
        if pixels.shape[0] > 25000:
            print(f"Limiting dataset to 25000 samples (original: {pixels.shape[0]})")
            indices = np.random.choice(pixels.shape[0], 25000, replace=False)
            pixels = pixels[indices]
            labels = labels[indices]

        # Split data
        print(f"\nSplitting data with train ratio: {config.TR_PERCENT}")
        x_train, x_test, y_train, y_test = train_test_split(
            pixels, labels, 
            train_size=config.TR_PERCENT,
            stratify=labels,
            random_state=42
        )
        print(f"Train set: {x_train.shape}, Test set: {x_test.shape}")

        if config.USE_VAL:
            print(f"Creating validation set with ratio: {config.VAL_PERCENT}")
            x_val, x_test, y_val, y_test = train_test_split(
                x_test, y_test,
                train_size=config.VAL_PERCENT,
                stratify=y_test,
                random_state=42
            )
            val_data = HyperData((np.transpose(x_val, (0, 3, 1, 2)).astype("float32"), y_val))
            val_loader = DataLoader(
                val_data, 
                batch_size=config.TEST_BATCH_SIZE, 
                shuffle=False,
                num_workers=0
            )
        else:
            val_loader = None

        # Create data loaders
        train_data = HyperData((np.transpose(x_train, (0, 3, 1, 2)).astype("float32"), y_train))
        test_data = HyperData((np.transpose(x_test, (0, 3, 1, 2)).astype("float32"), y_test))

        train_loader = DataLoader(
            train_data, 
            batch_size=config.TRAIN_BATCH_SIZE, 
            shuffle=True,
            num_workers=0
        )
        test_loader = DataLoader(
            test_data, 
            batch_size=config.TEST_BATCH_SIZE, 
            shuffle=False,
            num_workers=0
        )

        return train_loader, test_loader, val_loader, num_classes, pixels.shape[-1]

    except Exception as e:
        print(f"Error in prepare_data: {str(e)}")
        raise

def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
    """Create image cubes from the data with memory-efficient batch processing"""
    try:
        print(f"Input data shape: {X.shape}")
        margin = int((windowSize - 1) / 2)
        print(f"Adding padding with margin: {margin}")

        zeroPaddedX = padWithZeros(X, margin=margin)
        print(f"Padded data shape: {zeroPaddedX.shape}")

        # Calculate total patches and batch size
        total_patches = (X.shape[0] * X.shape[1])
        batch_size = min(5000, total_patches)
        print(f"Total patches to process: {total_patches}")
        print(f"Processing in batches of: {batch_size}")

        all_patches = []
        all_labels = []
        current_batch = []
        current_labels = []
        
        print("Creating patches in batches...")
        for r in range(margin, zeroPaddedX.shape[0] - margin):
            if r % 50 == 0:
                print(f"Processing row {r}/{zeroPaddedX.shape[0] - margin}")
            
            for c in range(margin, zeroPaddedX.shape[1] - margin):
                patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
                label = y[r-margin, c-margin]
                
                if not removeZeroLabels or label > 0:
                    current_batch.append(patch)
                    current_labels.append(label)
                
                # Process batch when it reaches the batch size
                if len(current_batch) >= batch_size:
                    patches_array = np.array(current_batch)
                    labels_array = np.array(current_labels)
                    
                    if removeZeroLabels:
                        labels_array = labels_array - 1  # Adjust labels
                    
                    all_patches.append(patches_array)
                    all_labels.append(labels_array)
                    
                    print(f"Processed batch of size {len(current_batch)}")
                    current_batch = []
                    current_labels = []

        # Process remaining patches
        if current_batch:
            patches_array = np.array(current_batch)
            labels_array = np.array(current_labels)
            
            if removeZeroLabels:
                labels_array = labels_array - 1  # Adjust labels
            
            all_patches.append(patches_array)
            all_labels.append(labels_array)

        # Combine all batches
        print("Combining all batches...")
        final_patches = np.concatenate(all_patches, axis=0)
        final_labels = np.concatenate(all_labels, axis=0)

        print(f"Final dataset shape - Patches: {final_patches.shape}, Labels: {final_labels.shape}")
        return final_patches, final_labels.astype("int")

    except Exception as e:
        print(f"Error in createImageCubes: {str(e)}")
        return None, None


def padWithZeros(X, margin=2):
    """Add zero padding to the data cube"""
    try:
        newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
        x_offset = margin
        y_offset = margin
        newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
        return newX
    except Exception as e:
        print(f"Error in padWithZeros: {str(e)}")
