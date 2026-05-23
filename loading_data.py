import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class MRNetDataset(Dataset):
    def __init__(self, data_dir, labels_csv_path, split='train', view='sagittal'):
        """
        Args:
            data_dir (str): Root path to the extracted mrnet_images folder.
            labels_csv_path (str): Path to our verified unified_labels.csv.
            split (str): 'train' or 'valid'.
            view (str): 'sagittal', 'coronal', or 'axial'.
        """
        # 1. Load the single source of truth we generated
        self.labels_df = pd.read_csv(labels_csv_path, index_col='exam_id', dtype={'exam_id': str})
        
        # 2. Filter by the requested split
        self.labels_df = self.labels_df[self.labels_df['split'] == split]
        
        self.data_dir = data_dir
        self.split = split
        self.view = view.lower()
        self.exam_ids = self.labels_df.index.tolist()

    def __len__(self):
        return len(self.exam_ids)

    def __getitem__(self, idx):
        exam_id = self.exam_ids[idx]
        
        # 3. Dynamic Pathing (handles both train and valid folder structures)
        if self.split == 'train':
            file_path = os.path.join(self.data_dir, self.view, f"{exam_id}.npy")
        else:
            file_path = os.path.join(self.data_dir, 'valid_files', self.view, f"{exam_id}.npy")
            
        # 4. Load raw volume
        volume = np.load(file_path)
        
        # 5. Tensor Conversion & Normalization
        # Convert uint8 [0, 255] to PyTorch float32 [0.0, 1.0]
        tensor_volume = torch.tensor(volume, dtype=torch.float32) / 255.0
        
        # 6. Pseudo-RGB Channel Stacking
        # Raw shape is (Slices, Height, Width). Pre-trained models need 3 color channels.
        # unsqueeze(1) creates (Slices, 1, Height, Width)
        # expand duplicates the grayscale channel 3 times -> (Slices, 3, Height, Width)
        tensor_volume = tensor_volume.unsqueeze(1).expand(-1, 3, -1, -1)
        
        # 7. Fetch the Multi-Task Labels [Abnormal, ACL, Meniscus]
        row = self.labels_df.loc[exam_id]
        labels = torch.tensor([row['abnormal'], row['acl'], row['meniscus']], dtype=torch.float32)
        
        return tensor_volume, labels

def get_data_loaders(data_dir, labels_csv_path, view='sagittal', batch_size=1, num_workers=0):
    """
    Factory function to generate both train and validation DataLoaders.
    Keeps the Colab notebook clean.
    """
    train_dataset = MRNetDataset(data_dir, labels_csv_path, split='train', view=view)
    valid_dataset = MRNetDataset(data_dir, labels_csv_path, split='valid', view=view)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, valid_loader
## run the test 
if __name__ == '__main__':
    print("--- Testing PyTorch Data Pipeline ---")
    # Change this path to where your mrnet_images folder actually lives locally
    LOCAL_ROOT = "./mrnet_images" 
    LABEL_PATH = "./unified_labels.csv"
    
    # Instantiate the factory
    train_loader, valid_loader = get_data_loaders(LOCAL_ROOT, LABEL_PATH, view='sagittal', batch_size=1)
    
    # Fetch exactly one batch (one patient)
    sample_volume, sample_labels = next(iter(train_loader))
    
    print(f"Success! Model Input Volume Shape: {sample_volume.shape}")
    print(f"Success! Model Target Labels Shape: {sample_labels.shape}")
    print(f"Target Label Values (Abnormal, ACL, Meniscus): {sample_labels}")