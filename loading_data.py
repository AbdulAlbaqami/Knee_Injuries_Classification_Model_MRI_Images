import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def load_unified_labels(data_dir, split='train'):
    """
    Reads the three separate diagnosis CSVs and merges them into a single DataFrame.
    Splits are typically 'train' or 'valid'.
    """
    # MRNet CSVs do not have headers; they use format: exam_id, label
    categories = ['abnormality', 'acl', 'meniscus']
    dfs = []
    
    for cat in categories:
        csv_path = os.path.join(data_dir, f'{split}-{cat}.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Expected CSV file not found at: {csv_path}")
            
        df = pd.read_csv(csv_path, header=None, names=['exam_id', cat], dtype={'exam_id': str})
        dfs.append(df)
    
    # Merge all three dataframes on the unique patient exam_id
    unified_df = dfs[0]
    for df in dfs[1:]:
        unified_df = pd.merge(unified_df, df, on='exam_id')
        
    # Set exam_id as index for rapid O(1) lookups during batch generation
    unified_df.set_index('exam_id', inplace=True)
    return unified_df


class MRNetDataset(Dataset):
    def __init__(self, data_dir, label_df, plane='sagittal', transform=None):
        """
        Args:
            data_dir (str): Path to the extracted MRNet dataset folder.
            label_df (DataFrame): The unified labels indexed by 4-digit exam_id string.
            plane (str): One of 'sagittal', 'coronal', or 'axial'.
            transform (callable, optional): Transformations applied per slice.
        """
        self.data_dir = data_dir
        self.label_df = label_df
        self.plane = plane.lower()
        self.exam_ids = label_df.index.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.exam_ids)

    def __getitem__(self, idx):
        exam_id = self.exam_ids[idx]
        
        # Construct absolute path to the .npy volume file
        # Example target: data_dir/train/sagittal/0000.npy
        split_folder = 'train' if exam_id in self.label_df.index else 'valid' # Handled based on context or setup
        # For simplicity, we assume label_df corresponds entirely to the correct split folder
        
        # Let's dynamically locate the split path based on parent function context
        # A clean layout option is grouping by data_dir/split/plane/ID.npy
        # We will refine the pathing once your exact local folder hierarchy is confirmed
        return exam_id
    
## This code is meant to be run in a local environment where the MRNet dataset is available.
if __name__ == '__main__':
    print("--- Running Local Data Pipeline Verification ---")
    
    # CHANGE THIS to your actual local data directory path for testing
    LOCAL_DATA_DIR = "./mini_data_patch" 
    
    try:
        # Test Step 1: Label Parsing
        print("Testing label parser...")
        labels = load_unified_labels(LOCAL_DATA_DIR, split='train')
        print(f"Successfully loaded labels. Shape: {labels.shape}")
        print("Sample data mapping:")
        print(labels.head(3))
        
        # Test Step 2: Dataset Instantiation
        print("\nTesting Dataset Initialization...")
        dataset = MRNetDataset(data_dir=LOCAL_DATA_DIR, label_df=labels, plane='sagittal')
        print(f"Dataset wrapper initialized with {len(dataset)} local samples.")
        
    except Exception as e:
        print(f"\n[ERROR] Pipeline test failed: {str(e)}")