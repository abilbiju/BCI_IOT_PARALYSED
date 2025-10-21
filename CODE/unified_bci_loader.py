"""
Unified BCI Data Loader
Combines BCI Competition III Dataset 3a and IV Dataset 2a for enhanced training

This module provides a unified interface to load and combine multiple BCI datasets,
dramatically increasing the available training data and improving model generalization.
"""

import numpy as np
import yaml
import logging
import os
from typing import Tuple, Dict, List, Optional
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from gdf_data_loader import GDFDataLoader
from bci4_2a_loader import BCI4_2A_Loader


class UnifiedBCIDataLoader:
    """
    Unified data loader that combines multiple BCI Competition datasets.
    
    Supports:
    - BCI Competition III Dataset 3a (k3b subject)
    - BCI Competition IV Dataset 2a (9 subjects)
    - Automatic dataset detection and standardization
    - Enhanced data augmentation for combined datasets
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize unified BCI data loader."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format']
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize individual loaders
        self.bci3_loader = GDFDataLoader(config_path)
        self.bci4_loader = BCI4_2A_Loader(config_path)
        
        # Dataset paths
        self.bci3_path = "BCI/k3b"
        self.bci4_path = "BCI/bci4_2a"
        
        # Target specifications (standardized to BCI IV 2a)
        self.target_channels = 22  # EEG channels from BCI IV 2a
        self.target_sampling_rate = 250  # Hz
        self.target_classes = 4  # left hand, right hand, foot, tongue
        self.class_names = ['left_hand', 'right_hand', 'foot', 'tongue']
        
    def detect_available_datasets(self) -> Dict[str, bool]:
        """
        Detect which datasets are available.
        
        Returns:
            Dictionary indicating dataset availability
        """
        availability = {
            'bci_iii_3a': False,
            'bci_iv_2a': False
        }
        
        # Check BCI Competition III Dataset 3a
        bci3_files = ['k3b_s.txt', 'k3b_HDR_TRIG.txt', 'k3b_HDR_Classlabel.txt']
        if all(os.path.exists(os.path.join(self.bci3_path, f)) for f in bci3_files):
            availability['bci_iii_3a'] = True
            self.logger.info("✓ BCI Competition III Dataset 3a detected")
        else:
            self.logger.info("✗ BCI Competition III Dataset 3a not available")
        
        # Check BCI Competition IV Dataset 2a
        bci4_files = [f"A{i:02d}T.gdf" for i in range(1, 10)]  # A01T.gdf to A09T.gdf
        available_subjects = sum(1 for f in bci4_files if os.path.exists(os.path.join(self.bci4_path, f)))
        
        if available_subjects > 0:
            availability['bci_iv_2a'] = True
            self.logger.info(f"✓ BCI Competition IV Dataset 2a detected ({available_subjects}/9 subjects)")
        else:
            self.logger.info("✗ BCI Competition IV Dataset 2a not available")
        
        return availability
    
    def load_bci3_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load BCI Competition III Dataset 3a.
        
        Returns:
            epochs: Preprocessed epochs
            labels: Class labels
        """
        self.logger.info("Loading BCI Competition III Dataset 3a...")
        
        try:
            # Load using existing GDF loader
            epochs, labels = self.bci3_loader.load_and_preprocess_data()
            
            self.logger.info(f"BCI III 3a loaded: {epochs.shape[0]} epochs")
            return epochs, labels
            
        except Exception as e:
            self.logger.error(f"Failed to load BCI III 3a: {e}")
            return np.array([]), np.array([])
    
    def load_bci4_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load BCI Competition IV Dataset 2a.
        
        Returns:
            epochs: Preprocessed epochs  
            labels: Class labels
        """
        self.logger.info("Loading BCI Competition IV Dataset 2a...")
        
        try:
            # Load using BCI4 2a loader
            epochs, labels = self.bci4_loader.load_all_subjects('T')  # Training session
            
            self.logger.info(f"BCI IV 2a loaded: {epochs.shape[0]} epochs")
            return epochs, labels
            
        except Exception as e:
            self.logger.error(f"Failed to load BCI IV 2a: {e}")
            return np.array([]), np.array([])
    
    def standardize_channel_count(self, epochs: np.ndarray, dataset_type: str) -> np.ndarray:
        """
        Standardize channel count across datasets.
        
        Args:
            epochs: Input epochs (trials, samples, channels)
            dataset_type: Type of dataset ('bci3' or 'bci4')
            
        Returns:
            Standardized epochs with target channel count
        """
        current_channels = epochs.shape[2]
        
        if dataset_type == 'bci4':
            # BCI IV 2a already has 22 EEG channels - use as is
            if current_channels >= self.target_channels:
                return epochs[:, :, :self.target_channels]
            else:
                # Pad with zeros if somehow fewer channels
                padded = np.zeros((epochs.shape[0], epochs.shape[1], self.target_channels))
                padded[:, :, :current_channels] = epochs
                return padded
        
        elif dataset_type == 'bci3':
            # BCI III 3a has different channel layout - select motor cortex channels
            if current_channels > self.target_channels:
                # Select central motor cortex channels
                motor_channels = self.config['eeg']['motor_imagery_channels'][:self.target_channels]
                if len(motor_channels) == self.target_channels and max(motor_channels) < current_channels:
                    return epochs[:, :, motor_channels]
                else:
                    # Fallback: select first 22 channels
                    return epochs[:, :, :self.target_channels]
            else:
                # Pad if fewer channels
                padded = np.zeros((epochs.shape[0], epochs.shape[1], self.target_channels))
                padded[:, :, :current_channels] = epochs
                return padded
        
        return epochs
    
    def load_combined_datasets(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and combine all available datasets.
        
        Returns:
            combined_epochs: Combined preprocessed epochs
            combined_labels: Combined class labels
        """
        self.logger.info("Loading and combining all available BCI datasets...")
        
        # Detect available datasets
        availability = self.detect_available_datasets()
        
        all_epochs = []
        all_labels = []
        dataset_sources = []
        
        # Load BCI Competition III Dataset 3a
        if availability['bci_iii_3a']:
            try:
                epochs_3a, labels_3a = self.load_bci3_dataset()
                
                if len(epochs_3a) > 0:
                    # Standardize channel count
                    epochs_3a_std = self.standardize_channel_count(epochs_3a, 'bci3')
                    
                    all_epochs.append(epochs_3a_std)
                    all_labels.append(labels_3a)
                    dataset_sources.extend(['bci3'] * len(epochs_3a_std))
                    
                    self.logger.info(f"Added BCI III 3a: {len(epochs_3a_std)} epochs")
                
            except Exception as e:
                self.logger.warning(f"Failed to load BCI III 3a: {e}")
        
        # Load BCI Competition IV Dataset 2a
        if availability['bci_iv_2a']:
            try:
                epochs_2a, labels_2a = self.load_bci4_dataset()
                
                if len(epochs_2a) > 0:
                    # Standardize channel count
                    epochs_2a_std = self.standardize_channel_count(epochs_2a, 'bci4')
                    
                    all_epochs.append(epochs_2a_std)
                    all_labels.append(labels_2a)
                    dataset_sources.extend(['bci4'] * len(epochs_2a_std))
                    
                    self.logger.info(f"Added BCI IV 2a: {len(epochs_2a_std)} epochs")
                
            except Exception as e:
                self.logger.warning(f"Failed to load BCI IV 2a: {e}")
        
        # Check if any datasets were loaded
        if not all_epochs:
            raise ValueError("No datasets were loaded successfully")
        
        # Combine all datasets
        combined_epochs = np.vstack(all_epochs)
        combined_labels = np.concatenate(all_labels)
        
        self.logger.info(f"Combined dataset created:")
        self.logger.info(f"  Total epochs: {combined_epochs.shape[0]}")
        self.logger.info(f"  Shape: {combined_epochs.shape}")
        self.logger.info(f"  Channel count: {combined_epochs.shape[2]}")
        
        # Show class distribution
        unique_labels, counts = np.unique(combined_labels, return_counts=True)
        self.logger.info("  Class distribution:")
        for label_idx, count in zip(unique_labels, counts):
            class_name = self.class_names[label_idx]
            self.logger.info(f"    {class_name}: {count} epochs")
        
        # Show dataset source distribution
        unique_sources, source_counts = np.unique(dataset_sources, return_counts=True)
        self.logger.info("  Dataset sources:")
        for source, count in zip(unique_sources, source_counts):
            dataset_name = "BCI III 3a" if source == 'bci3' else "BCI IV 2a"
            self.logger.info(f"    {dataset_name}: {count} epochs")
        
        return combined_epochs, combined_labels
    
    def enhanced_data_augmentation(self, X: np.ndarray, y: np.ndarray, 
                                 augmentation_factor: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply enhanced data augmentation for combined datasets.
        
        Args:
            X: Training data (trials, samples, channels)
            y: Training labels
            augmentation_factor: Number of augmented copies per original sample
            
        Returns:
            Augmented X and y
        """
        self.logger.info(f"Applying enhanced data augmentation (factor={augmentation_factor})...")
        
        original_count = len(X)
        augmented_X = [X]  # Start with original data
        augmented_y = [y]
        
        for aug_round in range(augmentation_factor):
            self.logger.info(f"Augmentation round {aug_round + 1}/{augmentation_factor}")
            
            X_aug = X.copy()
            
            # 1. Gaussian noise (varying levels)
            noise_std = 0.05 + (aug_round * 0.01)  # Increasing noise levels
            noise = np.random.normal(0, noise_std, X_aug.shape)
            X_aug += noise
            
            # 2. Time shifting (different amounts)
            shift_samples = np.random.randint(-15, 16)  # ±60ms at 250Hz
            X_aug = np.roll(X_aug, shift_samples, axis=1)
            
            # 3. Amplitude scaling
            scale_factor = 0.85 + (np.random.rand() * 0.3)  # 0.85 to 1.15
            X_aug *= scale_factor
            
            # 4. Channel dropout (randomly zero some channels)
            if aug_round % 2 == 0:  # Apply every other round
                n_dropout = np.random.randint(1, 4)  # Drop 1-3 channels
                dropout_channels = np.random.choice(X_aug.shape[2], n_dropout, replace=False)
                X_aug[:, :, dropout_channels] = 0
            
            # 5. Temporal smoothing
            if aug_round % 3 == 0:  # Apply every third round
                from scipy import ndimage
                sigma = 0.5 + np.random.rand() * 0.5  # 0.5 to 1.0
                for trial in range(X_aug.shape[0]):
                    for ch in range(X_aug.shape[2]):
                        X_aug[trial, :, ch] = ndimage.gaussian_filter1d(X_aug[trial, :, ch], sigma)
            
            augmented_X.append(X_aug)
            augmented_y.append(y)
        
        # Combine all augmented data
        X_final = np.vstack(augmented_X)
        y_final = np.concatenate(augmented_y)
        
        self.logger.info(f"Data augmentation completed:")
        self.logger.info(f"  Original: {original_count} epochs")
        self.logger.info(f"  Augmented: {len(X_final)} epochs")
        self.logger.info(f"  Increase factor: {len(X_final) / original_count:.1f}x")
        
        return X_final, y_final
    
    def prepare_for_training(self, augment_data: bool = True) -> Dict:
        """
        Complete pipeline to prepare combined datasets for training.
        
        Args:
            augment_data: Whether to apply data augmentation
            
        Returns:
            Dictionary with prepared data splits
        """
        self.logger.info("Starting complete unified data preparation pipeline...")
        
        # Load combined datasets
        epochs, labels = self.load_combined_datasets()
        
        # Convert labels to categorical
        y_categorical = to_categorical(labels, num_classes=4)
        
        # Reshape for EEGNet: (trials, samples, channels) -> (trials, channels, samples, 1)
        X_reshaped = epochs.transpose(0, 2, 1)  # (trials, channels, samples)
        X_reshaped = X_reshaped[..., np.newaxis]  # Add final dimension
        
        self.logger.info(f"Reshaped for EEGNet: {epochs.shape} -> {X_reshaped.shape}")
        
        # Create initial splits (larger test set due to more data)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_reshaped, y_categorical,
            test_size=0.15,  # 15% for testing
            random_state=42,
            stratify=labels
        )
        
        # Split remaining into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=0.176,  # ~15% of total for validation
            random_state=42,
            stratify=y_temp.argmax(axis=1)
        )
        
        # Apply enhanced data augmentation to training set
        if augment_data and len(X_train) > 50:  # Only augment if sufficient data
            X_train_samples = X_train.squeeze(-1).transpose(0, 2, 1)  # Back to (trials, samples, channels)
            y_train_indices = y_train.argmax(axis=1)
            
            X_train_aug, y_train_aug_indices = self.enhanced_data_augmentation(
                X_train_samples, y_train_indices, augmentation_factor=2
            )
            
            # Convert back to EEGNet format
            X_train = X_train_aug.transpose(0, 2, 1)[..., np.newaxis]
            y_train = to_categorical(y_train_aug_indices, num_classes=4)
        
        # Log final split information
        self.logger.info(f"Final data splits:")
        self.logger.info(f"  - Training: {X_train.shape[0]} samples ({X_train.shape[0]/(X_train.shape[0]+X_val.shape[0]+X_test.shape[0])*100:.1f}%)")
        self.logger.info(f"  - Validation: {X_val.shape[0]} samples ({X_val.shape[0]/(X_train.shape[0]+X_val.shape[0]+X_test.shape[0])*100:.1f}%)")
        self.logger.info(f"  - Test: {X_test.shape[0]} samples ({X_test.shape[0]/(X_train.shape[0]+X_val.shape[0]+X_test.shape[0])*100:.1f}%)")
        
        # Show class balance in each split
        for split_name, y_split in [("Training", y_train), ("Validation", y_val), ("Test", y_test)]:
            labels_split = y_split.argmax(axis=1)
            unique_labels, counts = np.unique(labels_split, return_counts=True)
            distribution = {self.class_names[i]: 0 for i in range(4)}
            for label, count in zip(unique_labels, counts):
                distribution[self.class_names[label]] = count
            self.logger.info(f"  - {split_name} distribution: {distribution}")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'original_labels': labels,
            'dataset_info': {
                'total_epochs': len(epochs),
                'n_channels': epochs.shape[2],
                'n_samples': epochs.shape[1],
                'n_classes': 4,
                'class_names': self.class_names
            }
        }


def test_unified_loader():
    """Test the unified BCI data loader."""
    print("Testing Unified BCI Data Loader...")
    
    try:
        # Initialize loader
        loader = UnifiedBCIDataLoader()
        
        # Test dataset detection
        print("\nTesting dataset detection...")
        availability = loader.detect_available_datasets()
        print(f"Dataset availability: {availability}")
        
        # Test combined loading
        print("\nTesting combined dataset loading...")
        data_splits = loader.prepare_for_training(augment_data=True)
        
        print(f"Training data: {data_splits['X_train'].shape}")
        print(f"Validation data: {data_splits['X_val'].shape}")
        print(f"Test data: {data_splits['X_test'].shape}")
        print(f"Dataset info: {data_splits['dataset_info']}")
        
        print("\nUnified loader test completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_unified_loader()