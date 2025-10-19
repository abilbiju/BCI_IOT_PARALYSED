"""
Data Loader for BCI Motor Imagery Classification
Phase 1: Core BCI Application (Intent Layer)

This module handles loading and preparing the BCI Competition IV data
for training the EEGNet model. It processes the NPZ file containing
EEG data and markers, and prepares it for motor imagery classification.
"""

import numpy as np
import yaml
import logging
from typing import Tuple, Dict, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import os
from preprocessing import PreProcessor


class BCIDataLoader:
    """
    Data loader for BCI motor imagery data.
    
    Handles loading from NPZ files, preprocessing, and preparing
    data for training/validation/testing splits.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize data loader with configuration.
        
        Args:
            config_path: Path to configuration YAML file
        """
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format']
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize preprocessor
        self.preprocessor = PreProcessor(config_path)
        
        # Data storage
        self.raw_data = None
        self.processed_data = None
        self.labels = None
        
        # Model parameters
        self.motor_imagery_channels = self.config['eeg']['motor_imagery_channels']
        self.sampling_rate = self.config['eeg']['sampling_rate']
        
    def load_npz_data(self, npz_path: Optional[str] = None) -> Dict:
        """
        Load BCI data from NPZ file.
        
        Args:
            npz_path: Path to NPZ file (uses config path if None)
            
        Returns:
            Dictionary containing loaded data
        """
        if npz_path is None:
            npz_path = self.config['paths']['data_file']
        
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Data file not found: {npz_path}")
        
        self.logger.info(f"Loading data from: {npz_path}")
        
        # Load NPZ file
        data = np.load(npz_path)
        
        # Separate EEG data files and marker files
        eeg_files = {k: v for k, v in data.items() if 'cnt' in k}
        marker_files = {k: v for k, v in data.items() if 'mrk' in k}
        
        self.logger.info(f"Found {len(eeg_files)} EEG files and {len(marker_files)} marker files")
        
        return {
            'eeg_files': eeg_files,
            'marker_files': marker_files,
            'all_data': data
        }
    
    def match_eeg_and_markers(self, eeg_files: Dict, marker_files: Dict) -> List[Tuple]:
        """
        Match EEG data files with their corresponding marker files.
        
        Args:
            eeg_files: Dictionary of EEG data files
            marker_files: Dictionary of marker files
            
        Returns:
            List of (eeg_data, markers, subject_id) tuples
        """
        matched_pairs = []
        
        for eeg_key, eeg_data in eeg_files.items():
            # Extract subject identifier from filename
            # e.g., "dataset_1_BCICIV_calib_ds1a_cnt.txt" -> "ds1a"
            parts = eeg_key.split('_')
            subject_id = None
            for part in parts:
                if part.startswith('ds1') and 'cnt' not in part:
                    subject_id = part
                    break
            
            if subject_id is None:
                continue
            
            # Find corresponding marker file
            marker_key = None
            for mk_key in marker_files.keys():
                if subject_id in mk_key and 'mrk' in mk_key:
                    marker_key = mk_key
                    break
            
            if marker_key is not None:
                markers = marker_files[marker_key]
                matched_pairs.append((eeg_data, markers, subject_id))
                self.logger.info(f"Matched {eeg_key} with {marker_key} (Subject: {subject_id})")
            else:
                self.logger.warning(f"No marker file found for {eeg_key}")
        
        return matched_pairs
    
    def process_subject_data(self, eeg_data: np.ndarray, markers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process data for a single subject.
        
        Args:
            eeg_data: Raw EEG data of shape (samples, channels)
            markers: Event markers of shape (events, 2)
            
        Returns:
            processed_epochs: Preprocessed epochs
            labels: Corresponding labels
        """
        # Filter markers for motor imagery classes only (1 and -1)
        valid_markers = markers[np.isin(markers[:, 1], [1, -1])]
        
        if len(valid_markers) == 0:
            self.logger.warning("No valid motor imagery markers found")
            return np.array([]), np.array([])
        
        # Process the data through preprocessing pipeline
        epochs, labels = self.preprocessor.preprocess_continuous_data(
            eeg_data, 
            valid_markers,
            selected_channels=self.motor_imagery_channels
        )
        
        return epochs, labels
    
    def load_and_process_all_data(self, npz_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and process all BCI data for training.
        
        Args:
            npz_path: Path to NPZ file (uses config path if None)
            
        Returns:
            X: Processed EEG epochs of shape (trials, channels, samples)
            y: Labels corresponding to each epoch
        """
        self.logger.info("Loading and processing all BCI data...")
        
        # Load raw data
        data_dict = self.load_npz_data(npz_path)
        eeg_files = data_dict['eeg_files']
        marker_files = data_dict['marker_files']
        
        # Match EEG files with marker files
        matched_pairs = self.match_eeg_and_markers(eeg_files, marker_files)
        
        if not matched_pairs:
            raise ValueError("No valid EEG-marker pairs found")
        
        # Process each subject's data
        all_epochs = []
        all_labels = []
        
        for eeg_data, markers, subject_id in matched_pairs:
            self.logger.info(f"Processing subject: {subject_id}")
            
            epochs, labels = self.process_subject_data(eeg_data, markers)
            
            if len(epochs) > 0:
                all_epochs.append(epochs)
                all_labels.append(labels)
                self.logger.info(f"  - Processed {len(epochs)} epochs")
            else:
                self.logger.warning(f"  - No valid epochs for subject {subject_id}")
        
        if not all_epochs:
            raise ValueError("No valid epochs processed from any subject")
        
        # Concatenate all data
        X = np.concatenate(all_epochs, axis=0)
        y = np.concatenate(all_labels, axis=0)
        
        self.logger.info(f"Total processed data: {X.shape[0]} epochs")
        self.logger.info(f"Data shape: {X.shape}")
        
        # Show label distribution (handles negative values)
        unique_labels, counts = np.unique(y, return_counts=True)
        label_dist = dict(zip(unique_labels, counts))
        self.logger.info(f"Label distribution: {label_dist}")
        
        # Store processed data
        self.processed_data = X
        self.labels = y
        
        return X, y
    
    def prepare_for_training(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        validation_split: float = 0.2,
        test_split: float = 0.1,
        random_state: int = 42
    ) -> Dict[str, np.ndarray]:
        """
        Prepare data for training with train/validation/test splits.
        
        Args:
            X: Input data of shape (trials, channels, samples)
            y: Labels
            validation_split: Fraction of data for validation
            test_split: Fraction of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing train/validation/test splits
        """
        self.logger.info("Preparing data for training...")
        
        # Convert labels to categorical (binary classification: -1, 1 -> 0, 1)
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        self.logger.info(f"Label encoding: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
        
        # Convert to categorical (one-hot encoding)
        y_categorical = to_categorical(y_encoded)
        
        # Reshape for EEGNet: transpose from (trials, time, channels) to (trials, channels, time, 1)
        X_transposed = X.transpose(0, 2, 1)  # (trials, time, channels) -> (trials, channels, time)
        X_reshaped = X_transposed[..., np.newaxis]  # Add final dimension for Conv2D
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_reshaped, y_categorical,
            test_size=test_split,
            random_state=random_state,
            stratify=y_categorical.argmax(axis=1)
        )
        
        # Second split: separate train and validation
        validation_size = validation_split / (1 - test_split)  # Adjust for already removed test set
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=validation_size,
            random_state=random_state,
            stratify=y_temp.argmax(axis=1)
        )
        
        # Log split information
        self.logger.info(f"Data splits:")
        self.logger.info(f"  - Training: {X_train.shape[0]} samples ({X_train.shape[0]/X_reshaped.shape[0]*100:.1f}%)")
        self.logger.info(f"  - Validation: {X_val.shape[0]} samples ({X_val.shape[0]/X_reshaped.shape[0]*100:.1f}%)")
        self.logger.info(f"  - Test: {X_test.shape[0]} samples ({X_test.shape[0]/X_reshaped.shape[0]*100:.1f}%)")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'label_encoder': label_encoder
        }
    
    def get_data_info(self) -> Dict:
        """
        Get information about the loaded data.
        
        Returns:
            Dictionary containing data information
        """
        if self.processed_data is None or self.labels is None:
            return {"status": "No data loaded"}
        
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        
        return {
            "total_epochs": len(self.processed_data),
            "data_shape": self.processed_data.shape,
            "label_distribution": dict(zip(unique_labels, counts)),
            "channels_used": len(self.motor_imagery_channels),
            "sampling_rate": self.sampling_rate,
            "epoch_length": self.config['eeg']['epoch_length']
        }
    
    def augment_data(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        noise_std: float = 0.01,
        time_shift_range: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply data augmentation to increase training data diversity.
        
        Args:
            X: Input data of shape (trials, channels, samples, 1)
            y: Labels
            noise_std: Standard deviation for Gaussian noise
            time_shift_range: Maximum samples to shift in time
            
        Returns:
            Augmented data and labels
        """
        self.logger.info("Applying data augmentation...")
        
        X_aug = []
        y_aug = []
        
        for i in range(len(X)):
            # Original data
            X_aug.append(X[i])
            y_aug.append(y[i])
            
            # Add Gaussian noise
            noise = np.random.normal(0, noise_std, X[i].shape)
            X_noisy = X[i] + noise
            X_aug.append(X_noisy)
            y_aug.append(y[i])
            
            # Time shifting
            if time_shift_range > 0:
                shift = np.random.randint(-time_shift_range, time_shift_range + 1)
                X_shifted = np.roll(X[i], shift, axis=1)  # Shift along time axis
                X_aug.append(X_shifted)
                y_aug.append(y[i])
        
        X_augmented = np.array(X_aug)
        y_augmented = np.array(y_aug)
        
        self.logger.info(f"Data augmentation completed: {len(X)} -> {len(X_augmented)} samples")
        
        return X_augmented, y_augmented


def test_data_loader():
    """Test function for data loader."""
    print("Testing BCI data loader...")
    
    # Initialize data loader
    data_loader = BCIDataLoader()
    
    # Load and process data
    try:
        X, y = data_loader.load_and_process_all_data()
        print(f"Loaded data shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Unique labels: {np.unique(y)}")
        
        # Prepare for training
        data_splits = data_loader.prepare_for_training(X, y)
        
        print(f"Training data shape: {data_splits['X_train'].shape}")
        print(f"Validation data shape: {data_splits['X_val'].shape}")
        print(f"Test data shape: {data_splits['X_test'].shape}")
        
        # Get data info
        info = data_loader.get_data_info()
        print(f"Data info: {info}")
        
        print("Data loader test completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        print("Make sure the bci_preprocessed_data.npz file exists and is valid")


if __name__ == "__main__":
    test_data_loader()