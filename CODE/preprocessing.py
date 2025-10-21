"""
Signal Preprocessing Module for BCI Motor Imagery Classification
Phase 1: Core BCI Application (Intent Layer)

This module contains all signal preprocessing operations required for EEG data
before feeding it to the EEGNet model for motor imagery classification.
"""

import numpy as np
import yaml
from scipy import signal
from scipy.signal import butter, filtfilt, hilbert
from sklearn.preprocessing import StandardScaler
import logging
from typing import Tuple, Optional, Union


class PreProcessor:
    """
    Static class containing all preprocessing methods for EEG signals.
    Ensures consistency between training and inference phases.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize preprocessor with configuration parameters."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.sampling_rate = self.config['eeg']['sampling_rate']
        self.preprocessing_config = self.config['preprocessing']
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format']
        )
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    def apply_bandpass_filter(
        data: np.ndarray, 
        low_freq: float, 
        high_freq: float, 
        sampling_rate: float,
        filter_order: int = 4
    ) -> np.ndarray:
        """
        Apply bandpass Butterworth filter to isolate motor imagery frequency bands.
        
        Args:
            data: EEG data of shape (samples, channels) or (trials, samples, channels)
            low_freq: Lower cutoff frequency (Hz)
            high_freq: Upper cutoff frequency (Hz)
            sampling_rate: Sampling frequency (Hz)
            filter_order: Filter order for Butterworth filter
            
        Returns:
            Filtered EEG data with same shape as input
        """
        # Design Butterworth bandpass filter
        nyquist = sampling_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # Ensure frequencies are in valid range
        low = max(low, 0.01)  # Avoid DC
        high = min(high, 0.99)  # Avoid Nyquist
        
        b, a = butter(filter_order, [low, high], btype='band')
        
        # Apply filter
        if data.ndim == 2:  # (samples, channels)
            filtered_data = np.zeros_like(data)
            for ch in range(data.shape[1]):
                filtered_data[:, ch] = filtfilt(b, a, data[:, ch])
        elif data.ndim == 3:  # (trials, samples, channels)
            filtered_data = np.zeros_like(data)
            for trial in range(data.shape[0]):
                for ch in range(data.shape[2]):
                    filtered_data[trial, :, ch] = filtfilt(b, a, data[trial, :, ch])
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}")
            
        return filtered_data
    
    @staticmethod
    def apply_car_reference(data: np.ndarray) -> np.ndarray:
        """
        Apply Common Average Reference (CAR) to remove common noise.
        
        Args:
            data: EEG data of shape (samples, channels) or (trials, samples, channels)
            
        Returns:
            CAR-referenced EEG data with same shape as input
        """
        if data.ndim == 2:  # (samples, channels)
            # Compute average across channels for each sample
            average_ref = np.mean(data, axis=1, keepdims=True)
            referenced_data = data - average_ref
        elif data.ndim == 3:  # (trials, samples, channels)
            # Compute average across channels for each trial and sample
            average_ref = np.mean(data, axis=2, keepdims=True)
            referenced_data = data - average_ref
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}")
            
        return referenced_data
    
    @staticmethod
    def extract_epochs(
        data: np.ndarray, 
        markers: np.ndarray, 
        epoch_length: float, 
        sampling_rate: float,
        baseline_length: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract epochs from continuous EEG data based on event markers.
        
        Args:
            data: Continuous EEG data of shape (samples, channels)
            markers: Event markers of shape (events, 2) - [position, class]
            epoch_length: Length of each epoch in seconds
            sampling_rate: Sampling frequency (Hz)
            baseline_length: Length of baseline period in seconds before event
            
        Returns:
            epochs: Extracted epochs of shape (trials, samples, channels)
            labels: Corresponding labels for each epoch
        """
        samples_per_epoch = int(epoch_length * sampling_rate)
        baseline_samples = int(baseline_length * sampling_rate)
        
        epochs = []
        labels = []
        
        for marker in markers:
            start_sample = int(marker[0]) - baseline_samples
            end_sample = start_sample + samples_per_epoch
            
            # Check if epoch is within data bounds
            if start_sample >= 0 and end_sample <= data.shape[0]:
                epoch = data[start_sample:end_sample, :]
                epochs.append(epoch)
                labels.append(int(marker[1]))
        
        return np.array(epochs), np.array(labels)
    
    @staticmethod
    def normalize_epochs(epochs: np.ndarray, method: str = "zscore") -> np.ndarray:
        """
        Normalize epochs to improve model performance.
        
        Args:
            epochs: EEG epochs of shape (trials, samples, channels)
            method: Normalization method ("zscore", "minmax", or "robust")
            
        Returns:
            Normalized epochs with same shape as input
        """
        if method == "zscore":
            # Z-score normalization (zero mean, unit variance)
            normalized_epochs = np.zeros_like(epochs)
            for trial in range(epochs.shape[0]):
                for ch in range(epochs.shape[2]):
                    data = epochs[trial, :, ch]
                    if np.std(data) > 0:  # Avoid division by zero
                        normalized_epochs[trial, :, ch] = (data - np.mean(data)) / np.std(data)
                    else:
                        normalized_epochs[trial, :, ch] = data
                        
        elif method == "minmax":
            # Min-max normalization to [0, 1]
            normalized_epochs = np.zeros_like(epochs)
            for trial in range(epochs.shape[0]):
                for ch in range(epochs.shape[2]):
                    data = epochs[trial, :, ch]
                    data_min, data_max = np.min(data), np.max(data)
                    if data_max - data_min > 0:  # Avoid division by zero
                        normalized_epochs[trial, :, ch] = (data - data_min) / (data_max - data_min)
                    else:
                        normalized_epochs[trial, :, ch] = data
                        
        elif method == "robust":
            # Robust normalization using median and IQR
            normalized_epochs = np.zeros_like(epochs)
            for trial in range(epochs.shape[0]):
                for ch in range(epochs.shape[2]):
                    data = epochs[trial, :, ch]
                    median = np.median(data)
                    q75, q25 = np.percentile(data, [75, 25])
                    iqr = q75 - q25
                    if iqr > 0:  # Avoid division by zero
                        normalized_epochs[trial, :, ch] = (data - median) / iqr
                    else:
                        normalized_epochs[trial, :, ch] = data
        else:
            raise ValueError(f"Unknown normalization method: {method}")
            
        return normalized_epochs
    
    @staticmethod
    def remove_artifacts(
        epochs: np.ndarray, 
        labels: np.ndarray, 
        threshold: float = 100.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove epochs with artifacts based on amplitude threshold.
        
        Args:
            epochs: EEG epochs of shape (trials, samples, channels)
            labels: Corresponding labels for each epoch
            threshold: Amplitude threshold for artifact detection (microvolts)
            
        Returns:
            clean_epochs: Epochs without artifacts
            clean_labels: Corresponding labels for clean epochs
        """
        # Find epochs with amplitudes exceeding threshold
        max_amplitudes = np.max(np.abs(epochs), axis=(1, 2))
        clean_indices = max_amplitudes < threshold
        
        clean_epochs = epochs[clean_indices]
        clean_labels = labels[clean_indices]
        
        return clean_epochs, clean_labels
    
    @staticmethod
    def select_channels(data: np.ndarray, channel_indices: list) -> np.ndarray:
        """
        Select specific channels from EEG data.
        
        Args:
            data: EEG data of shape (samples, channels) or (trials, samples, channels)
            channel_indices: List of channel indices to select
            
        Returns:
            Selected channels with shape (samples, selected_channels) or 
            (trials, samples, selected_channels)
        """
        if data.ndim == 2:  # (samples, channels)
            return data[:, channel_indices]
        elif data.ndim == 3:  # (trials, samples, channels)
            return data[:, :, channel_indices]
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}")
    
    def preprocess_continuous_data(
        self, 
        data: np.ndarray, 
        markers: np.ndarray,
        selected_channels: Optional[list] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete preprocessing pipeline for continuous EEG data.
        
        Args:
            data: Continuous EEG data of shape (samples, channels)
            markers: Event markers of shape (events, 2) - [position, class]
            selected_channels: List of channel indices to use (optional)
            
        Returns:
            preprocessed_epochs: Preprocessed epochs ready for model input
            labels: Corresponding class labels
        """
        self.logger.info("Starting complete preprocessing pipeline...")
        
        # 1. Select channels if specified
        if selected_channels is not None:
            data = self.select_channels(data, selected_channels)
            self.logger.info(f"Selected {len(selected_channels)} channels")
        
        # 2. Apply bandpass filter
        filtered_data = self.apply_bandpass_filter(
            data,
            self.preprocessing_config['bandpass']['low_freq'],
            self.preprocessing_config['bandpass']['high_freq'],
            self.sampling_rate,
            self.preprocessing_config['bandpass']['filter_order']
        )
        self.logger.info("Applied bandpass filter")
        
        # 3. Apply CAR reference
        if self.preprocessing_config['reference'] == "CAR":
            referenced_data = self.apply_car_reference(filtered_data)
            self.logger.info("Applied CAR reference")
        else:
            referenced_data = filtered_data
        
        # 4. Extract epochs
        epochs, labels = self.extract_epochs(
            referenced_data,
            markers,
            self.config['eeg']['epoch_length'],
            self.sampling_rate
        )
        self.logger.info(f"Extracted {len(epochs)} epochs")
        
        # 5. Remove artifacts
        clean_epochs, clean_labels = self.remove_artifacts(
            epochs,
            labels,
            self.preprocessing_config['artifact_threshold']
        )
        self.logger.info(f"Removed artifacts, {len(clean_epochs)} epochs remaining")
        
        # 6. Normalize epochs
        normalized_epochs = self.normalize_epochs(
            clean_epochs,
            self.preprocessing_config['normalization']
        )
        self.logger.info("Applied normalization")
        
        self.logger.info("Preprocessing pipeline completed successfully")
        
        return normalized_epochs, clean_labels
    
    def preprocess_single_epoch(
        self, 
        epoch: np.ndarray,
        selected_channels: Optional[list] = None
    ) -> np.ndarray:
        """
        Preprocess a single epoch for real-time classification.
        
        Args:
            epoch: Single epoch of shape (samples, channels)
            selected_channels: List of channel indices to use (optional)
            
        Returns:
            Preprocessed epoch ready for model input
        """
        # 1. Select channels if specified
        if selected_channels is not None:
            epoch = self.select_channels(epoch, selected_channels)
        
        # 2. Apply bandpass filter
        filtered_epoch = self.apply_bandpass_filter(
            epoch,
            self.preprocessing_config['bandpass']['low_freq'],
            self.preprocessing_config['bandpass']['high_freq'],
            self.sampling_rate,
            self.preprocessing_config['bandpass']['filter_order']
        )
        
        # 3. Apply CAR reference
        if self.preprocessing_config['reference'] == "CAR":
            referenced_epoch = self.apply_car_reference(filtered_epoch)
        else:
            referenced_epoch = filtered_epoch
        
        # 4. Normalize epoch
        # Add trial dimension for normalization
        epoch_with_trial_dim = referenced_epoch[np.newaxis, :, :]
        normalized_epoch = self.normalize_epochs(
            epoch_with_trial_dim,
            self.preprocessing_config['normalization']
        )
        
        # Remove trial dimension
        return normalized_epoch[0]


def test_preprocessing():
    """Test function for preprocessing module."""
    print("Testing preprocessing module...")
    
    # Create synthetic data
    sampling_rate = 256
    n_samples = 1024
    n_channels = 20
    
    # Generate synthetic EEG data
    data = np.random.randn(n_samples, n_channels) * 10
    
    # Generate synthetic markers
    markers = np.array([
        [256, 1],    # Event at 1 second, class 1
        [512, -1],   # Event at 2 seconds, class -1
        [768, 1],    # Event at 3 seconds, class 1
    ])
    
    # Initialize preprocessor
    preprocessor = PreProcessor()
    
    # Test complete preprocessing pipeline
    epochs, labels = preprocessor.preprocess_continuous_data(data, markers)
    
    print(f"Input data shape: {data.shape}")
    print(f"Output epochs shape: {epochs.shape}")
    print(f"Output labels: {labels}")
    print(f"Labels shape: {labels.shape}")
    
    # Test single epoch preprocessing
    single_epoch = data[256:512, :]  # 1 second of data
    processed_epoch = preprocessor.preprocess_single_epoch(single_epoch)
    
    print(f"Single epoch input shape: {single_epoch.shape}")
    print(f"Single epoch output shape: {processed_epoch.shape}")
    
    print("Preprocessing module test completed successfully!")


if __name__ == "__main__":
    test_preprocessing()