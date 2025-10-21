"""
EEG Data Simulation Module for BCI Motor Imagery Classification
Phase 1: Core BCI Application (Intent Layer)

This module provides a simulation layer that loads pre-recorded EEG data
and streams it in real-time chunks, mimicking a live BCI system like OpenBCI.
"""

import numpy as np
import yaml
import logging
import time
import threading
from typing import Tuple, Optional, Generator, Callable
from queue import Queue, Empty
import random
from data_loader import BCIDataLoader


class SimEEGStreamer:
    """
    Simulates real-time EEG streaming using pre-recorded data.
    
    This class loads BCI data and provides it in chunks that simulate
    real-time acquisition from an EEG headset.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize EEG streamer with configuration.
        
        Args:
            config_path: Path to configuration YAML file
        """
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.simulation_config = self.config['simulation']
        self.eeg_config = self.config['eeg']
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format']
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize data loader
        self.data_loader = BCIDataLoader(config_path)
        
        # Streaming state
        self.is_streaming = False
        self.stream_thread = None
        self.data_queue = Queue()
        
        # Loaded data
        self.eeg_data = None
        self.markers = None
        self.current_position = 0
        
        # Streaming parameters
        self.chunk_size = self.simulation_config['chunk_size']
        self.stream_delay = self.simulation_config['stream_delay']
        self.noise_level = self.simulation_config['noise_level']
        self.sampling_rate = self.eeg_config['sampling_rate']
        self.motor_imagery_channels = self.eeg_config['motor_imagery_channels']
    
    def load_simulation_data(self, subject_filter: Optional[str] = None) -> None:
        """
        Load data for simulation from the preprocessed dataset.
        
        Args:
            subject_filter: Optional subject ID to filter (e.g., 'ds1a')
        """
        self.logger.info("Loading data for simulation...")
        
        # Load raw data
        data_dict = self.data_loader.load_npz_data()
        eeg_files = data_dict['eeg_files']
        marker_files = data_dict['marker_files']
        
        # Match EEG files with marker files
        matched_pairs = self.data_loader.match_eeg_and_markers(eeg_files, marker_files)
        
        if not matched_pairs:
            raise ValueError("No valid EEG-marker pairs found for simulation")
        
        # Select data for simulation
        if subject_filter:
            # Filter for specific subject
            selected_pair = None
            for eeg_data, markers, subject_id in matched_pairs:
                if subject_filter in subject_id:
                    selected_pair = (eeg_data, markers, subject_id)
                    break
            
            if selected_pair is None:
                raise ValueError(f"Subject '{subject_filter}' not found")
                
            self.eeg_data, self.markers, subject_id = selected_pair
            self.logger.info(f"Loaded data for subject: {subject_id}")
        else:
            # Use first available subject
            self.eeg_data, self.markers, subject_id = matched_pairs[0]
            self.logger.info(f"Loaded data for subject: {subject_id} (first available)")
        
        # Select motor imagery channels
        if len(self.motor_imagery_channels) < self.eeg_data.shape[1]:
            self.eeg_data = self.eeg_data[:, self.motor_imagery_channels]
            self.logger.info(f"Selected {len(self.motor_imagery_channels)} motor imagery channels")
        
        self.logger.info(f"Simulation data loaded: {self.eeg_data.shape}")
        self.logger.info(f"Available markers: {len(self.markers)}")
        
        # Reset position
        self.current_position = 0
    
    def add_noise(self, data: np.ndarray) -> np.ndarray:
        """
        Add realistic noise to EEG data.
        
        Args:
            data: Clean EEG data
            
        Returns:
            EEG data with added noise
        """
        if self.noise_level <= 0:
            return data
        
        # Add Gaussian noise
        noise = np.random.normal(0, self.noise_level, data.shape)
        
        # Add occasional "blink" artifacts (larger amplitude)
        if random.random() < 0.05:  # 5% chance of blink artifact
            blink_channels = random.sample(range(data.shape[1]), k=min(3, data.shape[1]))
            for ch in blink_channels:
                blink_amplitude = random.uniform(20, 50)  # Microvolts
                blink_duration = random.randint(10, 30)  # Samples
                start_idx = random.randint(0, max(1, data.shape[0] - blink_duration))
                noise[start_idx:start_idx+blink_duration, ch] += blink_amplitude
        
        return data + noise
    
    def get_chunk(self, apply_noise: bool = True) -> Tuple[np.ndarray, Optional[int]]:
        """
        Get the next chunk of EEG data.
        
        Args:
            apply_noise: Whether to add simulated noise
            
        Returns:
            data_chunk: EEG data chunk of shape (chunk_size, channels)
            marker: Event marker if present in this chunk (None otherwise)
        """
        if self.eeg_data is None:
            raise ValueError("No data loaded. Call load_simulation_data() first.")
        
        # Check if we've reached the end
        if self.current_position >= self.eeg_data.shape[0] - self.chunk_size:
            self.current_position = 0  # Loop back to beginning
            self.logger.info("Reached end of data, looping back to beginning")
        
        # Extract chunk
        end_position = self.current_position + self.chunk_size
        data_chunk = self.eeg_data[self.current_position:end_position].copy()
        
        # Check for markers in this chunk
        marker = None
        for marker_data in self.markers:
            marker_position = int(marker_data[0])
            if self.current_position <= marker_position < end_position:
                marker = int(marker_data[1])
                break
        
        # Add noise if requested
        if apply_noise:
            data_chunk = self.add_noise(data_chunk)
        
        # Update position
        self.current_position = end_position
        
        return data_chunk, marker
    
    def start_streaming(self, callback: Optional[Callable] = None) -> None:
        """
        Start streaming EEG data in a separate thread.
        
        Args:
            callback: Optional callback function to call with each chunk
        """
        if self.is_streaming:
            self.logger.warning("Streaming already active")
            return
        
        if self.eeg_data is None:
            raise ValueError("No data loaded. Call load_simulation_data() first.")
        
        self.is_streaming = True
        self.logger.info("Starting EEG streaming...")
        
        def stream_worker():
            """Worker function for streaming thread."""
            while self.is_streaming:
                try:
                    # Get next chunk
                    data_chunk, marker = self.get_chunk()
                    
                    # Create timestamp
                    timestamp = time.time()
                    
                    # Package data
                    stream_data = {
                        'data': data_chunk,
                        'marker': marker,
                        'timestamp': timestamp,
                        'position': self.current_position - self.chunk_size
                    }
                    
                    # Put in queue
                    self.data_queue.put(stream_data)
                    
                    # Call callback if provided
                    if callback:
                        callback(stream_data)
                    
                    # Sleep to simulate real-time streaming
                    time.sleep(self.stream_delay)
                    
                except Exception as e:
                    self.logger.error(f"Error in streaming: {e}")
                    break
            
            self.logger.info("Streaming stopped")
        
        # Start streaming thread
        self.stream_thread = threading.Thread(target=stream_worker, daemon=True)
        self.stream_thread.start()
    
    def stop_streaming(self) -> None:
        """Stop EEG streaming."""
        if not self.is_streaming:
            return
        
        self.is_streaming = False
        self.logger.info("Stopping EEG streaming...")
        
        if self.stream_thread:
            self.stream_thread.join(timeout=2.0)
    
    def get_stream_data(self, timeout: float = 1.0) -> Optional[dict]:
        """
        Get the next data chunk from the stream queue.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Stream data dictionary or None if timeout
        """
        try:
            return self.data_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def stream_generator(
        self, 
        max_chunks: Optional[int] = None
    ) -> Generator[dict, None, None]:
        """
        Generator that yields stream data chunks.
        
        Args:
            max_chunks: Maximum number of chunks to yield (None for infinite)
            
        Yields:
            Stream data dictionaries
        """
        if self.eeg_data is None:
            raise ValueError("No data loaded. Call load_simulation_data() first.")
        
        chunk_count = 0
        
        while max_chunks is None or chunk_count < max_chunks:
            # Get next chunk
            data_chunk, marker = self.get_chunk()
            
            # Create timestamp
            timestamp = time.time()
            
            # Package data
            stream_data = {
                'data': data_chunk,
                'marker': marker,
                'timestamp': timestamp,
                'position': self.current_position - self.chunk_size,
                'chunk_id': chunk_count
            }
            
            yield stream_data
            
            chunk_count += 1
            
            # Sleep to simulate real-time streaming
            time.sleep(self.stream_delay)
    
    def get_random_epoch(self, target_class: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """
        Get a random epoch around a marker event.
        
        Args:
            target_class: Specific class to target (None for random)
            
        Returns:
            epoch_data: EEG epoch of shape (samples, channels)
            class_label: Class label for the epoch
        """
        if self.eeg_data is None or self.markers is None:
            raise ValueError("No data loaded. Call load_simulation_data() first.")
        
        # Filter markers by target class if specified
        if target_class is not None:
            valid_markers = self.markers[self.markers[:, 1] == target_class]
        else:
            valid_markers = self.markers
        
        if len(valid_markers) == 0:
            raise ValueError(f"No markers found for target class: {target_class}")
        
        # Select random marker
        marker_idx = random.randint(0, len(valid_markers) - 1)
        selected_marker = valid_markers[marker_idx]
        
        marker_position = int(selected_marker[0])
        class_label = int(selected_marker[1])
        
        # Extract epoch around marker
        epoch_samples = int(self.eeg_config['epoch_length'] * self.sampling_rate)
        start_idx = marker_position
        end_idx = start_idx + epoch_samples
        
        # Ensure we don't go out of bounds
        if end_idx > self.eeg_data.shape[0]:
            start_idx = self.eeg_data.shape[0] - epoch_samples
            end_idx = self.eeg_data.shape[0]
        
        epoch_data = self.eeg_data[start_idx:end_idx].copy()
        
        # Add noise
        epoch_data = self.add_noise(epoch_data)
        
        return epoch_data, class_label
    
    def get_simulation_info(self) -> dict:
        """
        Get information about the simulation setup.
        
        Returns:
            Dictionary containing simulation information
        """
        if self.eeg_data is None:
            return {"status": "No data loaded"}
        
        return {
            "data_shape": self.eeg_data.shape,
            "total_samples": self.eeg_data.shape[0],
            "channels": self.eeg_data.shape[1],
            "duration_seconds": self.eeg_data.shape[0] / self.sampling_rate,
            "chunk_size": self.chunk_size,
            "stream_delay": self.stream_delay,
            "sampling_rate": self.sampling_rate,
            "noise_level": self.noise_level,
            "total_markers": len(self.markers) if self.markers is not None else 0,
            "is_streaming": self.is_streaming,
            "current_position": self.current_position
        }


def test_simulation():
    """Test function for simulation module."""
    print("Testing EEG simulation module...")
    
    # Initialize streamer
    streamer = SimEEGStreamer()
    
    try:
        # Load simulation data
        streamer.load_simulation_data()
        
        # Get simulation info
        info = streamer.get_simulation_info()
        print(f"Simulation info: {info}")
        
        # Test getting chunks
        print("\nTesting chunk retrieval:")
        for i in range(3):
            chunk, marker = streamer.get_chunk()
            print(f"Chunk {i+1}: shape={chunk.shape}, marker={marker}")
        
        # Test random epoch
        print("\nTesting random epoch:")
        epoch, label = streamer.get_random_epoch()
        print(f"Random epoch: shape={epoch.shape}, label={label}")
        
        # Test generator
        print("\nTesting stream generator (5 chunks):")
        for i, stream_data in enumerate(streamer.stream_generator(max_chunks=5)):
            print(f"Stream chunk {i+1}: {stream_data['data'].shape}, marker={stream_data['marker']}")
        
        print("\nSimulation module test completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        print("Make sure the bci_preprocessed_data.npz file exists and is valid")


if __name__ == "__main__":
    test_simulation()