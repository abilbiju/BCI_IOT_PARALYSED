"""
Enhanced BCI Stream Processor for 4-Class Motor Imagery
Real-time streaming capabilities for BCI Competition III Dataset 3a

This module provides advanced streaming capabilities for real-time motor imagery
classification with proper buffer management, event-based epoching, and 
subject-specific adaptation.
"""

import numpy as np
import yaml
import logging
import threading
import time
import queue
from typing import Tuple, Dict, Optional, Callable, List
from collections import deque
import warnings

from preprocessing import PreProcessor
from eegnet_model import EEGNet
from gdf_data_loader import GDFDataLoader


class MotorImageryStreamer:
    """
    Real-time motor imagery classification streamer with advanced features.
    
    Features:
    - Continuous EEG data buffering
    - Event-based epoch extraction
    - Real-time classification
    - Adaptive thresholding
    - Performance monitoring
    - Subject-specific calibration
    """
    
    def __init__(self, config_path: str = "config.yaml", model_path: Optional[str] = None):
        """
        Initialize the motor imagery streamer.
        
        Args:
            config_path: Path to configuration YAML file
            model_path: Path to trained model (uses config if None)
        """
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format']
        )
        self.logger = logging.getLogger(__name__)
        
        # EEG Parameters
        self.sampling_rate = self.config['eeg']['sampling_rate']  # 250 Hz
        self.channels = self.config['eeg']['channels']  # 60 channels
        self.motor_imagery_channels = self.config['eeg']['motor_imagery_channels']
        self.epoch_length = self.config['eeg']['epoch_length']  # 4.0 seconds
        self.baseline_length = self.config['eeg']['baseline_length']  # 0.5 seconds
        
        # Motor imagery configuration
        self.mi_config = self.config['eeg']['motor_imagery']
        self.class_names = self.mi_config['class_names']
        self.command_mapping = self.config['commands']['class_mapping']
        
        # Buffer parameters
        buffer_duration = 10.0  # seconds of data to keep in buffer
        self.buffer_size = int(buffer_duration * self.sampling_rate)
        self.epoch_samples = int(self.epoch_length * self.sampling_rate)
        self.baseline_samples = int(self.baseline_length * self.sampling_rate)
        
        # Initialize components
        self.preprocessor = PreProcessor(config_path)
        self.eegnet = EEGNet(config_path)
        
        # Data buffers (thread-safe)
        self.data_buffer = deque(maxlen=self.buffer_size)  # Ring buffer for EEG data
        self.buffer_lock = threading.Lock()
        self.data_queue = queue.Queue()  # Queue for incoming data
        
        # Event detection
        self.event_queue = queue.Queue()  # Queue for detected events
        self.last_event_time = 0
        self.min_event_interval = 2.0  # Minimum seconds between events
        
        # Classification parameters
        self.confidence_threshold = 0.6  # Minimum confidence for valid classification
        self.adaptive_threshold = True
        self.recent_confidences = deque(maxlen=100)
        
        # Performance tracking
        self.classification_count = 0
        self.processing_times = deque(maxlen=100)
        self.classification_history = deque(maxlen=1000)
        
        # Streaming control
        self.streaming = False
        self.processing_thread = None
        self.classification_callback = None
        
        # Load model
        self.model_loaded = self.load_model(model_path)
        
        self.logger.info("Motor Imagery Streamer initialized successfully")
        self.logger.info(f"Buffer size: {self.buffer_size} samples ({buffer_duration}s)")
        self.logger.info(f"Epoch length: {self.epoch_samples} samples ({self.epoch_length}s)")
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load trained EEGNet model for classification.
        
        Args:
            model_path: Path to model file
            
        Returns:
            True if model loaded successfully
        """
        try:
            if model_path is None:
                import os
                models_dir = self.config['paths']['model_save_path']
                model_path = os.path.join(models_dir, self.config['paths']['model_filename'])
            
            if not os.path.exists(model_path):
                self.logger.warning(f"Model file not found: {model_path}")
                return False
            
            self.eegnet.load_model(model_path)
            self.logger.info(f"Model loaded successfully from: {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
    
    def add_data_chunk(self, chunk: np.ndarray) -> None:
        """
        Add new EEG data chunk to the buffer.
        
        Args:
            chunk: EEG data chunk of shape (samples, channels)
        """
        if chunk.shape[1] != self.channels:
            self.logger.warning(f"Expected {self.channels} channels, got {chunk.shape[1]}")
        
        # Add to thread-safe queue
        self.data_queue.put(chunk)
    
    def _process_data_buffer(self) -> None:
        """Internal method to process incoming data (runs in separate thread)."""
        while self.streaming:
            try:
                # Get new data chunk (with timeout)
                chunk = self.data_queue.get(timeout=0.1)
                
                # Add to ring buffer (thread-safe)
                with self.buffer_lock:
                    for sample in chunk:
                        self.data_buffer.append(sample)
                
                # Check for motor imagery events
                self._detect_motor_imagery_events()
                
            except queue.Empty:
                continue  # No new data, continue monitoring
            except Exception as e:
                self.logger.error(f"Error processing data buffer: {e}")
    
    def _detect_motor_imagery_events(self) -> None:
        """
        Detect potential motor imagery events in the buffer.
        
        This is a simplified event detection - in real applications,
        you would use proper trigger signals or advanced detection algorithms.
        """
        current_time = time.time()
        
        # Simple event detection based on time intervals
        if current_time - self.last_event_time > self.min_event_interval:
            
            with self.buffer_lock:
                if len(self.data_buffer) >= self.epoch_samples:
                    # Simulate event detection (in real system, use proper triggers)
                    # Extract most recent epoch for classification
                    recent_data = list(self.data_buffer)[-self.epoch_samples:]
                    epoch = np.array(recent_data)
                    
                    # Add event to processing queue
                    self.event_queue.put({
                        'timestamp': current_time,
                        'epoch': epoch,
                        'type': 'motor_imagery'
                    })
                    
                    self.last_event_time = current_time
    
    def _process_events(self) -> None:
        """Process detected motor imagery events (runs in main thread)."""
        while not self.event_queue.empty():
            try:
                event = self.event_queue.get_nowait()
                
                # Process the motor imagery event
                result = self._classify_motor_imagery_epoch(event['epoch'])
                
                # Add timestamp to result
                result['timestamp'] = event['timestamp']
                
                # Store in history
                self.classification_history.append(result)
                
                # Call callback if provided
                if self.classification_callback:
                    self.classification_callback(result)
                
                self.classification_count += 1
                
            except queue.Empty:
                break
            except Exception as e:
                self.logger.error(f"Error processing event: {e}")
    
    def _classify_motor_imagery_epoch(self, epoch: np.ndarray) -> Dict:
        """
        Classify a motor imagery epoch.
        
        Args:
            epoch: EEG epoch of shape (samples, channels)
            
        Returns:
            Classification result dictionary
        """
        start_time = time.time()
        
        try:
            # Select motor imagery channels
            if epoch.shape[1] >= len(self.motor_imagery_channels):
                epoch_channels = epoch[:, self.motor_imagery_channels]
            else:
                epoch_channels = epoch
            
            # Preprocess epoch
            processed_epoch = self.preprocessor.preprocess_single_epoch(epoch_channels)
            
            # Reshape for EEGNet
            processed_epoch = processed_epoch.transpose(1, 0)  # (channels, samples)
            
            # Classify
            if self.model_loaded:
                probabilities, predicted_class = self.eegnet.predict_single(processed_epoch)
                confidence = np.max(probabilities)
                
                # Adaptive thresholding
                if self.adaptive_threshold:
                    self.recent_confidences.append(confidence)
                    if len(self.recent_confidences) > 10:
                        # Adjust threshold based on recent performance
                        recent_mean = np.mean(self.recent_confidences)
                        self.confidence_threshold = max(0.4, min(0.8, recent_mean * 0.8))
                
                # Determine if classification is reliable
                is_reliable = confidence >= self.confidence_threshold
                
                # Map to command
                if is_reliable:
                    command = self.command_mapping.get(predicted_class, "UNKNOWN")
                    class_name = self.class_names[predicted_class] if predicted_class < len(self.class_names) else "unknown"
                else:
                    command = "NO_COMMAND"
                    class_name = "uncertain"
                    
            else:
                probabilities = np.zeros(4)
                predicted_class = -1
                confidence = 0.0
                is_reliable = False
                command = "NO_MODEL"
                class_name = "no_model"
            
            # Calculate processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # Create result
            result = {
                'predicted_class': int(predicted_class),
                'class_name': class_name,
                'command': command,
                'confidence': float(confidence),
                'probabilities': probabilities.tolist() if hasattr(probabilities, 'tolist') else probabilities,
                'is_reliable': is_reliable,
                'processing_time': processing_time,
                'confidence_threshold': self.confidence_threshold
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in classification: {e}")
            return {
                'predicted_class': -1,
                'class_name': 'error',
                'command': 'ERROR',
                'confidence': 0.0,
                'probabilities': [0.0, 0.0, 0.0, 0.0],
                'is_reliable': False,
                'processing_time': time.time() - start_time,
                'error': str(e)
            }
    
    def start_streaming(self, callback: Optional[Callable] = None) -> None:
        """
        Start real-time streaming and classification.
        
        Args:
            callback: Function to call with classification results
        """
        if self.streaming:
            self.logger.warning("Streaming already active")
            return
        
        self.classification_callback = callback
        self.streaming = True
        
        # Start background data processing thread
        self.processing_thread = threading.Thread(target=self._process_data_buffer)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        self.logger.info("Motor imagery streaming started")
        self.logger.info(f"Confidence threshold: {self.confidence_threshold}")
    
    def stop_streaming(self) -> None:
        """Stop real-time streaming."""
        self.streaming = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
        
        self.logger.info("Motor imagery streaming stopped")
    
    def process_frame(self) -> List[Dict]:
        """
        Process one frame of events (call this regularly in your main loop).
        
        Returns:
            List of new classification results
        """
        results = []
        
        # Process any pending events
        self._process_events()
        
        # Return recent results
        while len(self.classification_history) > 0:
            results.append(self.classification_history.popleft())
        
        return results
    
    def get_performance_stats(self) -> Dict:
        """Get current performance statistics."""
        if len(self.processing_times) > 0:
            avg_processing_time = np.mean(self.processing_times)
            max_processing_time = np.max(self.processing_times)
        else:
            avg_processing_time = 0.0
            max_processing_time = 0.0
        
        if len(self.recent_confidences) > 0:
            avg_confidence = np.mean(self.recent_confidences)
        else:
            avg_confidence = 0.0
        
        return {
            'total_classifications': self.classification_count,
            'buffer_size': len(self.data_buffer),
            'buffer_utilization': len(self.data_buffer) / self.buffer_size * 100,
            'avg_processing_time': avg_processing_time,
            'max_processing_time': max_processing_time,
            'avg_confidence': avg_confidence,
            'confidence_threshold': self.confidence_threshold,
            'streaming_active': self.streaming,
            'model_loaded': self.model_loaded
        }
    
    def calibrate_subject(self, calibration_data: np.ndarray, calibration_labels: np.ndarray) -> Dict:
        """
        Perform subject-specific calibration to adapt thresholds.
        
        Args:
            calibration_data: Calibration epochs of shape (trials, samples, channels)
            calibration_labels: Calibration labels
            
        Returns:
            Calibration results
        """
        self.logger.info("Starting subject-specific calibration...")
        
        if not self.model_loaded:
            raise ValueError("No model loaded for calibration")
        
        confidences = []
        accuracies_per_class = {i: [] for i in range(4)}
        
        # Process calibration data
        for i, (epoch, true_label) in enumerate(zip(calibration_data, calibration_labels)):
            
            # Select channels
            if epoch.shape[1] >= len(self.motor_imagery_channels):
                epoch_channels = epoch[:, self.motor_imagery_channels]
            else:
                epoch_channels = epoch
            
            # Preprocess
            processed_epoch = self.preprocessor.preprocess_single_epoch(epoch_channels)
            processed_epoch = processed_epoch.transpose(1, 0)
            
            # Classify
            probabilities, predicted_class = self.eegnet.predict_single(processed_epoch)
            confidence = np.max(probabilities)
            
            confidences.append(confidence)
            
            # Track accuracy per class
            is_correct = (predicted_class == true_label)
            accuracies_per_class[true_label].append(is_correct)
        
        # Calculate optimal threshold
        confidences = np.array(confidences)
        
        # Use percentile-based thresholding
        optimal_threshold = np.percentile(confidences, 25)  # 25th percentile
        optimal_threshold = max(0.3, min(0.8, optimal_threshold))  # Clamp to reasonable range
        
        self.confidence_threshold = optimal_threshold
        
        # Calculate calibration stats
        overall_accuracy = np.mean([predicted_class == true_label for predicted_class, true_label in 
                                   zip(np.argmax([self.eegnet.predict_single(epoch.transpose(1, 0))[1] 
                                                  for epoch in calibration_data[:, :, self.motor_imagery_channels]]), 
                                       calibration_labels)])
        
        class_accuracies = {}
        for class_idx in range(4):
            if len(accuracies_per_class[class_idx]) > 0:
                class_accuracies[self.class_names[class_idx]] = np.mean(accuracies_per_class[class_idx])
            else:
                class_accuracies[self.class_names[class_idx]] = 0.0
        
        calibration_results = {
            'optimal_threshold': optimal_threshold,
            'overall_accuracy': overall_accuracy,
            'class_accuracies': class_accuracies,
            'mean_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences),
            'calibration_trials': len(calibration_data)
        }
        
        self.logger.info(f"Calibration completed: threshold={optimal_threshold:.3f}, accuracy={overall_accuracy:.3f}")
        
        return calibration_results


def test_motor_imagery_streamer():
    """Test the motor imagery streamer with simulated data."""
    print("Testing Motor Imagery Streamer...")
    
    try:
        # Initialize streamer
        streamer = MotorImageryStreamer()
        
        # Create synthetic EEG data
        sampling_rate = 250
        channels = 60
        duration = 5.0  # seconds
        samples = int(duration * sampling_rate)
        
        synthetic_eeg = np.random.randn(samples, channels) * 10
        
        # Test data processing
        print("Testing data buffering...")
        
        # Add data in chunks
        chunk_size = 125  # 0.5 second chunks
        for i in range(0, samples, chunk_size):
            chunk = synthetic_eeg[i:i+chunk_size]
            streamer.add_data_chunk(chunk)
        
        # Start streaming (with dummy callback)
        def classification_callback(result):
            print(f"Classification: {result['class_name']} (confidence: {result['confidence']:.3f})")
        
        streamer.start_streaming(callback=classification_callback)
        
        # Simulate real-time processing
        print("Simulating real-time processing...")
        for _ in range(10):
            results = streamer.process_frame()
            if results:
                print(f"Processed {len(results)} classifications")
            time.sleep(0.1)
        
        # Stop streaming
        streamer.stop_streaming()
        
        # Get performance stats
        stats = streamer.get_performance_stats()
        print(f"Performance stats: {stats}")
        
        print("Motor Imagery Streamer test completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")


if __name__ == "__main__":
    test_motor_imagery_streamer()