"""
BCI Pipeline Orchestrator for Motor Imagery Classification
Updated for BCI Competition III Dataset 3a (4-class motor imagery)

This module orchestrates the complete BCI pipeline: Data → Preprocessing → 
Model Inference → Command Mapping. It serves as the master class that 
coordinates all components for real-time motor imagery classification.

Supports both batch processing and real-time streaming for 4-class motor imagery:
- Left hand, Right hand, Foot, Tongue movements
"""

import numpy as np
import yaml
import logging
import os
from typing import Tuple, Dict, Optional, Union, List
import time

from preprocessing import PreProcessor
from eegnet_model import EEGNet
from sim_generator import SimEEGStreamer


class BCIPipeline:
    """
    Master BCI Pipeline class that orchestrates the complete data flow.
    
    This class coordinates:
    1. Data preprocessing
    2. Model inference 
    3. Command mapping
    4. Result interpretation
    """
    
    def __init__(self, config_path: str = "config.yaml", model_path: Optional[str] = None):
        """
        Initialize BCI Pipeline with all components.
        
        Args:
            config_path: Path to configuration YAML file
            model_path: Path to trained model (uses config path if None)
        """
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format']
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.preprocessor = PreProcessor(config_path)
        self.eegnet = EEGNet(config_path)
        self.streamer = SimEEGStreamer(config_path)
        
        # Load trained model
        self.model_loaded = False
        self.load_model(model_path)
        
        # Configuration shortcuts
        self.motor_imagery_channels = self.config['eeg']['motor_imagery_channels']
        self.command_mapping = self.config['commands']['class_mapping']
        self.sampling_rate = self.config['eeg']['sampling_rate']
        self.epoch_length = self.config['eeg']['epoch_length']
        
        # Performance tracking
        self.classification_history = []
        self.performance_stats = {
            'total_classifications': 0,
            'processing_times': [],
            'confidence_scores': []
        }
        
        self.logger.info("BCI Pipeline initialized successfully")
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load the trained EEGNet model.
        
        Args:
            model_path: Path to model file (uses config path if None)
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if model_path is None:
                models_dir = self.config['paths']['model_save_path']
                model_path = os.path.join(models_dir, self.config['paths']['model_filename'])
            
            # Check if model file exists
            if not os.path.exists(model_path):
                self.logger.warning(f"Model file not found: {model_path}")
                self.logger.info("Pipeline will work but requires a trained model for classification")
                return False
            
            # Load model
            self.eegnet.load_model(model_path)
            self.model_loaded = True
            self.logger.info(f"Model loaded successfully from: {model_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            self.model_loaded = False
            return False
    
    def preprocess_epoch(self, raw_epoch: np.ndarray) -> np.ndarray:
        """
        Preprocess a single epoch for classification.
        
        Args:
            raw_epoch: Raw EEG epoch of shape (samples, channels)
            
        Returns:
            Preprocessed epoch ready for model input
        """
        # Select motor imagery channels
        if raw_epoch.shape[1] > len(self.motor_imagery_channels):
            epoch = raw_epoch[:, self.motor_imagery_channels]
        else:
            epoch = raw_epoch
        
        # Apply preprocessing pipeline
        processed_epoch = self.preprocessor.preprocess_single_epoch(epoch)
        
        # Reshape for EEGNet: (samples, channels) -> (1, channels, samples, 1)
        # EEGNet expects (batch, channels, time, 1)
        processed_epoch = processed_epoch.transpose(1, 0)  # (channels, samples)
        processed_epoch = processed_epoch[np.newaxis, :, :, np.newaxis]  # (1, channels, samples, 1)
        
        return processed_epoch
    
    def classify_epoch(self, processed_epoch: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        Classify a preprocessed epoch and map to command.
        
        Args:
            processed_epoch: Preprocessed EEG epoch
            
        Returns:
            command: Mapped command string
            confidence: Confidence score (0-1)
            probabilities: Raw class probabilities
        """
        if not self.model_loaded:
            raise ValueError("No trained model loaded. Cannot perform classification.")
        
        # Get model predictions
        probabilities, predicted_class = self.eegnet.predict_single(processed_epoch)
        
        # Calculate confidence (maximum probability)
        confidence = np.max(probabilities)
        
        # Map class to command
        command = self.command_mapping.get(predicted_class, "UNKNOWN")
        
        return command, confidence, probabilities
    
    def process_single_epoch(
        self, 
        raw_epoch: np.ndarray,
        log_result: bool = True
    ) -> Dict[str, Union[str, float, np.ndarray]]:
        """
        Complete processing pipeline for a single epoch.
        
        Args:
            raw_epoch: Raw EEG epoch of shape (samples, channels)
            log_result: Whether to log the classification result
            
        Returns:
            Dictionary containing classification results
        """
        start_time = time.time()
        
        try:
            # 1. Preprocess epoch
            processed_epoch = self.preprocess_epoch(raw_epoch)
            
            # 2. Classify epoch
            command, confidence, probabilities = self.classify_epoch(processed_epoch)
            
            # 3. Calculate processing time
            processing_time = time.time() - start_time
            
            # 4. Create result dictionary
            result = {
                'command': command,
                'confidence': confidence,
                'probabilities': probabilities.tolist(),
                'processing_time': processing_time,
                'timestamp': time.time(),
                'status': 'success'
            }
            
            # 5. Update performance tracking
            self.performance_stats['total_classifications'] += 1
            self.performance_stats['processing_times'].append(processing_time)
            self.performance_stats['confidence_scores'].append(confidence)
            
            # 6. Store in history
            if log_result:
                self.classification_history.append(result)
                # Keep only last 100 classifications
                if len(self.classification_history) > 100:
                    self.classification_history.pop(0)
            
            # 7. Log result
            if log_result:
                self.logger.info(
                    f"Classification: {command} (confidence: {confidence:.3f}, "
                    f"time: {processing_time:.3f}s)"
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in processing pipeline: {e}")
            return {
                'command': 'ERROR',
                'confidence': 0.0,
                'probabilities': [],
                'processing_time': time.time() - start_time,
                'timestamp': time.time(),
                'status': 'error',
                'error_message': str(e)
            }
    
    def process_stream_data(self, stream_data: Dict) -> Dict:
        """
        Process data from the simulation stream.
        
        Args:
            stream_data: Stream data dictionary from SimEEGStreamer
            
        Returns:
            Classification result dictionary
        """
        raw_epoch = stream_data['data']
        
        # Process the epoch
        result = self.process_single_epoch(raw_epoch, log_result=True)
        
        # Add stream metadata
        result['stream_timestamp'] = stream_data['timestamp']
        result['stream_position'] = stream_data['position']
        result['stream_marker'] = stream_data.get('marker', None)
        
        return result
    
    def run_real_time_simulation(
        self, 
        duration: Optional[float] = None,
        callback: Optional[callable] = None
    ) -> None:
        """
        Run real-time simulation using the EEG streamer.
        
        Args:
            duration: Duration in seconds (None for indefinite)
            callback: Optional callback function for each classification
        """
        if not self.model_loaded:
            raise ValueError("No trained model loaded. Cannot run simulation.")
        
        self.logger.info("Starting real-time BCI simulation...")
        
        # Load simulation data
        self.streamer.load_simulation_data()
        
        # Start streaming
        def stream_callback(stream_data):
            """Callback for each stream chunk."""
            result = self.process_stream_data(stream_data)
            
            if callback:
                callback(result)
        
        self.streamer.start_streaming(callback=stream_callback)
        
        try:
            # Run for specified duration or until interrupted
            if duration:
                time.sleep(duration)
            else:
                self.logger.info("Press Ctrl+C to stop simulation")
                while True:
                    time.sleep(1)
                    
        except KeyboardInterrupt:
            self.logger.info("Simulation interrupted by user")
        finally:
            self.streamer.stop_streaming()
            self.logger.info("Real-time simulation stopped")
    
    def test_with_random_epochs(self, num_epochs: int = 10) -> list:
        """
        Test pipeline with random epochs from the dataset.
        
        Args:
            num_epochs: Number of random epochs to test
            
        Returns:
            List of classification results
        """
        if not self.model_loaded:
            raise ValueError("No trained model loaded. Cannot perform testing.")
        
        self.logger.info(f"Testing pipeline with {num_epochs} random epochs...")
        
        # Load simulation data
        self.streamer.load_simulation_data()
        
        results = []
        
        for i in range(num_epochs):
            try:
                # Get random epoch
                epoch, true_label = self.streamer.get_random_epoch()
                
                # Process epoch
                result = self.process_single_epoch(epoch, log_result=False)
                
                # Add true label for comparison
                result['true_label'] = true_label
                result['epoch_id'] = i
                
                results.append(result)
                
                self.logger.info(
                    f"Epoch {i+1}: Predicted={result['command']}, "
                    f"True={true_label}, Confidence={result['confidence']:.3f}"
                )
                
            except Exception as e:
                self.logger.error(f"Error processing epoch {i+1}: {e}")
        
        # Calculate accuracy
        correct = sum(1 for r in results if self._compare_prediction_with_true(r))
        accuracy = correct / len(results) if results else 0
        
        self.logger.info(f"Test completed: {correct}/{len(results)} correct (accuracy: {accuracy:.3f})")
        
        return results
    
    def _compare_prediction_with_true(self, result: Dict) -> bool:
        """
        Compare predicted command with true label.
        
        Args:
            result: Classification result dictionary
            
        Returns:
            True if prediction matches true label
        """
        true_label = result.get('true_label')
        predicted_command = result.get('command')
        
        if true_label is None or predicted_command is None:
            return False
        
        # Map true label to expected command
        expected_command = self.command_mapping.get(
            0 if true_label == -1 else 1, "UNKNOWN"
        )
        
        return predicted_command == expected_command
    
    def get_performance_stats(self) -> Dict:
        """
        Get pipeline performance statistics.
        
        Returns:
            Dictionary containing performance metrics
        """
        stats = self.performance_stats.copy()
        
        if stats['processing_times']:
            stats['avg_processing_time'] = np.mean(stats['processing_times'])
            stats['max_processing_time'] = np.max(stats['processing_times'])
            stats['min_processing_time'] = np.min(stats['processing_times'])
        
        if stats['confidence_scores']:
            stats['avg_confidence'] = np.mean(stats['confidence_scores'])
            stats['min_confidence'] = np.min(stats['confidence_scores'])
        
        return stats
    
    def get_pipeline_info(self) -> Dict:
        """
        Get information about the pipeline configuration.
        
        Returns:
            Dictionary containing pipeline information
        """
        return {
            'model_loaded': self.model_loaded,
            'channels_used': len(self.motor_imagery_channels),
            'sampling_rate': self.sampling_rate,
            'epoch_length': self.epoch_length,
            'command_mapping': self.command_mapping,
            'total_classifications': self.performance_stats['total_classifications'],
            'history_length': len(self.classification_history)
        }
    
    def reset_performance_stats(self) -> None:
        """Reset performance tracking statistics."""
        self.performance_stats = {
            'total_classifications': 0,
            'processing_times': [],
            'confidence_scores': []
        }
        self.classification_history = []
        self.logger.info("Performance statistics reset")


def test_bci_pipeline():
    """Test function for BCI pipeline."""
    print("Testing BCI Pipeline...")
    
    try:
        # Initialize pipeline
        pipeline = BCIPipeline()
        
        # Get pipeline info
        info = pipeline.get_pipeline_info()
        print(f"Pipeline info: {info}")
        
        if pipeline.model_loaded:
            # Test with random epochs
            print("\nTesting with random epochs...")
            results = pipeline.test_with_random_epochs(num_epochs=5)
            
            # Show performance stats
            stats = pipeline.get_performance_stats()
            print(f"Performance stats: {stats}")
            
        else:
            print("No trained model available. Train model first for full testing.")
            
            # Test preprocessing only
            print("Testing preprocessing pipeline...")
            pipeline.streamer.load_simulation_data()
            epoch, label = pipeline.streamer.get_random_epoch()
            processed = pipeline.preprocess_epoch(epoch)
            print(f"Preprocessing test: {epoch.shape} -> {processed.shape}")
        
        print("BCI Pipeline test completed!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        print("Make sure the bci_preprocessed_data.npz file exists")


def demo_real_time_classification():
    """Demo of real-time classification."""
    print("BCI Real-Time Classification Demo")
    print("=" * 40)
    
    try:
        # Initialize pipeline
        pipeline = BCIPipeline()
        
        if not pipeline.model_loaded:
            print("No trained model available. Please train model first.")
            return
        
        # Demo callback function
        def classification_callback(result):
            print(f"🧠 Command: {result['command']} | "
                  f"Confidence: {result['confidence']:.3f} | "
                  f"Time: {result['processing_time']:.3f}s")
        
        print("Starting 10-second real-time demo...")
        print("Press Ctrl+C to stop early")
        
        # Run simulation for 10 seconds
        pipeline.run_real_time_simulation(
            duration=10.0,
            callback=classification_callback
        )
        
        # Show final stats
        stats = pipeline.get_performance_stats()
        print(f"\nDemo completed!")
        print(f"Total classifications: {stats['total_classifications']}")
        print(f"Average processing time: {stats.get('avg_processing_time', 0):.3f}s")
        print(f"Average confidence: {stats.get('avg_confidence', 0):.3f}")
        
    except Exception as e:
        print(f"Error during demo: {e}")


if __name__ == "__main__":
    # Run tests
    test_bci_pipeline()
    
    # Uncomment to run real-time demo
    # demo_real_time_classification()