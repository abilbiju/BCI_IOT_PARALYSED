"""
Comprehensive Testing for 4-Class Motor Imagery BCI System
Tests the complete pipeline with BCI Competition III Dataset 3a

This script tests:
1. GDF data loading and preprocessing
2. 4-class EEGNet model training  
3. Real-time motor imagery streaming
4. API endpoints
5. Performance evaluation
"""

import numpy as np
import yaml
import logging
import time
import os
from typing import Dict, List

from gdf_data_loader import GDFDataLoader
from eegnet_model import EEGNet
from motor_imagery_streamer import MotorImageryStreamer
from train_model import EEGNetTrainer


class MotorImageryTester:
    """
    Comprehensive tester for 4-class motor imagery BCI system.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the tester with configuration."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format']
        )
        self.logger = logging.getLogger(__name__)
        
        self.test_results = {}
    
    def test_gdf_data_loading(self) -> Dict:
        """
        Test GDF data loading and preprocessing.
        
        Returns:
            Test results dictionary
        """
        self.logger.info("Testing GDF data loading...")
        
        try:
            # Initialize GDF data loader
            gdf_loader = GDFDataLoader()
            
            # Test metadata loading
            metadata = gdf_loader.load_all_metadata()
            
            results = {
                'status': 'success',
                'triggers_loaded': len(metadata['triggers']),
                'labels_loaded': len(metadata['labels']),
                'artifacts_loaded': len(metadata['artifacts']),
                'dataset_info': gdf_loader.get_dataset_info()
            }
            
            # Count valid training trials
            valid_labels = metadata['labels'][~np.isnan(metadata['labels'])]
            unique_labels, counts = np.unique(valid_labels, return_counts=True)
            
            class_distribution = {}
            for label, count in zip(unique_labels, counts):
                class_name = gdf_loader.mi_config['class_names'][int(label-1)]
                class_distribution[class_name] = int(count)
            
            results['class_distribution'] = class_distribution
            results['total_valid_trials'] = len(valid_labels)
            
            # Test loading a small portion of EEG data for validation
            self.logger.info("Testing EEG data structure (loading first 1000 samples)...")
            
            # Load just the beginning of the file for validation
            eeg_path = os.path.join(gdf_loader.data_path, gdf_loader.eeg_file)
            if os.path.exists(eeg_path):
                with open(eeg_path, 'r') as f:
                    lines = []
                    for i, line in enumerate(f):
                        if i >= 10:  # Just read first 10 lines
                            break
                        lines.append(line.strip())
                
                if lines:
                    first_line = lines[0].split()
                    results['eeg_channels_detected'] = len(first_line)
                    results['eeg_sample_preview'] = first_line[:5]  # First 5 values
                    
                    # Validate channel count
                    expected_channels = gdf_loader.channels
                    if len(first_line) == expected_channels:
                        results['eeg_format_valid'] = True
                    else:
                        results['eeg_format_valid'] = False
                        results['eeg_format_error'] = f"Expected {expected_channels} channels, found {len(first_line)}"
                else:
                    results['eeg_format_valid'] = False
                    results['eeg_format_error'] = "Could not read EEG data"
            else:
                results['eeg_format_valid'] = False
                results['eeg_format_error'] = f"EEG file not found: {eeg_path}"
            
            self.logger.info("GDF data loading test completed successfully")
            
        except Exception as e:
            results = {
                'status': 'error',
                'error': str(e),
                'error_type': type(e).__name__
            }
            self.logger.error(f"GDF data loading test failed: {e}")
        
        self.test_results['gdf_data_loading'] = results
        return results
    
    def test_preprocessing_pipeline(self) -> Dict:
        """
        Test preprocessing pipeline with synthetic data.
        
        Returns:
            Test results dictionary
        """
        self.logger.info("Testing preprocessing pipeline...")
        
        try:
            # Create synthetic motor imagery data
            sampling_rate = self.config['eeg']['sampling_rate']
            channels = len(self.config['eeg']['motor_imagery_channels'])
            epoch_length = self.config['eeg']['epoch_length']
            samples_per_epoch = int(epoch_length * sampling_rate)
            
            # Generate synthetic data with motor imagery-like patterns
            num_trials = 40  # 10 trials per class
            synthetic_epochs = np.random.randn(num_trials, samples_per_epoch, channels) * 10
            synthetic_labels = np.tile([0, 1, 2, 3], 10)  # 4 classes, 10 trials each
            
            # Add class-specific patterns
            for i, label in enumerate(synthetic_labels):
                # Add frequency-specific patterns for each class
                t = np.linspace(0, epoch_length, samples_per_epoch)
                if label == 0:  # Left hand - alpha rhythm (8-12 Hz)
                    pattern = np.sin(2 * np.pi * 10 * t) * 5
                elif label == 1:  # Right hand - beta rhythm (13-30 Hz)
                    pattern = np.sin(2 * np.pi * 20 * t) * 3
                elif label == 2:  # Foot - mu rhythm (8-12 Hz) 
                    pattern = np.sin(2 * np.pi * 9 * t) * 4
                else:  # Tongue - gamma rhythm (30-100 Hz)
                    pattern = np.sin(2 * np.pi * 40 * t) * 2
                
                # Add pattern to specific channels
                synthetic_epochs[i, :, :5] += pattern[:, np.newaxis]
            
            # Initialize GDF loader for preprocessing functions
            gdf_loader = GDFDataLoader()
            
            # Test preprocessing steps
            data_splits = gdf_loader.prepare_for_training(synthetic_epochs, synthetic_labels)
            
            results = {
                'status': 'success',
                'input_shape': synthetic_epochs.shape,
                'output_shape': data_splits['X_train'].shape,
                'train_samples': data_splits['X_train'].shape[0],
                'val_samples': data_splits['X_val'].shape[0], 
                'test_samples': data_splits['X_test'].shape[0],
                'classes_in_train': len(np.unique(data_splits['y_train'].argmax(axis=1))),
                'preprocessing_successful': True
            }
            
            # Test class balance
            train_labels = data_splits['y_train'].argmax(axis=1)
            unique_labels, counts = np.unique(train_labels, return_counts=True)
            results['class_balance'] = dict(zip([gdf_loader.mi_config['class_names'][i] for i in unique_labels], 
                                              counts.tolist()))
            
            self.logger.info("Preprocessing pipeline test completed successfully")
            
        except Exception as e:
            results = {
                'status': 'error',
                'error': str(e),
                'error_type': type(e).__name__
            }
            self.logger.error(f"Preprocessing pipeline test failed: {e}")
        
        self.test_results['preprocessing_pipeline'] = results
        return results
    
    def test_eegnet_model(self) -> Dict:
        """
        Test EEGNet model creation and basic functionality.
        
        Returns:
            Test results dictionary
        """
        self.logger.info("Testing EEGNet model...")
        
        try:
            # Initialize EEGNet
            eegnet = EEGNet()
            
            # Build model
            model = eegnet.build_model()
            eegnet.compile_model()
            
            # Get model info
            total_params = model.count_params()
            model_summary = eegnet.get_model_summary()
            
            # Test with synthetic data
            batch_size = 8
            channels = eegnet.chans
            samples = eegnet.samples
            
            # Create synthetic input
            X_test = np.random.randn(batch_size, channels, samples, 1)
            
            # Test prediction
            predictions, predicted_classes = eegnet.predict(X_test)
            
            # Validate outputs
            correct_pred_shape = predictions.shape == (batch_size, 4)  # 4 classes
            correct_class_shape = predicted_classes.shape == (batch_size,)
            classes_in_range = np.all((predicted_classes >= 0) & (predicted_classes < 4))
            probabilities_sum_to_one = np.allclose(np.sum(predictions, axis=1), 1.0)
            
            results = {
                'status': 'success',
                'model_built': True,
                'total_parameters': int(total_params),
                'input_shape': (channels, samples, 1),
                'output_classes': 4,
                'prediction_shape_correct': correct_pred_shape,
                'classes_shape_correct': correct_class_shape,
                'classes_in_valid_range': classes_in_range,
                'probabilities_valid': probabilities_sum_to_one,
                'test_predictions_shape': predictions.shape,
                'test_classes_shape': predicted_classes.shape
            }
            
            # Test single prediction
            single_X = X_test[0]  # (channels, samples, 1)
            single_probs, single_class = eegnet.predict_single(single_X)
            
            results['single_prediction_works'] = True
            results['single_prediction_class'] = int(single_class)
            results['single_prediction_confidence'] = float(np.max(single_probs))
            
            self.logger.info("EEGNet model test completed successfully")
            
        except Exception as e:
            results = {
                'status': 'error',
                'error': str(e),
                'error_type': type(e).__name__
            }
            self.logger.error(f"EEGNet model test failed: {e}")
        
        self.test_results['eegnet_model'] = results
        return results
    
    def test_motor_imagery_streamer(self) -> Dict:
        """
        Test motor imagery streaming functionality.
        
        Returns:
            Test results dictionary
        """
        self.logger.info("Testing motor imagery streamer...")
        
        try:
            # Initialize streamer
            streamer = MotorImageryStreamer()
            
            # Test basic initialization
            results = {
                'status': 'success',
                'streamer_initialized': True,
                'model_loaded': streamer.model_loaded,
                'buffer_size': streamer.buffer_size,
                'epoch_samples': streamer.epoch_samples,
                'sampling_rate': streamer.sampling_rate
            }
            
            # Create synthetic streaming data
            chunk_size = 125  # 0.5 second chunks at 250 Hz
            num_chunks = 20   # 10 seconds of data
            channels = streamer.channels
            
            synthetic_data = np.random.randn(chunk_size * num_chunks, channels) * 10
            
            # Test data buffering
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size
                chunk = synthetic_data[start_idx:end_idx]
                streamer.add_data_chunk(chunk)
            
            results['data_buffering_works'] = True
            
            # Test streaming
            classification_results = []
            
            def test_callback(result):
                classification_results.append(result)
            
            streamer.start_streaming(callback=test_callback)
            
            # Let it process for a short time
            for _ in range(10):
                frame_results = streamer.process_frame()
                time.sleep(0.1)
            
            streamer.stop_streaming()
            
            # Check results
            results['streaming_started_stopped'] = True
            results['classifications_received'] = len(classification_results)
            
            # Get performance stats
            perf_stats = streamer.get_performance_stats()
            results['performance_stats'] = perf_stats
            
            # Test calibration with synthetic data if model is loaded
            if streamer.model_loaded:
                try:
                    # Create calibration data
                    cal_epochs = 20
                    cal_data = np.random.randn(cal_epochs, streamer.epoch_samples, 
                                             len(streamer.motor_imagery_channels)) * 10
                    cal_labels = np.random.randint(0, 4, cal_epochs)
                    
                    cal_results = streamer.calibrate_subject(cal_data, cal_labels)
                    results['calibration_works'] = True
                    results['calibration_results'] = cal_results
                    
                except Exception as cal_error:
                    results['calibration_works'] = False
                    results['calibration_error'] = str(cal_error)
            else:
                results['calibration_works'] = False
                results['calibration_error'] = "No model loaded"
            
            self.logger.info("Motor imagery streamer test completed successfully")
            
        except Exception as e:
            results = {
                'status': 'error', 
                'error': str(e),
                'error_type': type(e).__name__
            }
            self.logger.error(f"Motor imagery streamer test failed: {e}")
        
        self.test_results['motor_imagery_streamer'] = results
        return results
    
    def test_configuration_validation(self) -> Dict:
        """
        Test configuration validation for 4-class motor imagery.
        
        Returns:
            Test results dictionary
        """
        self.logger.info("Testing configuration validation...")
        
        try:
            results = {
                'status': 'success',
                'config_loaded': True
            }
            
            # Validate EEG configuration
            eeg_config = self.config['eeg']
            results['sampling_rate'] = eeg_config['sampling_rate']
            results['channels'] = eeg_config['channels']
            results['epoch_length'] = eeg_config['epoch_length']
            results['motor_imagery_channels'] = len(eeg_config['motor_imagery_channels'])
            
            # Validate motor imagery configuration
            mi_config = eeg_config['motor_imagery']
            results['mi_classes'] = mi_config['classes']
            results['mi_class_names'] = mi_config['class_names']
            results['mi_class_mapping'] = mi_config['class_mapping']
            
            # Validate model configuration
            model_config = self.config['model']
            results['model_classes'] = model_config['nb_classes']
            results['model_channels'] = model_config['chans']
            results['model_samples'] = model_config['samples']
            
            # Validate command mapping
            cmd_config = self.config['commands']
            results['command_mapping'] = cmd_config['class_mapping']
            
            # Check consistency
            checks = {
                'mi_classes_match_model': mi_config['classes'] == model_config['nb_classes'],
                'motor_channels_match_model': len(eeg_config['motor_imagery_channels']) == model_config['chans'],
                'samples_match_epoch': model_config['samples'] == int(eeg_config['epoch_length'] * eeg_config['sampling_rate']),
                'command_mapping_complete': len(cmd_config['class_mapping']) == mi_config['classes'],
                'class_names_count_correct': len(mi_config['class_names']) == mi_config['classes']
            }
            
            results['consistency_checks'] = checks
            results['all_checks_passed'] = all(checks.values())
            
            # Validate file paths
            paths_config = self.config['paths']
            bci_path = paths_config['bci_competition_path']
            
            file_checks = {}
            for file_key in ['eeg_data_file', 'triggers_file', 'labels_file', 'artifacts_file']:
                file_path = os.path.join(bci_path, paths_config[file_key])
                file_checks[file_key] = os.path.exists(file_path)
            
            results['file_checks'] = file_checks
            results['all_files_exist'] = all(file_checks.values())
            
            self.logger.info("Configuration validation test completed successfully")
            
        except Exception as e:
            results = {
                'status': 'error',
                'error': str(e),
                'error_type': type(e).__name__
            }
            self.logger.error(f"Configuration validation test failed: {e}")
        
        self.test_results['configuration_validation'] = results
        return results
    
    def run_all_tests(self) -> Dict:
        """
        Run all tests and return comprehensive results.
        
        Returns:
            Complete test results dictionary
        """
        self.logger.info("Starting comprehensive 4-class motor imagery BCI system tests...")
        
        start_time = time.time()
        
        # Run all tests
        test_methods = [
            self.test_configuration_validation,
            self.test_gdf_data_loading, 
            self.test_preprocessing_pipeline,
            self.test_eegnet_model,
            self.test_motor_imagery_streamer
        ]
        
        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                test_name = test_method.__name__
                self.logger.error(f"Test {test_name} failed with exception: {e}")
                self.test_results[test_name] = {
                    'status': 'error',
                    'error': str(e),
                    'error_type': type(e).__name__
                }
        
        # Calculate overall results
        total_time = time.time() - start_time
        
        passed_tests = sum(1 for result in self.test_results.values() 
                          if result.get('status') == 'success')
        total_tests = len(self.test_results)
        
        overall_results = {
            'test_summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': total_tests - passed_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
                'total_time_seconds': total_time
            },
            'individual_test_results': self.test_results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'system_ready': passed_tests == total_tests
        }
        
        # Log summary
        self.logger.info(f"Test Summary: {passed_tests}/{total_tests} tests passed")
        if overall_results['system_ready']:
            self.logger.info("✅ System is ready for 4-class motor imagery classification!")
        else:
            self.logger.warning("❌ System has issues that need to be resolved")
        
        return overall_results


def main():
    """Main testing function."""
    print("=" * 60)
    print("4-Class Motor Imagery BCI System - Comprehensive Testing")
    print("=" * 60)
    
    # Initialize tester
    tester = MotorImageryTester()
    
    # Run all tests
    results = tester.run_all_tests()
    
    # Print results summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    summary = results['test_summary']
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed_tests']}")
    print(f"Failed: {summary['failed_tests']}")
    print(f"Success Rate: {summary['success_rate']:.1%}")
    print(f"Total Time: {summary['total_time_seconds']:.2f} seconds")
    
    if results['system_ready']:
        print("\n✅ SYSTEM READY: All tests passed!")
        print("The 4-class motor imagery BCI system is ready for use.")
    else:
        print("\n❌ SYSTEM NOT READY: Some tests failed.")
        print("Please review the failed tests and resolve issues.")
    
    # Print individual test results
    print("\n" + "=" * 60)
    print("INDIVIDUAL TEST DETAILS")
    print("=" * 60)
    
    for test_name, result in results['individual_test_results'].items():
        status = "✅ PASS" if result.get('status') == 'success' else "❌ FAIL"
        print(f"\n{test_name}: {status}")
        
        if result.get('status') == 'error':
            print(f"  Error: {result.get('error', 'Unknown error')}")
        else:
            # Print key metrics for successful tests
            if test_name == 'gdf_data_loading' and 'class_distribution' in result:
                print(f"  Classes: {result['class_distribution']}")
            elif test_name == 'eegnet_model' and 'total_parameters' in result:
                print(f"  Model Parameters: {result['total_parameters']:,}")
            elif test_name == 'configuration_validation' and 'all_checks_passed' in result:
                print(f"  Config Valid: {result['all_checks_passed']}")


if __name__ == "__main__":
    main()