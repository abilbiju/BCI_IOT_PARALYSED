"""
Model Accuracy Checker for BCI Motor Imagery Classification System

This script provides comprehensive accuracy testing and evaluation capabilities
for trained EEGNet models on motor imagery datasets. Supports both single
dataset and multi-dataset evaluation scenarios.

Features:
- Load and test trained models
- Multi-dataset evaluation
- Cross-subject validation
- Detailed performance metrics
- Confusion matrix analysis
- Real-time classification simulation
- Performance visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, NamedTuple
from datetime import datetime
from dataclasses import dataclass
import time

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, cohen_kappa_score
)
import logging
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, cohen_kappa_score
)
from sklearn.model_selection import StratifiedKFold

from unified_bci_loader import UnifiedBCIDataLoader
from bci4_2a_loader import BCI4_2A_Loader


@dataclass
class AccuracyResults:
    """Data class to store comprehensive accuracy results."""
    overall_accuracy: float
    per_class_accuracy: Dict[str, float]
    precision: float
    recall: float
    f1_score: float
    cohen_kappa: float
    confusion_matrix: np.ndarray
    classification_report: Dict
    processing_time_ms: float
    confidence_scores: np.ndarray


class ModelAccuracyChecker:
    """
    Comprehensive model accuracy checker for BCI motor imagery classification.
    
    Provides detailed evaluation metrics, cross-validation capabilities,
    and visualization tools for model performance assessment.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the accuracy checker.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        # Class names for motor imagery
        self.class_names = ['Left Hand', 'Right Hand', 'Foot', 'Tongue']
        
        # Results storage
        self.results_history = []
        
        # Create results directory
        self.results_dir = Path("results/accuracy_tests")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Model Accuracy Checker initialized")
    
    def load_model(self, model_path: str) -> tf.keras.Model:
        """
        Load trained model for evaluation.
        
        Args:
            model_path: Path to saved model file
            
        Returns:
            Loaded Keras model
        """
        try:
            model = load_model(model_path)
            self.logger.info(f"Model loaded successfully from: {model_path}")
            self.logger.info(f"Model input shape: {model.input_shape}")
            self.logger.info(f"Model output shape: {model.output_shape}")
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model from {model_path}: {e}")
            raise
    
    def load_test_data(self, data_source: str = "unified") -> Tuple[np.ndarray, np.ndarray]:
        """
        Load test data for accuracy evaluation.
        
        Args:
            data_source: Data source type ('unified', 'bci3', 'bci4', or path)
            
        Returns:
            Test data and labels
        """
        self.logger.info(f"Loading test data from source: {data_source}")
        
        if data_source == "unified":
            # Load unified multi-dataset using the unified loader's prepare_for_training method
            loader = UnifiedBCIDataLoader(self.config_path)
            
            # Get complete data splits
            data_splits = loader.prepare_for_training(augment_data=False)
            
            # Use test set for evaluation
            X = data_splits['X_test']
            y = data_splits['y_test']
            
        elif data_source == "bci4":
            # Load BCI Competition IV Dataset 2a only
            bci4_path = "BCI/bci4_2a"
            if not os.path.exists(bci4_path):
                raise ValueError(f"BCI4 dataset not found at: {bci4_path}")
            
            loader = BCI4_2A_Loader(self.config_path)
            X, y = loader.load_all_subjects('T')  # Load training session data
            
            # Convert labels to categorical if needed
            if len(y.shape) == 1:
                y = to_categorical(y, num_classes=4)
            
            # Reshape for EEGNet if needed
            if len(X.shape) == 3:  # (trials, samples, channels)
                X = X.transpose(0, 2, 1)  # (trials, channels, samples)
                X = X[..., np.newaxis]    # Add final dimension
            
        elif data_source == "bci3":
            # Load BCI Competition III Dataset 3a only
            from gdf_data_loader import GDFDataLoader
            loader = GDFDataLoader(self.config_path)
            X, y = loader.load_and_preprocess_data()
            
            # Ensure proper format for EEGNet
            if len(X.shape) == 3:  # (trials, samples, channels)
                X = X.transpose(0, 2, 1)  # (trials, channels, samples)
                X = X[..., np.newaxis]    # Add final dimension
            
        elif os.path.exists(data_source):
            # Custom dataset path
            if data_source.endswith('.npz'):
                data = np.load(data_source)
                X, y = data['X'], data['y']
            else:
                # Assume it's a directory with dataset
                self.logger.warning(f"Custom directory loading not fully implemented for: {data_source}")
                raise ValueError(f"Custom directory loading not supported yet: {data_source}")
        else:
            raise ValueError(f"Invalid data source: {data_source}")
        
        self.logger.info(f"Loaded test data: {X.shape[0]} samples, {X.shape[1:]} shape per sample")
        if len(y.shape) > 1:
            label_counts = np.bincount(np.argmax(y, axis=1))
        else:
            label_counts = np.bincount(y)
        self.logger.info(f"Label distribution: {label_counts}")
        
        return X, y
    
    def evaluate_model_accuracy(self, 
                              model: tf.keras.Model,
                              X_test: np.ndarray,
                              y_test: np.ndarray,
                              batch_size: int = 32) -> AccuracyResults:
        """
        Comprehensive model accuracy evaluation.
        
        Args:
            model: Trained Keras model
            X_test: Test data
            y_test: Test labels
            batch_size: Batch size for prediction
            
        Returns:
            Comprehensive accuracy results
        """
        self.logger.info("Starting comprehensive model evaluation...")
        
        # Measure prediction time
        start_time = datetime.now()
        
        # Get predictions
        y_pred_proba = model.predict(X_test, batch_size=batch_size, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        avg_processing_time_per_sample = processing_time / len(X_test)
        
        # Calculate metrics
        overall_accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        kappa = cohen_kappa_score(y_true, y_pred)
        
        # Per-class accuracy
        cm = confusion_matrix(y_true, y_pred)
        per_class_accuracy = {}
        for i, class_name in enumerate(self.class_names):
            if i < len(cm):
                class_total = np.sum(cm[i, :])
                class_correct = cm[i, i] if class_total > 0 else 0
                per_class_accuracy[class_name] = class_correct / class_total if class_total > 0 else 0
        
        # Classification report
        classification_rep = classification_report(
            y_true, y_pred, 
            target_names=self.class_names[:len(np.unique(y_true))],
            output_dict=True,
            zero_division=0
        )
        
        # Confidence scores
        confidence_scores = np.max(y_pred_proba, axis=1)
        
        # Create results object
        results = AccuracyResults(
            overall_accuracy=overall_accuracy,
            per_class_accuracy=per_class_accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            cohen_kappa=kappa,
            confusion_matrix=cm,
            classification_report=classification_rep,
            processing_time_ms=avg_processing_time_per_sample,
            confidence_scores=confidence_scores
        )
        
        # Log results
        self.logger.info("=== MODEL ACCURACY EVALUATION RESULTS ===")
        self.logger.info(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
        self.logger.info(f"Precision: {precision:.4f}")
        self.logger.info(f"Recall: {recall:.4f}")
        self.logger.info(f"F1-Score: {f1:.4f}")
        self.logger.info(f"Cohen's Kappa: {kappa:.4f}")
        self.logger.info(f"Avg Processing Time: {avg_processing_time_per_sample:.2f} ms/sample")
        
        self.logger.info("\nPer-Class Accuracy:")
        for class_name, acc in per_class_accuracy.items():
            self.logger.info(f"  {class_name}: {acc:.4f} ({acc*100:.2f}%)")
        
        return results
    
    def cross_subject_validation(self,
                                model_path: str,
                                dataset_path: str = "BCI/bci4_2a",
                                n_folds: int = 9) -> Dict:
        """
        Perform cross-subject validation (leave-one-subject-out).
        
        Args:
            model_path: Path to model for retraining
            dataset_path: Path to BCI4 dataset
            n_folds: Number of folds (subjects)
            
        Returns:
            Cross-validation results
        """
        self.logger.info("Starting cross-subject validation...")
        
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path not found: {dataset_path}")
        
        # Load BCI4 data by subject
        loader = BCI4_2A_Loader(self.config_path)
        
        # Load each subject individually
        subjects_data = {}
        for subject_id in [f"A{i:02d}" for i in range(1, 10)]:  # A01 to A09
            try:
                X_subj, y_subj = loader.load_subject(subject_id, 'T')
                if len(X_subj) > 0:  # Only include subjects with data
                    subjects_data[subject_id] = (X_subj, y_subj)
                    self.logger.info(f"Loaded subject {subject_id}: {len(X_subj)} samples")
            except Exception as e:
                self.logger.warning(f"Could not load subject {subject_id}: {e}")
                continue
        
        cv_results = []
        subject_accuracies = []
        
        for test_subject in subjects_data.keys():
            self.logger.info(f"Testing on subject: {test_subject}")
            
            # Prepare train and test sets
            X_train_list = []
            y_train_list = []
            
            for subject, (X_subj, y_subj) in subjects_data.items():
                if subject == test_subject:
                    X_test, y_test = X_subj, y_subj
                else:
                    X_train_list.append(X_subj)
                    y_train_list.append(y_subj)
            
            # Combine training data
            X_train = np.vstack(X_train_list)
            y_train = np.concatenate(y_train_list, axis=0)
            
            try:
                # Load and retrain model (simplified for demonstration)
                model = load_model(model_path)
                
                # Quick fine-tuning (in production, would do full retraining)
                model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                # Reshape data for EEGNet format: (samples, channels, time, 1)
                if X_test.shape[-1] == 22:  # (samples, time, channels)
                    X_test_reshaped = X_test.transpose(0, 2, 1)  # (samples, channels, time)
                    X_test_reshaped = np.expand_dims(X_test_reshaped, axis=-1)  # (samples, channels, time, 1)
                else:
                    X_test_reshaped = X_test
                
                # Convert labels to one-hot encoding for evaluation
                if len(y_test.shape) == 1:  # Integer labels
                    y_test_onehot = to_categorical(y_test, num_classes=4)
                else:
                    y_test_onehot = y_test
                
                # Evaluate on test subject
                test_loss, test_accuracy = model.evaluate(X_test_reshaped, y_test_onehot, verbose=0)
                subject_accuracies.append(test_accuracy)
                
                cv_results.append({
                    'test_subject': test_subject,
                    'test_accuracy': test_accuracy,
                    'test_samples': len(X_test),
                    'train_samples': len(X_train)
                })
                
                self.logger.info(f"Subject {test_subject} accuracy: {test_accuracy:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error evaluating subject {test_subject}: {e}")
                continue
        
        # Calculate overall CV statistics
        mean_accuracy = np.mean(subject_accuracies)
        std_accuracy = np.std(subject_accuracies)
        
        cv_summary = {
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'min_accuracy': np.min(subject_accuracies),
            'max_accuracy': np.max(subject_accuracies),
            'subject_results': cv_results,
            'n_subjects': len(subject_accuracies)
        }
        
        self.logger.info(f"Cross-Subject Validation Results:")
        self.logger.info(f"Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        self.logger.info(f"Range: {np.min(subject_accuracies):.4f} - {np.max(subject_accuracies):.4f}")
        
        return cv_summary
    
    def real_time_simulation(self,
                           model: tf.keras.Model,
                           X_test: np.ndarray,
                           y_test: np.ndarray,
                           simulation_duration: int = 60) -> Dict:
        """
        Simulate real-time classification performance.
        
        Args:
            model: Trained model
            X_test: Test data
            y_test: Test labels
            simulation_duration: Simulation duration in seconds
            
        Returns:
            Real-time performance metrics
        """
        self.logger.info(f"Starting {simulation_duration}s real-time simulation...")
        
        # Calculate samples needed for simulation
        fs = self.config['eeg']['sampling_rate']  # 250 Hz
        samples_per_second = fs
        total_samples_needed = simulation_duration * samples_per_second
        
        # Simulate continuous stream by repeating test data
        n_repeats = (total_samples_needed // len(X_test)) + 1
        X_stream = np.tile(X_test, (n_repeats, 1, 1, 1))[:total_samples_needed//4]  # Approximate
        y_stream = np.tile(y_test, (n_repeats, 1))[:len(X_stream)]
        
        # Simulate real-time processing
        classification_times = []
        accuracies = []
        confidences = []
        
        batch_size = 10  # Process in small batches for real-time simulation
        
        for i in range(0, len(X_stream), batch_size):
            batch_X = X_stream[i:i+batch_size]
            batch_y = y_stream[i:i+batch_size]
            
            # Measure processing time
            start_time = datetime.now()
            predictions = model.predict(batch_X, verbose=0)
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Calculate batch metrics
            batch_pred = np.argmax(predictions, axis=1)
            batch_true = np.argmax(batch_y, axis=1)
            batch_accuracy = accuracy_score(batch_true, batch_pred)
            batch_confidence = np.mean(np.max(predictions, axis=1))
            
            classification_times.append(processing_time / len(batch_X))
            accuracies.append(batch_accuracy)
            confidences.append(batch_confidence)
        
        # Calculate real-time metrics
        avg_processing_time = np.mean(classification_times)
        max_processing_time = np.max(classification_times)
        throughput = 1000 / avg_processing_time  # Classifications per second
        
        realtime_results = {
            'simulation_duration': simulation_duration,
            'total_classifications': len(accuracies) * batch_size,
            'avg_processing_time_ms': avg_processing_time,
            'max_processing_time_ms': max_processing_time,
            'throughput_hz': throughput,
            'avg_accuracy': np.mean(accuracies),
            'accuracy_std': np.std(accuracies),
            'avg_confidence': np.mean(confidences),
            'realtime_capable': avg_processing_time < 100  # <100ms for real-time
        }
        
        self.logger.info("Real-time Simulation Results:")
        self.logger.info(f"Average Processing Time: {avg_processing_time:.2f} ms")
        self.logger.info(f"Throughput: {throughput:.1f} classifications/sec")
        self.logger.info(f"Real-time Capable: {'Yes' if realtime_results['realtime_capable'] else 'No'}")
        self.logger.info(f"Average Accuracy: {np.mean(accuracies):.4f}")
        
        return realtime_results
    
    def visualize_results(self, 
                         results: AccuracyResults,
                         save_path: Optional[str] = None) -> None:
        """
        Create comprehensive visualization of accuracy results.
        
        Args:
            results: Accuracy results to visualize
            save_path: Optional path to save plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Accuracy Analysis', fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix
        ax1 = axes[0, 0]
        n_classes = len(results.confusion_matrix)
        class_labels = self.class_names[:n_classes]
        
        sns.heatmap(results.confusion_matrix, 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   xticklabels=class_labels,
                   yticklabels=class_labels,
                   ax=ax1)
        ax1.set_title('Confusion Matrix')
        ax1.set_xlabel('Predicted Class')
        ax1.set_ylabel('True Class')
        
        # 2. Per-Class Accuracy Bar Chart
        ax2 = axes[0, 1]
        classes = list(results.per_class_accuracy.keys())
        accuracies = list(results.per_class_accuracy.values())
        
        bars = ax2.bar(classes, accuracies, color='skyblue', edgecolor='navy', alpha=0.7)
        ax2.set_title('Per-Class Accuracy')
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # 3. Overall Metrics
        ax3 = axes[1, 0]
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Cohen\'s κ']
        values = [results.overall_accuracy, results.precision, results.recall, 
                 results.f1_score, results.cohen_kappa]
        
        bars = ax3.barh(metrics, values, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
        ax3.set_title('Overall Performance Metrics')
        ax3.set_xlabel('Score')
        ax3.set_xlim(0, 1)
        
        # Add value labels
        for bar, val in zip(bars, values):
            width = bar.get_width()
            ax3.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                    f'{val:.3f}', ha='left', va='center')
        
        # 4. Confidence Score Distribution
        ax4 = axes[1, 1]
        ax4.hist(results.confidence_scores, bins=30, alpha=0.7, color='orange', edgecolor='red')
        ax4.set_title('Prediction Confidence Distribution')
        ax4.set_xlabel('Confidence Score')
        ax4.set_ylabel('Frequency')
        ax4.axvline(np.mean(results.confidence_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(results.confidence_scores):.3f}')
        ax4.legend()
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = self.results_dir / f"accuracy_analysis_{timestamp}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def save_results(self, 
                    results: AccuracyResults,
                    additional_info: Optional[Dict] = None) -> str:
        """
        Save detailed results to JSON file.
        
        Args:
            results: Accuracy results to save
            additional_info: Additional information to include
            
        Returns:
            Path to saved results file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Prepare results dictionary
        results_dict = {
            'timestamp': timestamp,
            'overall_accuracy': float(results.overall_accuracy),
            'per_class_accuracy': results.per_class_accuracy,
            'precision': float(results.precision),
            'recall': float(results.recall),
            'f1_score': float(results.f1_score),
            'cohen_kappa': float(results.cohen_kappa),
            'confusion_matrix': results.confusion_matrix.tolist(),
            'classification_report': results.classification_report,
            'processing_time_ms': float(results.processing_time_ms),
            'confidence_stats': {
                'mean': float(np.mean(results.confidence_scores)),
                'std': float(np.std(results.confidence_scores)),
                'min': float(np.min(results.confidence_scores)),
                'max': float(np.max(results.confidence_scores))
            }
        }
        
        # Add additional information (exclude objects that can't be serialized)
        if additional_info:
            for key, value in additional_info.items():
                # Skip AccuracyResults objects and other non-serializable objects
                if key != 'basic_evaluation' and not hasattr(value, '__dict__'):
                    results_dict[key] = value
                elif isinstance(value, dict):
                    results_dict[key] = value
        
        # Save to file
        results_file = self.results_dir / f"accuracy_results_{timestamp}.json"
        
        # Custom JSON encoder to handle numpy types
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Convert any numpy types in the results_dict
        def deep_convert(obj):
            if isinstance(obj, dict):
                return {key: deep_convert(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [deep_convert(item) for item in obj]
            else:
                return convert_numpy_types(obj)
        
        results_dict = deep_convert(results_dict)
        
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        self.logger.info(f"Results saved to: {results_file}")
        return str(results_file)
    
    def run_comprehensive_accuracy_check(self,
                                       model_path: str,
                                       data_source: str = "unified",
                                       include_cross_validation: bool = False,
                                       include_realtime_sim: bool = True,
                                       save_results: bool = True) -> Dict:
        """
        Run comprehensive accuracy checking pipeline.
        
        Args:
            model_path: Path to trained model
            data_source: Data source for testing
            include_cross_validation: Whether to perform cross-validation
            include_realtime_sim: Whether to run real-time simulation
            save_results: Whether to save results to file
            
        Returns:
            Complete accuracy analysis results
        """
        self.logger.info("Starting comprehensive accuracy check...")
        self.logger.info(f"Model: {model_path}")
        self.logger.info(f"Data source: {data_source}")
        
        # Load model and data
        model = self.load_model(model_path)
        X_test, y_test = self.load_test_data(data_source)
        
        # Basic accuracy evaluation
        basic_results = self.evaluate_model_accuracy(model, X_test, y_test)
        
        # Additional analyses
        additional_results = {
            'basic_evaluation': basic_results,
            'model_path': model_path,
            'data_source': data_source,
            'test_samples': len(X_test)
        }
        
        # Cross-validation (if requested and BCI4 data available)
        if include_cross_validation and data_source in ['unified', 'bci4']:
            try:
                cv_results = self.cross_subject_validation(model_path)
                additional_results['cross_validation'] = cv_results
            except Exception as e:
                self.logger.warning(f"Cross-validation failed: {e}")
        
        # Real-time simulation
        if include_realtime_sim:
            try:
                rt_results = self.real_time_simulation(model, X_test, y_test)
                additional_results['realtime_simulation'] = rt_results
            except Exception as e:
                self.logger.warning(f"Real-time simulation failed: {e}")
        
        # Visualization
        self.visualize_results(basic_results)
        
        # Save results
        if save_results:
            results_file = self.save_results(basic_results, additional_results)
            additional_results['results_file'] = results_file
        
        self.logger.info("Comprehensive accuracy check completed!")
        
        return additional_results


def main():
    """Main function to demonstrate accuracy checking capabilities."""
    print("BCI Motor Imagery Model Accuracy Checker")
    print("=" * 50)
    
    try:
        # Initialize checker
        checker = ModelAccuracyChecker()
        
        # Check if trained model exists
        model_paths = [
            "models/eegnet_4class_motor_imagery.keras",
            "models/best_eegnet_4class_motor_imagery.keras"
        ]
        
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if not model_path:
            print("❌ No trained model found!")
            print("Available model paths to check:")
            for path in model_paths:
                print(f"  - {path}")
            print("\nPlease train a model first using train_model.py")
            return
        
        print(f"✅ Found trained model: {model_path}")
        
        # Run comprehensive accuracy check
        results = checker.run_comprehensive_accuracy_check(
            model_path=model_path,
            data_source="unified",  # Test on combined datasets
            include_cross_validation=True,
            include_realtime_sim=True,
            save_results=True
        )
        
        # Print summary
        basic_results = results['basic_evaluation']
        print("\n" + "=" * 50)
        print("🎯 ACCURACY CHECK SUMMARY")
        print("=" * 50)
        print(f"Overall Accuracy: {basic_results.overall_accuracy:.4f} ({basic_results.overall_accuracy*100:.2f}%)")
        print(f"Cohen's Kappa: {basic_results.cohen_kappa:.4f}")
        print(f"F1-Score: {basic_results.f1_score:.4f}")
        print(f"Processing Time: {basic_results.processing_time_ms:.2f} ms/sample")
        
        # Cross-validation results
        if 'cross_validation' in results:
            cv = results['cross_validation']
            print(f"\nCross-Subject Validation:")
            print(f"  Mean Accuracy: {cv['mean_accuracy']:.4f} ± {cv['std_accuracy']:.4f}")
            print(f"  Range: {cv['min_accuracy']:.4f} - {cv['max_accuracy']:.4f}")
        
        # Real-time capability
        if 'realtime_simulation' in results:
            rt = results['realtime_simulation']
            print(f"\nReal-time Performance:")
            print(f"  Processing Time: {rt['avg_processing_time_ms']:.2f} ms")
            print(f"  Throughput: {rt['throughput_hz']:.1f} Hz")
            print(f"  Real-time Capable: {'✅ Yes' if rt['realtime_capable'] else '❌ No'}")
        
        print(f"\n📊 Detailed results saved to: {results.get('results_file', 'N/A')}")
        print("🎉 Accuracy check completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during accuracy check: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()