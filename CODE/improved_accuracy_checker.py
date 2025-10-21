#!/usr/bin/env python3
"""
Improved BCI Motor Imagery Model Accuracy Checker
Addresses class imbalance issues and provides detailed analysis
"""

import os
import sys
import logging
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Deep learning
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# Machine learning metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    cohen_kappa_score, confusion_matrix, classification_report,
    balanced_accuracy_score
)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

# Data loading
from unified_bci_loader import UnifiedBCIDataLoader
from bci4_2a_loader import BCI4_2A_Loader

# Configuration
import yaml

class ImprovedModelAccuracyChecker:
    """Enhanced accuracy checker with class imbalance handling"""
    
    def __init__(self, config_path='config.yaml'):
        self.config_path = config_path
        self.logger = self._setup_logging()
        self.model = None
        self.config = self._load_config()
        self.class_names = ['Left Hand', 'Right Hand', 'Foot', 'Tongue']
        
        # Create results directory
        self.results_dir = Path('results/improved_accuracy_tests')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Improved Model Accuracy Checker initialized")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return {}
    
    def load_model(self, model_path):
        """Load trained model"""
        try:
            if not os.path.exists(model_path):
                self.logger.error(f"Model file not found: {model_path}")
                return False
            
            self.model = load_model(model_path)
            self.logger.info(f"Model loaded successfully from: {model_path}")
            self.logger.info(f"Model input shape: {self.model.input_shape}")
            self.logger.info(f"Model output shape: {self.model.output_shape}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
    
    def load_balanced_data(self, data_source='unified', min_samples_per_class=20):
        """Load and balance data for better evaluation"""
        try:
            self.logger.info(f"Loading balanced data from source: {data_source}")
            
            if data_source == 'unified':
                loader = UnifiedBCIDataLoader()
                X_all, y_all = loader.load_combined_datasets()
                
            else:
                self.logger.error(f"Unsupported data source: {data_source}")
                return None, None
            
            # Analyze class distribution
            unique, counts = np.unique(y_all, return_counts=True)
            self.logger.info("Original class distribution:")
            for cls, count in zip(unique, counts):
                self.logger.info(f"  Class {cls} ({self.class_names[cls]}): {count} samples")
            
            # Balance classes by undersampling majority classes and oversampling minority classes
            X_balanced, y_balanced = self._balance_classes(X_all, y_all, min_samples_per_class)
            
            # Reshape for EEGNet format: (samples, time, channels) -> (samples, channels, time, 1)
            X_balanced = X_balanced.transpose(0, 2, 1)  # (samples, channels, time)
            X_balanced = np.expand_dims(X_balanced, axis=-1)  # (samples, channels, time, 1)
            
            # Split balanced data
            from sklearn.model_selection import train_test_split
            X_train_bal, X_test_bal, y_train_bal, y_test_bal = train_test_split(
                X_balanced, y_balanced, test_size=0.3, random_state=42, stratify=y_balanced
            )
            
            self.logger.info("Balanced class distribution:")
            unique, counts = np.unique(y_test_bal, return_counts=True)
            for cls, count in zip(unique, counts):
                self.logger.info(f"  Test Class {cls} ({self.class_names[cls]}): {count} samples")
            
            return X_test_bal, y_test_bal
            
        except Exception as e:
            self.logger.error(f"Error loading balanced data: {e}")
            return None, None
    
    def _balance_classes(self, X, y, min_samples_per_class=20):
        """Balance classes using combination of undersampling and oversampling"""
        from sklearn.utils import resample
        
        unique_classes = np.unique(y)
        X_balanced = []
        y_balanced = []
        
        # Calculate target size (use median of class sizes, but at least min_samples_per_class)
        class_counts = [np.sum(y == cls) for cls in unique_classes]
        target_size = max(min_samples_per_class, int(np.median(class_counts)))
        
        self.logger.info(f"Target samples per class: {target_size}")
        
        for cls in unique_classes:
            cls_mask = y == cls
            X_cls = X[cls_mask]
            y_cls = y[cls_mask]
            
            current_size = len(X_cls)
            
            if current_size < target_size:
                # Oversample minority class
                X_resampled, y_resampled = resample(
                    X_cls, y_cls, 
                    n_samples=target_size, 
                    random_state=42,
                    replace=True
                )
                self.logger.info(f"Oversampled class {cls} from {current_size} to {target_size}")
            else:
                # Undersample majority class
                X_resampled, y_resampled = resample(
                    X_cls, y_cls, 
                    n_samples=target_size, 
                    random_state=42,
                    replace=False
                )
                self.logger.info(f"Undersampled class {cls} from {current_size} to {target_size}")
            
            X_balanced.append(X_resampled)
            y_balanced.append(y_resampled)
        
        X_balanced = np.vstack(X_balanced)
        y_balanced = np.hstack(y_balanced)
        
        # Shuffle the balanced dataset
        indices = np.random.RandomState(42).permutation(len(X_balanced))
        X_balanced = X_balanced[indices]
        y_balanced = y_balanced[indices]
        
        return X_balanced, y_balanced
    
    def evaluate_model_accuracy(self, X_test, y_test):
        """Comprehensive model evaluation with class imbalance metrics"""
        try:
            self.logger.info("Starting comprehensive model evaluation...")
            
            # Get predictions
            predictions = self.model.predict(X_test, verbose=0)
            y_pred = np.argmax(predictions, axis=1)
            y_true = y_test
            
            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            kappa = cohen_kappa_score(y_true, y_pred)
            
            # Per-class metrics
            precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
            recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
            f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Classification report
            report = classification_report(
                y_true, y_pred, 
                target_names=self.class_names,
                output_dict=True,
                zero_division=0
            )
            
            # Confidence statistics
            confidence_scores = np.max(predictions, axis=1)
            confidence_stats = {
                'mean': float(np.mean(confidence_scores)),
                'std': float(np.std(confidence_scores)),
                'min': float(np.min(confidence_scores)),
                'max': float(np.max(confidence_scores))
            }
            
            # Log results
            self.logger.info("=== IMPROVED MODEL ACCURACY EVALUATION RESULTS ===")
            self.logger.info(f"Standard Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            self.logger.info(f"Balanced Accuracy: {balanced_accuracy:.4f} ({balanced_accuracy*100:.2f}%)")
            self.logger.info(f"Precision: {precision:.4f}")
            self.logger.info(f"Recall: {recall:.4f}")
            self.logger.info(f"F1-Score: {f1:.4f}")
            self.logger.info(f"Cohen's Kappa: {kappa:.4f}")
            
            self.logger.info("\nPer-Class Performance:")
            for i, class_name in enumerate(self.class_names):
                self.logger.info(f"  {class_name}:")
                self.logger.info(f"    Precision: {precision_per_class[i]:.4f}")
                self.logger.info(f"    Recall: {recall_per_class[i]:.4f}")
                self.logger.info(f"    F1-Score: {f1_per_class[i]:.4f}")
            
            return {
                'accuracy': accuracy,
                'balanced_accuracy': balanced_accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cohen_kappa': kappa,
                'per_class_precision': precision_per_class.tolist(),
                'per_class_recall': recall_per_class.tolist(),
                'per_class_f1': f1_per_class.tolist(),
                'confusion_matrix': cm.tolist(),
                'classification_report': report,
                'confidence_stats': confidence_stats,
                'predictions': y_pred.tolist(),
                'true_labels': y_true.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Error in model evaluation: {e}")
            return None
    
    def train_balanced_model(self, X_train, y_train, X_test, y_test):
        """Train a new model with class balancing techniques"""
        try:
            self.logger.info("Training balanced model...")
            
            # Calculate class weights
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(y_train),
                y=y_train
            )
            class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
            
            self.logger.info("Class weights:")
            for cls, weight in class_weight_dict.items():
                self.logger.info(f"  {self.class_names[cls]}: {weight:.4f}")
            
            # Convert labels to categorical
            y_train_cat = to_categorical(y_train, num_classes=4)
            y_test_cat = to_categorical(y_test, num_classes=4)
            
            # Create EEGNet model
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import (
                Input, Conv2D, BatchNormalization, Activation, Dropout,
                DepthwiseConv2D, AveragePooling2D, Flatten, Dense
            )
            
            def EEGNet(nb_classes=4, Chans=22, Samples=1000, dropoutRate=0.5):
                input_layer = Input(shape=(Chans, Samples, 1))
                
                # Block 1
                conv1 = Conv2D(8, (1, 64), padding='same', use_bias=False)(input_layer)
                conv1 = BatchNormalization()(conv1)
                depthwise_conv1 = DepthwiseConv2D((Chans, 1), use_bias=False, 
                                                depth_multiplier=2, 
                                                depthwise_constraint=tf.keras.constraints.max_norm(1.))(conv1)
                depthwise_conv1 = BatchNormalization()(depthwise_conv1)
                depthwise_conv1 = Activation('elu')(depthwise_conv1)
                depthwise_conv1 = AveragePooling2D((1, 4))(depthwise_conv1)
                depthwise_conv1 = Dropout(dropoutRate)(depthwise_conv1)
                
                # Block 2
                separable_conv1 = Conv2D(16, (1, 16), padding='same', use_bias=False)(depthwise_conv1)
                separable_conv1 = BatchNormalization()(separable_conv1)
                separable_conv1 = Activation('elu')(separable_conv1)
                separable_conv1 = AveragePooling2D((1, 8))(separable_conv1)
                separable_conv1 = Dropout(dropoutRate)(separable_conv1)
                
                # Classifier
                flatten = Flatten()(separable_conv1)
                dense = Dense(nb_classes, activation='softmax',
                            kernel_constraint=tf.keras.constraints.max_norm(0.25))(flatten)
                
                model = Model(inputs=input_layer, outputs=dense)
                return model
            
            # Create and compile model
            balanced_model = EEGNet(nb_classes=4, Chans=22, Samples=1000)
            balanced_model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Train with class weights
            history = balanced_model.fit(
                X_train, y_train_cat,
                validation_data=(X_test, y_test_cat),
                epochs=100,
                batch_size=32,
                class_weight=class_weight_dict,
                verbose=1
            )
            
            # Save balanced model
            balanced_model_path = 'models/eegnet_4class_balanced.keras'
            os.makedirs('models', exist_ok=True)
            balanced_model.save(balanced_model_path)
            self.logger.info(f"Balanced model saved to: {balanced_model_path}")
            
            # Evaluate balanced model
            test_loss, test_accuracy = balanced_model.evaluate(X_test, y_test_cat, verbose=0)
            self.logger.info(f"Balanced model test accuracy: {test_accuracy:.4f}")
            
            return balanced_model, history
            
        except Exception as e:
            self.logger.error(f"Error training balanced model: {e}")
            return None, None
    
    def visualize_improved_results(self, results, save_path=None):
        """Create comprehensive visualization of improved results"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Improved Model Accuracy Analysis', fontsize=16, fontweight='bold')
            
            # 1. Confusion Matrix
            cm = np.array(results['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=self.class_names,
                       yticklabels=self.class_names,
                       ax=axes[0,0])
            axes[0,0].set_title('Confusion Matrix')
            axes[0,0].set_xlabel('Predicted Class')
            axes[0,0].set_ylabel('True Class')
            
            # 2. Per-Class Performance Comparison
            metrics = ['Precision', 'Recall', 'F1-Score']
            class_metrics = np.array([
                results['per_class_precision'],
                results['per_class_recall'], 
                results['per_class_f1']
            ])
            
            x = np.arange(len(self.class_names))
            width = 0.25
            
            for i, metric in enumerate(metrics):
                axes[0,1].bar(x + i*width, class_metrics[i], width, 
                             label=metric, alpha=0.8)
            
            axes[0,1].set_xlabel('Class')
            axes[0,1].set_ylabel('Score')
            axes[0,1].set_title('Per-Class Performance Metrics')
            axes[0,1].set_xticks(x + width)
            axes[0,1].set_xticklabels(self.class_names, rotation=45)
            axes[0,1].legend()
            axes[0,1].set_ylim(0, 1.1)
            
            # 3. Overall Performance Metrics
            overall_metrics = {
                'Standard Accuracy': results['accuracy'],
                'Balanced Accuracy': results['balanced_accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score'],
                'Cohen\'s κ': results['cohen_kappa']
            }
            
            bars = axes[0,2].bar(overall_metrics.keys(), overall_metrics.values(), 
                               color=['lightblue', 'lightgreen', 'lightcoral', 
                                     'lightyellow', 'lightpink', 'lightgray'])
            axes[0,2].set_title('Overall Performance Metrics')
            axes[0,2].set_ylabel('Score')
            axes[0,2].tick_params(axis='x', rotation=45)
            axes[0,2].set_ylim(0, 1.1)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[0,2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                              f'{height:.3f}', ha='center', va='bottom')
            
            # 4. Class Distribution in Test Set
            true_labels = np.array(results['true_labels'])
            unique, counts = np.unique(true_labels, return_counts=True)
            class_counts = {self.class_names[i]: counts[list(unique).index(i)] 
                           if i in unique else 0 for i in range(len(self.class_names))}
            
            axes[1,0].pie(class_counts.values(), labels=class_counts.keys(), 
                         autopct='%1.1f%%', startangle=90)
            axes[1,0].set_title('Test Set Class Distribution')
            
            # 5. Prediction Confidence Distribution
            confidence_scores = []
            predictions = np.array(results['predictions'])
            
            if 'confidence_stats' in results:
                # Use saved confidence if available
                axes[1,1].text(0.5, 0.5, 
                              f"Mean Confidence: {results['confidence_stats']['mean']:.3f}\n"
                              f"Std: {results['confidence_stats']['std']:.3f}\n"
                              f"Range: {results['confidence_stats']['min']:.3f} - {results['confidence_stats']['max']:.3f}",
                              ha='center', va='center', transform=axes[1,1].transAxes,
                              fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
                axes[1,1].set_title('Prediction Confidence Stats')
                axes[1,1].set_xticks([])
                axes[1,1].set_yticks([])
            
            # 6. Class-wise Accuracy Comparison
            class_accuracies = []
            for i, class_name in enumerate(self.class_names):
                class_mask = true_labels == i
                if np.sum(class_mask) > 0:
                    class_pred = predictions[class_mask]
                    class_acc = np.mean(class_pred == i)
                    class_accuracies.append(class_acc)
                else:
                    class_accuracies.append(0)
            
            bars = axes[1,2].bar(self.class_names, class_accuracies, 
                               color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
            axes[1,2].set_title('Per-Class Accuracy')
            axes[1,2].set_ylabel('Accuracy')
            axes[1,2].tick_params(axis='x', rotation=45)
            axes[1,2].set_ylim(0, 1.1)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[1,2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                              f'{height:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Improved visualization saved to: {save_path}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating improved visualization: {e}")
            return None
    
    def run_comprehensive_analysis(self, model_path='models/eegnet_4class_motor_imagery.keras'):
        """Run comprehensive analysis with balanced data"""
        try:
            print("\n" + "="*60)
            print("🔬 IMPROVED BCI Motor Imagery Model Analysis")
            print("="*60)
            
            # Load model
            if not self.load_model(model_path):
                return
            
            # Load balanced test data
            X_test, y_test = self.load_balanced_data('unified', min_samples_per_class=30)
            if X_test is None:
                return
            
            # Evaluate on balanced data
            results = self.evaluate_model_accuracy(X_test, y_test)
            if results is None:
                return
            
            # Create visualization
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            viz_path = self.results_dir / f'improved_analysis_{timestamp}.png'
            self.visualize_improved_results(results, viz_path)
            
            # Save results
            results_path = self.results_dir / f'improved_results_{timestamp}.json'
            
            # Prepare results for JSON serialization
            json_results = {
                'timestamp': timestamp,
                'model_path': model_path,
                'data_source': 'unified_balanced',
                'test_samples': len(y_test),
                'standard_accuracy': results['accuracy'],
                'balanced_accuracy': results['balanced_accuracy'],
                'precision': results['precision'],
                'recall': results['recall'],
                'f1_score': results['f1_score'],
                'cohen_kappa': results['cohen_kappa'],
                'per_class_metrics': {
                    self.class_names[i]: {
                        'precision': results['per_class_precision'][i],
                        'recall': results['per_class_recall'][i],
                        'f1_score': results['per_class_f1'][i]
                    } for i in range(len(self.class_names))
                },
                'confusion_matrix': results['confusion_matrix'],
                'classification_report': results['classification_report'],
                'confidence_stats': results['confidence_stats']
            }
            
            with open(results_path, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            # Print summary
            print(f"\n🎯 IMPROVED ANALYSIS SUMMARY")
            print("="*40)
            print(f"Standard Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
            print(f"Balanced Accuracy: {results['balanced_accuracy']:.4f} ({results['balanced_accuracy']*100:.2f}%)")
            print(f"Cohen's Kappa: {results['cohen_kappa']:.4f}")
            print(f"F1-Score: {results['f1_score']:.4f}")
            
            print(f"\nPer-Class Performance:")
            for i, class_name in enumerate(self.class_names):
                print(f"  {class_name}: P={results['per_class_precision'][i]:.3f}, "
                      f"R={results['per_class_recall'][i]:.3f}, "
                      f"F1={results['per_class_f1'][i]:.3f}")
            
            print(f"\n📊 Results saved to: {results_path}")
            print(f"📈 Visualization saved to: {viz_path}")
            print("🎉 Improved analysis completed successfully!")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive analysis: {e}")
            return None

def main():
    """Main execution function"""
    try:
        checker = ImprovedModelAccuracyChecker()
        results = checker.run_comprehensive_analysis()
        return results
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        return None

if __name__ == "__main__":
    main()