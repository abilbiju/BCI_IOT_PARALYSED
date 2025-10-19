"""
EEGNet Model Implementation for Motor Imagery Classification
Phase 1: Core BCI Application (Intent Layer)

This module implements the EEGNet architecture specifically designed for 
EEG-based Brain-Computer Interface applications. EEGNet combines temporal 
and spatial filtering in a compact CNN architecture optimized for EEG data.

Reference:
Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. (2018).
EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces.
Journal of neural engineering, 15(5), 056013.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, Conv2D, DepthwiseConv2D, SeparableConv2D,
    BatchNormalization, Activation, AveragePooling2D, Dropout,
    Flatten, Dense, Reshape, Permute
)
from tensorflow.keras.constraints import max_norm
import numpy as np
import yaml
import logging
from typing import Tuple, Optional
import os


class EEGNet:
    """
    EEGNet implementation for motor imagery classification.
    
    The EEGNet architecture consists of:
    1. Temporal convolution to learn frequency filters
    2. Depthwise convolution to learn spatial filters  
    3. Separable convolution for feature combination
    4. Dense layer for final classification
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize EEGNet with configuration parameters.
        
        Args:
            config_path: Path to configuration YAML file
        """
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.model_config = self.config['model']
        
        # Model architecture parameters
        self.nb_classes = self.model_config['nb_classes']
        self.chans = self.model_config['chans']
        self.samples = self.model_config['samples']
        self.dropoutRate = self.model_config['dropoutRate']
        self.kernLength = self.model_config['kernLength']
        self.F1 = self.model_config['F1']
        self.D = self.model_config['D']
        self.F2 = self.model_config['F2']
        self.norm_rate = self.model_config['norm_rate']
        self.dropoutType = self.model_config['dropoutType']
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format']
        )
        self.logger = logging.getLogger(__name__)
        
        self.model = None
    
    def build_model(self) -> Model:
        """
        Build the EEGNet model architecture.
        
        Returns:
            Compiled Keras model ready for training
        """
        self.logger.info("Building EEGNet model...")
        
        # Input layer
        input_layer = Input(shape=(self.chans, self.samples, 1))
        
        ##################################################################
        # Block 1: Temporal Convolution
        ##################################################################
        
        # First temporal convolution learns temporal filters
        block1 = Conv2D(
            filters=self.F1,
            kernel_size=(1, self.kernLength),
            padding='same',
            use_bias=False,
            input_shape=(self.chans, self.samples, 1)
        )(input_layer)
        
        block1 = BatchNormalization()(block1)
        
        # Depthwise convolution learns spatial filters for each temporal filter
        block1 = DepthwiseConv2D(
            kernel_size=(self.chans, 1),
            use_bias=False,
            depth_multiplier=self.D,
            depthwise_constraint=max_norm(1.)
        )(block1)
        
        block1 = BatchNormalization()(block1)
        block1 = Activation('elu')(block1)
        block1 = AveragePooling2D(pool_size=(1, 4))(block1)
        
        # Apply dropout
        if self.dropoutType == 'SpatialDropout2D':
            block1 = layers.SpatialDropout2D(self.dropoutRate)(block1)
        else:
            block1 = Dropout(self.dropoutRate)(block1)
        
        ##################################################################
        # Block 2: Separable Convolution
        ##################################################################
        
        # Separable convolution for feature combination
        block2 = SeparableConv2D(
            filters=self.F2,
            kernel_size=(1, 16),
            use_bias=False,
            padding='same'
        )(block1)
        
        block2 = BatchNormalization()(block2)
        block2 = Activation('elu')(block2)
        block2 = AveragePooling2D(pool_size=(1, 8))(block2)
        
        # Apply dropout
        if self.dropoutType == 'SpatialDropout2D':
            block2 = layers.SpatialDropout2D(self.dropoutRate)(block2)
        else:
            block2 = Dropout(self.dropoutRate)(block2)
        
        ##################################################################
        # Classification Block
        ##################################################################
        
        # Flatten for dense layer
        flatten = Flatten(name='flatten')(block2)
        
        # Dense layer with constraint for classification
        dense = Dense(
            units=self.nb_classes,
            kernel_constraint=max_norm(self.norm_rate),
            name='classification'
        )(flatten)
        
        # Softmax activation for multi-class classification
        softmax = Activation('softmax', name='softmax')(dense)
        
        # Create model
        self.model = Model(inputs=input_layer, outputs=softmax)
        
        self.logger.info(f"EEGNet model built successfully:")
        self.logger.info(f"  - Input shape: ({self.chans}, {self.samples}, 1)")
        self.logger.info(f"  - Output classes: {self.nb_classes}")
        self.logger.info(f"  - Total parameters: {self.model.count_params():,}")
        
        return self.model
    
    def compile_model(
        self, 
        optimizer: str = 'adam',
        learning_rate: Optional[float] = None,
        loss: str = 'categorical_crossentropy',
        metrics: list = ['accuracy']
    ) -> None:
        """
        Compile the EEGNet model with specified optimizer and loss.
        
        Args:
            optimizer: Optimizer name (default: 'adam')
            learning_rate: Learning rate (uses config if None)
            loss: Loss function name
            metrics: List of metrics to track
        """
        if self.model is None:
            raise ValueError("Model must be built before compilation")
        
        # Use learning rate from config if not specified
        if learning_rate is None:
            learning_rate = self.config['training']['learning_rate']
        
        # Create optimizer with learning rate
        if optimizer.lower() == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer.lower() == 'rmsprop':
            opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            opt = optimizer
        
        self.model.compile(
            optimizer=opt,
            loss=loss,
            metrics=metrics
        )
        
        self.logger.info(f"Model compiled with {optimizer} optimizer (lr={learning_rate})")
    
    def get_model_summary(self) -> str:
        """
        Get a string representation of the model architecture.
        
        Returns:
            Model summary as string
        """
        if self.model is None:
            return "Model not built yet"
        
        summary_list = []
        self.model.summary(print_fn=lambda x: summary_list.append(x))
        return '\n'.join(summary_list)
    
    def save_model(self, filepath: Optional[str] = None) -> str:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save model (uses config path if None)
            
        Returns:
            Path where model was saved
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        if filepath is None:
            # Create models directory if it doesn't exist
            models_dir = self.config['paths']['model_save_path']
            os.makedirs(models_dir, exist_ok=True)
            filepath = os.path.join(models_dir, self.config['paths']['model_filename'])
        
        self.model.save(filepath)
        self.logger.info(f"Model saved to: {filepath}")
        
        return filepath
    
    def load_model(self, filepath: Optional[str] = None) -> Model:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to load model from (uses config path if None)
            
        Returns:
            Loaded Keras model
        """
        if filepath is None:
            models_dir = self.config['paths']['model_save_path']
            filepath = os.path.join(models_dir, self.config['paths']['model_filename'])
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        self.model = keras.models.load_model(filepath)
        self.logger.info(f"Model loaded from: {filepath}")
        
        return self.model
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on input data.
        
        Args:
            X: Input data of shape (trials, channels, samples, 1)
            
        Returns:
            predictions: Class probabilities
            predicted_classes: Predicted class indices
        """
        if self.model is None:
            raise ValueError("Model must be built and trained before prediction")
        
        # Ensure correct input shape
        if X.ndim == 3:  # (trials, channels, samples)
            X = X[..., np.newaxis]  # Add channel dimension
        
        predictions = self.model.predict(X)
        predicted_classes = np.argmax(predictions, axis=1)
        
        return predictions, predicted_classes
    
    def predict_single(self, x: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Make prediction on a single epoch.
        
        Args:
            x: Single epoch of shape (channels, samples) or (channels, samples, 1)
            
        Returns:
            probabilities: Class probabilities
            predicted_class: Predicted class index
        """
        if self.model is None:
            raise ValueError("Model must be built and trained before prediction")
        
        # Ensure correct input shape
        if x.ndim == 2:  # (channels, samples)
            x = x[np.newaxis, ..., np.newaxis]  # Add batch and channel dimensions
        elif x.ndim == 3:  # (channels, samples, 1)
            x = x[np.newaxis, ...]  # Add batch dimension
        
        probabilities = self.model.predict(x, verbose=0)[0]  # Get first (and only) prediction
        predicted_class = np.argmax(probabilities)
        
        return probabilities, predicted_class


def create_eegnet_model(config_path: str = "config.yaml") -> EEGNet:
    """
    Factory function to create and build an EEGNet model.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Built and compiled EEGNet model
    """
    eegnet = EEGNet(config_path)
    eegnet.build_model()
    eegnet.compile_model()
    
    return eegnet


def test_eegnet():
    """Test function for EEGNet model."""
    print("Testing EEGNet model...")
    
    # Create EEGNet instance
    eegnet = EEGNet()
    
    # Build model
    model = eegnet.build_model()
    eegnet.compile_model()
    
    # Print model summary
    print("\nModel Summary:")
    print(eegnet.get_model_summary())
    
    # Test with synthetic data
    batch_size = 32
    X_test = np.random.randn(batch_size, eegnet.chans, eegnet.samples, 1)
    
    print(f"\nTesting with synthetic data: {X_test.shape}")
    
    # Make predictions
    predictions, predicted_classes = eegnet.predict(X_test)
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predicted classes shape: {predicted_classes.shape}")
    print(f"Sample predictions: {predictions[:5]}")
    print(f"Sample predicted classes: {predicted_classes[:5]}")
    
    # Test single prediction
    single_x = X_test[0]  # Shape: (channels, samples, 1)
    probs, pred_class = eegnet.predict_single(single_x)
    
    print(f"\nSingle prediction:")
    print(f"Probabilities: {probs}")
    print(f"Predicted class: {pred_class}")
    
    print("\nEEGNet model test completed successfully!")


if __name__ == "__main__":
    test_eegnet()