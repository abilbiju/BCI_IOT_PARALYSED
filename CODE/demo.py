"""
Demo Script for BCI Motor Imagery Classification
Phase 1: Core BCI Application (Intent Layer)

This script demonstrates the complete BCI pipeline functionality
including data loading, preprocessing, model training, and real-time classification.
"""

import os
import sys
import time
import numpy as np
from datetime import datetime

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import BCIDataLoader
from eegnet_model import EEGNet
from train_model import EEGNetTrainer
from bci_pipeline import BCIPipeline
from sim_generator import SimEEGStreamer


def demo_data_loading():
    """Demonstrate data loading and preprocessing."""
    print("🔄 Demo 1: Data Loading and Preprocessing")
    print("=" * 50)
    
    try:
        # Initialize data loader
        data_loader = BCIDataLoader()
        
        # Load and process data
        print("Loading BCI data...")
        X, y = data_loader.load_and_process_all_data()
        
        print(f"✅ Data loaded successfully:")
        print(f"   Shape: {X.shape}")
        print(f"   Classes: {np.unique(y)}")
        
        # Show class distribution (handles negative values)
        unique_labels, counts = np.unique(y, return_counts=True)
        class_dist = dict(zip(unique_labels, counts))
        print(f"   Class distribution: {class_dist}")
        
        # Get data info
        info = data_loader.get_data_info()
        print(f"   Total epochs: {info['total_epochs']}")
        print(f"   Channels used: {info['channels_used']}")
        print(f"   Sampling rate: {info['sampling_rate']} Hz")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in data loading demo: {e}")
        return False


def demo_model_creation():
    """Demonstrate EEGNet model creation and compilation."""
    print("\n🧠 Demo 2: EEGNet Model Creation")
    print("=" * 50)
    
    try:
        # Create EEGNet model
        print("Creating EEGNet model...")
        eegnet = EEGNet()
        model = eegnet.build_model()
        eegnet.compile_model()
        
        print("✅ Model created successfully:")
        print(f"   Total parameters: {model.count_params():,}")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        
        # Test with random data
        print("\nTesting model with random data...")
        batch_size = 16
        X_test = np.random.randn(batch_size, eegnet.chans, eegnet.samples, 1)
        predictions = model.predict(X_test, verbose=0)
        
        print(f"✅ Model prediction test:")
        print(f"   Input shape: {X_test.shape}")
        print(f"   Output shape: {predictions.shape}")
        print(f"   Sample predictions: {predictions[0]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in model creation demo: {e}")
        return False


def demo_training():
    """Demonstrate model training (abbreviated version)."""
    print("\n🎯 Demo 3: Model Training (Quick Demo)")
    print("=" * 50)
    
    try:
        # Check if model already exists
        model_path = os.path.join("models", "eegnet_motor_imagery.h5")
        if os.path.exists(model_path):
            print(f"✅ Trained model already exists at: {model_path}")
            print("   Skipping training demo. Delete model file to retrain.")
            return True
        
        print("⚠️  No trained model found.")
        print("   For full training, run: python train_model.py")
        print("   This demo shows the training setup only.")
        
        # Initialize trainer
        trainer = EEGNetTrainer()
        
        # Load and prepare data (without training)
        print("Setting up training data...")
        data_splits = trainer.load_and_prepare_data()
        
        print("✅ Training setup completed:")
        print(f"   Training samples: {data_splits['X_train'].shape[0]}")
        print(f"   Validation samples: {data_splits['X_val'].shape[0]}")
        print(f"   Test samples: {data_splits['X_test'].shape[0]}")
        
        print("\nTo run full training:")
        print("   python train_model.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in training demo: {e}")
        return False


def demo_simulation():
    """Demonstrate EEG data simulation."""
    print("\n📡 Demo 4: EEG Data Simulation")
    print("=" * 50)
    
    try:
        # Initialize simulator
        print("Initializing EEG simulator...")
        streamer = SimEEGStreamer()
        streamer.load_simulation_data()
        
        # Get simulation info
        info = streamer.get_simulation_info()
        print(f"✅ Simulation ready:")
        print(f"   Data duration: {info['duration_seconds']:.1f} seconds")
        print(f"   Channels: {info['channels']}")
        print(f"   Chunk size: {info['chunk_size']} samples")
        print(f"   Total markers: {info['total_markers']}")
        
        # Demonstrate streaming
        print("\nSimulating real-time streaming (5 chunks):")
        for i, stream_data in enumerate(streamer.stream_generator(max_chunks=5)):
            chunk = stream_data['data']
            marker = stream_data['marker']
            timestamp = stream_data['timestamp']
            
            print(f"   Chunk {i+1}: {chunk.shape}, marker={marker}, "
                  f"timestamp={datetime.fromtimestamp(timestamp).strftime('%H:%M:%S.%f')[:-3]}")
        
        # Demonstrate random epochs
        print("\nRandom epoch examples:")
        for class_target in [-1, 1]:
            epoch, label = streamer.get_random_epoch(target_class=class_target)
            print(f"   Class {class_target}: epoch shape={epoch.shape}, label={label}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in simulation demo: {e}")
        return False


def demo_pipeline():
    """Demonstrate complete BCI pipeline."""
    print("\n🚀 Demo 5: Complete BCI Pipeline")
    print("=" * 50)
    
    try:
        # Initialize pipeline
        print("Initializing BCI pipeline...")
        pipeline = BCIPipeline()
        
        # Get pipeline info
        info = pipeline.get_pipeline_info()
        print(f"✅ Pipeline initialized:")
        print(f"   Model loaded: {info['model_loaded']}")
        print(f"   Channels used: {info['channels_used']}")
        print(f"   Command mapping: {info['command_mapping']}")
        
        if not info['model_loaded']:
            print("⚠️  No trained model available.")
            print("   Run 'python train_model.py' first for full pipeline demo.")
            print("   Showing preprocessing demo only...")
            
            # Test preprocessing only
            pipeline.streamer.load_simulation_data()
            epoch, label = pipeline.streamer.get_random_epoch()
            processed = pipeline.preprocess_epoch(epoch)
            
            print(f"✅ Preprocessing test:")
            print(f"   Input: {epoch.shape} -> Output: {processed.shape}")
            
            return True
        
        # Full pipeline test with classification
        print("\nTesting complete pipeline with random epochs...")
        results = pipeline.test_with_random_epochs(num_epochs=5)
        
        print(f"✅ Pipeline test completed:")
        for i, result in enumerate(results):
            print(f"   Epoch {i+1}: {result['command']} "
                  f"(confidence: {result['confidence']:.3f}, "
                  f"time: {result['processing_time']:.3f}s)")
        
        # Show performance stats
        stats = pipeline.get_performance_stats()
        print(f"\nPerformance statistics:")
        print(f"   Total classifications: {stats['total_classifications']}")
        print(f"   Average processing time: {stats.get('avg_processing_time', 0):.3f}s")
        print(f"   Average confidence: {stats.get('avg_confidence', 0):.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in pipeline demo: {e}")
        return False


def demo_real_time_classification():
    """Demonstrate real-time classification (if model is available)."""
    print("\n⚡ Demo 6: Real-Time Classification")
    print("=" * 50)
    
    try:
        # Initialize pipeline
        pipeline = BCIPipeline()
        
        if not pipeline.model_loaded:
            print("⚠️  No trained model available for real-time demo.")
            print("   Train model first: python train_model.py")
            return True
        
        print("Starting 5-second real-time classification demo...")
        print("Commands will be displayed as they are classified:")
        print("-" * 40)
        
        # Classification counter
        classification_count = 0
        
        def classification_callback(result):
            nonlocal classification_count
            classification_count += 1
            
            command = result['command']
            confidence = result['confidence']
            processing_time = result['processing_time']
            
            # Create visual indicator based on confidence
            confidence_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
            
            print(f"🧠 [{classification_count:2d}] {command:>12} | "
                  f"Confidence: {confidence_bar} {confidence:.3f} | "
                  f"Time: {processing_time:.3f}s")
        
        # Run real-time simulation
        pipeline.run_real_time_simulation(
            duration=5.0,
            callback=classification_callback
        )
        
        print("-" * 40)
        print(f"✅ Real-time demo completed!")
        print(f"   Total classifications: {classification_count}")
        
        # Final stats
        stats = pipeline.get_performance_stats()
        print(f"   Average processing time: {stats.get('avg_processing_time', 0):.3f}s")
        print(f"   Average confidence: {stats.get('avg_confidence', 0):.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in real-time demo: {e}")
        return False


def demo_api_endpoints():
    """Demonstrate API server endpoints (basic test)."""
    print("\n🌐 Demo 7: API Server (Basic Test)")
    print("=" * 50)
    
    try:
        print("API server demonstration:")
        print("   To start the API server, run: python api_server.py")
        print("   Server will be available at: http://localhost:5000")
        print()
        print("Available endpoints:")
        print("   GET  /              - Server information")
        print("   POST /classify      - Single epoch classification")
        print("   POST /classify_batch - Batch classification")
        print("   GET  /status        - System status")
        print("   GET  /model_info    - Model information")
        print("   GET  /performance   - Performance statistics")
        print("   POST /simulate      - Simulation with stored data")
        print()
        print("Example API usage:")
        print("   curl -X GET http://localhost:5000/status")
        print("   curl -X POST http://localhost:5000/simulate")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error in API demo: {e}")
        return False


def main():
    """Run complete BCI system demonstration."""
    print("🧠 BCI Motor Imagery Classification System Demo")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Demo sequence
    demos = [
        ("Data Loading & Preprocessing", demo_data_loading),
        ("EEGNet Model Creation", demo_model_creation),
        ("Training Setup", demo_training),
        ("EEG Simulation", demo_simulation),
        ("BCI Pipeline", demo_pipeline),
        ("Real-Time Classification", demo_real_time_classification),
        ("API Server", demo_api_endpoints)
    ]
    
    results = []
    
    for demo_name, demo_func in demos:
        try:
            success = demo_func()
            results.append((demo_name, success))
            
            if success:
                print(f"✅ {demo_name} completed successfully")
            else:
                print(f"❌ {demo_name} failed")
            
        except KeyboardInterrupt:
            print(f"\n⚠️  Demo interrupted by user at {demo_name}")
            break
        except Exception as e:
            print(f"❌ Unexpected error in {demo_name}: {e}")
            results.append((demo_name, False))
        
        # Small delay between demos
        time.sleep(0.5)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Demo Summary")
    print("=" * 60)
    
    successful = sum(1 for _, success in results if success)
    total = len(results)
    
    for demo_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} {demo_name}")
    
    print(f"\nOverall: {successful}/{total} demos completed successfully")
    
    if successful == total:
        print("🎉 All demos completed successfully!")
        print("\nNext steps:")
        print("1. Train the model: python train_model.py")
        print("2. Start API server: python api_server.py")
        print("3. Integrate with IoT system using API endpoints")
    else:
        print("⚠️  Some demos failed. Check error messages above.")
        print("   Make sure bci_preprocessed_data.npz exists and dependencies are installed.")
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()