"""
Quick validation script for the BCI system
"""

import numpy as np
import yaml
import sys
import os

def validate_system():
    """Quick validation of the BCI system components."""
    print("🔍 BCI System Validation")
    print("=" * 40)
    
    try:
        # Test 1: Configuration loading
        print("1. Testing configuration loading...")
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("   ✅ Configuration loaded successfully")
        
        # Test 2: Data file existence
        print("2. Testing data file...")
        data_file = config['paths']['data_file']
        if os.path.exists(data_file):
            data = np.load(data_file)
            print(f"   ✅ Data file loaded: {len(data.keys())} files")
            
            # Check data structure
            eeg_files = [k for k in data.keys() if 'cnt' in k]
            mrk_files = [k for k in data.keys() if 'mrk' in k]
            print(f"   ✅ Found {len(eeg_files)} EEG files, {len(mrk_files)} marker files")
            
            # Sample data
            eeg_sample = data[eeg_files[0]]
            mrk_sample = data[mrk_files[0]]
            print(f"   ✅ EEG shape: {eeg_sample.shape}, Markers: {mrk_sample.shape}")
            
        else:
            print(f"   ❌ Data file not found: {data_file}")
            return False
        
        # Test 3: Module imports
        print("3. Testing module imports...")
        
        try:
            from preprocessing import PreProcessor
            print("   ✅ Preprocessing module")
        except Exception as e:
            print(f"   ❌ Preprocessing module: {e}")
            return False
        
        try:
            from eegnet_model import EEGNet
            print("   ✅ EEGNet module")
        except Exception as e:
            print(f"   ❌ EEGNet module: {e}")
            return False
        
        try:
            from sim_generator import SimEEGStreamer
            print("   ✅ Simulation module")
        except Exception as e:
            print(f"   ❌ Simulation module: {e}")
            return False
        
        try:
            from bci_pipeline import BCIPipeline
            print("   ✅ BCI Pipeline module")
        except Exception as e:
            print(f"   ❌ BCI Pipeline module: {e}")
            return False
        
        # Test 4: Model creation
        print("4. Testing model creation...")
        try:
            eegnet = EEGNet()
            model = eegnet.build_model()
            print(f"   ✅ EEGNet model created: {model.count_params():,} parameters")
        except Exception as e:
            print(f"   ❌ Model creation failed: {e}")
            return False
        
        # Test 5: Preprocessing
        print("5. Testing preprocessing...")
        try:
            preprocessor = PreProcessor()
            
            # Generate synthetic data
            samples = config['model']['samples']
            channels = config['model']['chans'] 
            synthetic_data = np.random.randn(samples, channels) * 10
            
            processed = preprocessor.preprocess_single_epoch(synthetic_data)
            print(f"   ✅ Preprocessing: {synthetic_data.shape} → {processed.shape}")
        except Exception as e:
            print(f"   ❌ Preprocessing failed: {e}")
            return False
        
        # Test 6: Simulation
        print("6. Testing simulation...")
        try:
            streamer = SimEEGStreamer()
            streamer.load_simulation_data()
            
            chunk, marker = streamer.get_chunk()
            print(f"   ✅ Simulation: chunk {chunk.shape}, marker {marker}")
        except Exception as e:
            print(f"   ❌ Simulation failed: {e}")
            return False
        
        print("\n🎉 All validation tests passed!")
        print("\nNext steps:")
        print("1. Train the model: python train_model.py")
        print("2. Run demo: python demo.py")
        print("3. Start API server: python api_server.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are installed."""
    print("📦 Checking Dependencies")
    print("=" * 30)
    
    required_packages = [
        'numpy', 'scipy', 'pandas', 'tensorflow', 'scikit-learn',
        'mne', 'Flask', 'PyYAML', 'matplotlib', 'tqdm'
    ]
    
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\nMissing packages: {missing}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All dependencies are installed!")
        return True

if __name__ == "__main__":
    print("🧠 BCI Motor Imagery System - Quick Validation")
    print("=" * 60)
    
    # Check dependencies first
    deps_ok = check_dependencies()
    if not deps_ok:
        sys.exit(1)
    
    print()
    
    # Validate system
    system_ok = validate_system()
    
    if system_ok:
        print("\n✅ System validation completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ System validation failed!")
        sys.exit(1)