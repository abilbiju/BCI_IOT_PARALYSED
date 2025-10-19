"""
Simple BCI System Test
"""

print("🧠 BCI Motor Imagery System - Simple Test")
print("=" * 50)

try:
    print("1. Testing basic imports...")
    import numpy as np
    import yaml
    print("   ✅ numpy, yaml")
    
    import tensorflow as tf
    print("   ✅ tensorflow")
    
    import sklearn
    print("   ✅ sklearn")
    
    import flask
    print("   ✅ flask")
    
    print("2. Testing BCI modules...")
    from preprocessing import PreProcessor
    print("   ✅ preprocessing")
    
    from eegnet_model import EEGNet
    print("   ✅ eegnet_model")
    
    from sim_generator import SimEEGStreamer
    print("   ✅ sim_generator")
    
    from bci_pipeline import BCIPipeline
    print("   ✅ bci_pipeline")
    
    print("3. Testing configuration...")
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print("   ✅ config.yaml loaded")
    
    print("4. Testing data file...")
    data = np.load('bci_preprocessed_data.npz')
    print(f"   ✅ Data loaded: {len(data.keys())} files")
    
    print("5. Testing model creation...")
    eegnet = EEGNet()
    model = eegnet.build_model()
    print(f"   ✅ Model created: {model.count_params():,} parameters")
    
    print("\n🎉 All basic tests passed!")
    print("\nYour BCI system is ready!")
    print("\nNext steps:")
    print("1. Train model: python train_model.py")
    print("2. Run demo: python demo.py") 
    print("3. Start API: python api_server.py")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()