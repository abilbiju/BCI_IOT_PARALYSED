# BCI Motor Imagery Classification System - Phase 1

## Overview

This is the **Phase 1: Core BCI Application (Intent Layer)** implementation for motor imagery classification using EEG signals. The system processes brain signals to classify motor imagery intentions (left hand vs right hand movement) and maps them to IoT control commands.

## 🧠 System Architecture

The system follows a modular pipeline architecture:

```
EEG Data → Preprocessing → EEGNet Model → Classification → Command Mapping → IoT Commands
```

### Core Components

1. **Data Source Module** (`sim_generator.py`) - Simulates real-time EEG streaming
2. **Preprocessing Module** (`preprocessing.py`) - Signal processing and feature extraction
3. **EEGNet Model** (`eegnet_model.py`) - Deep learning classification model
4. **BCI Pipeline** (`bci_pipeline.py`) - Orchestrates the complete data flow
5. **API Server** (`api_server.py`) - REST API gateway for IoT integration

## 📁 Project Structure

```
CODE/
├── config.yaml              # System configuration
├── requirements.txt         # Python dependencies
├──
├── # Core Modules
├── preprocessing.py          # Signal preprocessing
├── eegnet_model.py          # EEGNet deep learning model
├── data_loader.py           # Data loading and preparation
├── train_model.py           # Model training pipeline
├── sim_generator.py         # EEG data simulation
├── bci_pipeline.py          # Complete BCI pipeline
├── api_server.py            # Flask REST API server
├──
├── # Testing and Demo
├── demo.py                  # Complete system demonstration
├── test_api_client.py       # API testing client
├──
├── # Data
├── bci_preprocessed_data.npz # Preprocessed BCI Competition IV data
├── dataset_preprocess.py    # Data preprocessing script
├──
└── # Generated (after training)
    ├── models/              # Trained model storage
    └── logs/               # Training and system logs
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run System Demo

```bash
python demo.py
```

This will demonstrate all system components and validate the setup.

### 3. Train the EEGNet Model

```bash
python train_model.py
```

Training will:

- Load and preprocess BCI Competition IV data
- Train EEGNet model for motor imagery classification
- Save trained model to `models/` directory
- Generate training logs and performance plots

### 4. Start the API Server

```bash
python api_server.py
```

The API server will be available at `http://localhost:5000` with the following endpoints:

- `GET /` - Server information
- `POST /classify` - Single epoch classification
- `POST /classify_batch` - Batch classification
- `GET /status` - System status
- `GET /model_info` - Model information
- `POST /simulate` - Simulation with stored data

### 5. Test the API

```bash
python test_api_client.py
```

## 📊 Data Format

### Input EEG Data

- **Shape**: `(samples, channels)` for single epoch
- **Samples**: 256 (1 second at 256 Hz sampling rate)
- **Channels**: 35 motor imagery channels (C3, Cz, C4 region)
- **Data type**: Float values in microvolts

### Output Commands

- **Class 0**: `"TURN_LEFT"` (Left hand motor imagery)
- **Class 1**: `"TURN_RIGHT"` (Right hand motor imagery)

## 🔧 Configuration

The system is configured via `config.yaml`:

### Key Parameters

```yaml
eeg:
  sampling_rate: 256 # Hz
  epoch_length: 1.0 # seconds
  channels: 59 # total channels
  motor_imagery_channels: [7, 9, 11, ...] # selected channels

preprocessing:
  bandpass:
    low_freq: 4.0 # Hz (mu rhythm)
    high_freq: 30.0 # Hz (beta rhythm)
  reference: "CAR" # Common Average Reference
  normalization: "zscore" # Z-score normalization

model:
  nb_classes: 2 # Left vs Right hand
  chans: 35 # Motor imagery channels
  samples: 256 # Samples per epoch
  dropoutRate: 0.5
  kernLength: 32

commands:
  class_mapping:
    0: "TURN_LEFT"
    1: "TURN_RIGHT"
```

## 🧪 Testing and Validation

### System Tests

1. **Data Loading Test** - Validates BCI data preprocessing
2. **Model Architecture Test** - Confirms EEGNet model structure
3. **Preprocessing Test** - Verifies signal processing pipeline
4. **Classification Test** - Tests complete classification pipeline
5. **API Integration Test** - Validates REST API endpoints

### Performance Metrics

- **Classification Accuracy**: Typically 70-85% for motor imagery
- **Processing Time**: ~10-50ms per epoch
- **API Response Time**: ~100-300ms including network overhead

## 🔬 Technical Details

### EEGNet Architecture

The system uses EEGNet, a compact CNN designed for EEG classification:

1. **Temporal Convolution** - Learns frequency filters (F1=8)
2. **Depthwise Spatial Convolution** - Learns spatial filters (D=2)
3. **Separable Convolution** - Combines features (F2=16)
4. **Dense Classification** - Maps to motor imagery classes

### Preprocessing Pipeline

1. **Channel Selection** - Focus on motor cortex regions
2. **Bandpass Filtering** - 4-30 Hz (mu and beta rhythms)
3. **Common Average Reference** - Removes common noise
4. **Epoching** - Extract 1-second windows
5. **Artifact Removal** - Remove high-amplitude artifacts
6. **Normalization** - Z-score standardization

### Real-time Processing

- **Chunk-based Streaming** - Processes 256-sample chunks
- **Overlapping Windows** - Optional overlap for smoother classification
- **Command Debouncing** - Prevents rapid command switching
- **Confidence Thresholding** - Filters low-confidence predictions

## 📈 Performance Optimization

### Model Performance

- **Data Augmentation** - Gaussian noise and time shifting
- **Early Stopping** - Prevents overfitting
- **Learning Rate Scheduling** - Adaptive learning rate
- **Batch Normalization** - Stabilizes training

### API Performance

- **Threaded Flask Server** - Handles concurrent requests
- **Model Caching** - Loads model once at startup
- **Batch Processing** - Efficient multi-epoch classification
- **Response Compression** - Reduced network overhead

## 🔌 IoT Integration

### API Endpoints for IoT

#### Single Classification

```bash
curl -X POST http://localhost:5000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "data": [[...], [...], ...],  # 256x35 EEG data
    "metadata": {"device_id": "bci_001"}
  }'
```

Response:

```json
{
  "command": "TURN_LEFT",
  "confidence": 0.87,
  "processing_time": 0.045,
  "timestamp": "2025-10-18T10:30:45.123456"
}
```

#### Simulation Mode

```bash
curl -X POST http://localhost:5000/simulate \
  -H "Content-Type: application/json" \
  -d '{"num_epochs": 5}'
```

### Integration Examples

#### MQTT Integration

```python
import paho.mqtt.client as mqtt
import requests

def classify_and_publish(eeg_data):
    # Classify via API
    response = requests.post("http://localhost:5000/classify",
                           json={"data": eeg_data.tolist()})

    if response.status_code == 200:
        result = response.json()
        command = result["command"]

        # Publish to MQTT
        client.publish("bci/commands", command)
```

#### Arduino Integration

```cpp
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

void sendEEGData(float eegData[][35]) {
    HTTPClient http;
    http.begin("http://192.168.1.100:5000/classify");
    http.addHeader("Content-Type", "application/json");

    // Create JSON payload
    DynamicJsonDocument doc(8192);
    JsonArray data = doc.createNestedArray("data");

    for(int i = 0; i < 256; i++) {
        JsonArray sample = data.createNestedArray();
        for(int j = 0; j < 35; j++) {
            sample.add(eegData[i][j]);
        }
    }

    String payload;
    serializeJson(doc, payload);

    int httpResponseCode = http.POST(payload);

    if(httpResponseCode == 200) {
        String response = http.getString();
        // Parse command and control devices
    }
}
```

## 🛠️ Troubleshooting

### Common Issues

1. **ModuleNotFoundError**

   ```bash
   pip install -r requirements.txt
   ```

2. **Data file not found**

   - Ensure `bci_preprocessed_data.npz` exists
   - Run `python dataset_preprocess.py` if needed

3. **Model not loaded**

   ```bash
   python train_model.py  # Train model first
   ```

4. **API connection refused**

   ```bash
   python api_server.py  # Start server first
   ```

5. **Low classification accuracy**
   - Check data quality and preprocessing
   - Increase training epochs
   - Verify channel selection for motor imagery

### Performance Issues

1. **Slow processing**

   - Use GPU for training: `pip install tensorflow-gpu`
   - Reduce model complexity in config
   - Optimize preprocessing parameters

2. **Memory errors**
   - Reduce batch size in training
   - Use data generators for large datasets
   - Monitor system memory usage

## 📚 References

1. **EEGNet Paper**: Lawhern, V. J., et al. (2018). EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces. Journal of neural engineering, 15(5), 056013.

2. **Motor Imagery**: Pfurtscheller, G., & Neuper, C. (2001). Motor imagery and direct brain-computer communication. Proceedings of the IEEE, 89(7), 1123-1134.

3. **BCI Competition IV**: Tangermann, M., et al. (2012). Review of the BCI competition IV. Frontiers in neuroscience, 6, 55.

## 🔮 Future Enhancements (Phase 2+)

1. **Multi-class Classification** - Support for more motor imagery classes
2. **Real-time EEG Hardware** - Integration with OpenBCI, Emotiv, etc.
3. **Advanced Preprocessing** - Independent Component Analysis (ICA)
4. **Model Optimization** - Quantization for edge deployment
5. **IoT Device Drivers** - Direct hardware control interfaces
6. **Web Dashboard** - Real-time monitoring and control interface
7. **Multi-user Support** - User-specific model adaptation

## 📄 License

This project is part of a BCI IoT system for paralyzed individuals. Please ensure ethical use and appropriate medical/safety considerations when implementing.

## 🤝 Contributing

1. Follow the modular architecture
2. Add comprehensive tests for new features
3. Update configuration schema as needed
4. Document API changes
5. Ensure backward compatibility

---

**Phase 1 Complete**: Core BCI application with motor imagery classification, preprocessing pipeline, EEGNet model, and REST API for IoT integration.
