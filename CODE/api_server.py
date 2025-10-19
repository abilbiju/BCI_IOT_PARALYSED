"""
Flask API Server for BCI Motor Imagery Classification
Updated for 4-class motor imagery (BCI Competition III Dataset 3a)

This module implements the Flask REST API that serves as the gateway
for the IoT system to access BCI classification services.

Supports 4-class motor imagery: left hand, right hand, foot, tongue
"""

import numpy as np
import yaml
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional
import traceback

from flask import Flask, request, jsonify, Response
from flask_cors import CORS

from bci_pipeline import BCIPipeline
from motor_imagery_streamer import MotorImageryStreamer


class BCIAPIServer:
    """
    Flask API server for BCI motor imagery classification.
    
    Provides REST endpoints for:
    - Real-time EEG classification
    - System status monitoring
    - Model information
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize BCI API server.
        
        Args:
            config_path: Path to configuration YAML file
        """
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
        
        self.api_config = self.config['api']
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format']
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize Flask app
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for web clients
        
        # Initialize BCI pipeline
        self.pipeline = BCIPipeline(config_path)
        
        # Setup routes
        self._setup_routes()
        
        # Server state
        self.server_start_time = datetime.now()
        self.request_count = 0
        
        self.logger.info("BCI API Server initialized successfully")
    
    def _setup_routes(self) -> None:
        """Setup all API routes."""
        
        @self.app.route('/', methods=['GET'])
        def home():
            """Home endpoint with basic server information."""
            return jsonify({
                'service': 'BCI Motor Imagery Classification API',
                'version': '1.0.0',
                'status': 'running',
                'timestamp': datetime.now().isoformat(),
                'endpoints': {
                    'classify': '/classify',
                    'status': '/status',
                    'model_info': '/model_info',
                    'performance': '/performance'
                }
            })
        
        @self.app.route('/classify', methods=['POST'])
        def classify():
            """
            Main classification endpoint.
            
            Expects JSON with:
            - data: EEG data as 2D array (samples, channels)
            - metadata: Optional metadata dictionary
            
            Returns JSON with:
            - command: Classified command string
            - confidence: Confidence score (0-1)
            - probabilities: Class probability distribution
            - processing_time: Time taken for classification
            - timestamp: Server timestamp
            """
            self.request_count += 1
            
            try:
                # Validate request
                if not request.is_json:
                    return jsonify({
                        'error': 'Request must be JSON',
                        'status': 'error'
                    }), 400
                
                data = request.get_json()
                
                if 'data' not in data:
                    return jsonify({
                        'error': 'Missing required field: data',
                        'status': 'error'
                    }), 400
                
                # Extract EEG data
                eeg_data = np.array(data['data'])
                
                # Validate data shape
                if eeg_data.ndim != 2:
                    return jsonify({
                        'error': f'EEG data must be 2D array, got {eeg_data.ndim}D',
                        'status': 'error'
                    }), 400
                
                expected_samples = int(self.config['eeg']['epoch_length'] * 
                                     self.config['eeg']['sampling_rate'])
                
                if eeg_data.shape[0] != expected_samples:
                    return jsonify({
                        'error': f'Expected {expected_samples} samples, got {eeg_data.shape[0]}',
                        'status': 'error'
                    }), 400
                
                # Process through BCI pipeline
                result = self.pipeline.process_single_epoch(eeg_data, log_result=True)
                
                # Add request metadata
                result['request_id'] = self.request_count
                result['server_timestamp'] = datetime.now().isoformat()
                
                if 'metadata' in data:
                    result['client_metadata'] = data['metadata']
                
                return jsonify(result)
                
            except Exception as e:
                self.logger.error(f"Error in classification endpoint: {e}")
                return jsonify({
                    'error': str(e),
                    'status': 'error',
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/classify_batch', methods=['POST'])
        def classify_batch():
            """
            Batch classification endpoint for multiple epochs.
            
            Expects JSON with:
            - data: 3D array (epochs, samples, channels)
            - metadata: Optional metadata dictionary
            
            Returns JSON with array of classification results.
            """
            self.request_count += 1
            
            try:
                if not request.is_json:
                    return jsonify({
                        'error': 'Request must be JSON',
                        'status': 'error'
                    }), 400
                
                data = request.get_json()
                
                if 'data' not in data:
                    return jsonify({
                        'error': 'Missing required field: data',
                        'status': 'error'
                    }), 400
                
                eeg_data = np.array(data['data'])
                
                if eeg_data.ndim != 3:
                    return jsonify({
                        'error': f'Batch EEG data must be 3D array, got {eeg_data.ndim}D',
                        'status': 'error'
                    }), 400
                
                # Process each epoch
                results = []
                for i, epoch in enumerate(eeg_data):
                    try:
                        result = self.pipeline.process_single_epoch(epoch, log_result=False)
                        result['epoch_id'] = i
                        results.append(result)
                    except Exception as e:
                        results.append({
                            'epoch_id': i,
                            'error': str(e),
                            'status': 'error'
                        })
                
                return jsonify({
                    'results': results,
                    'total_epochs': len(eeg_data),
                    'successful': sum(1 for r in results if r.get('status') != 'error'),
                    'request_id': self.request_count,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                self.logger.error(f"Error in batch classification endpoint: {e}")
                return jsonify({
                    'error': str(e),
                    'status': 'error',
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/status', methods=['GET'])
        def status():
            """System status endpoint."""
            uptime = datetime.now() - self.server_start_time
            
            return jsonify({
                'status': 'running',
                'model_loaded': self.pipeline.model_loaded,
                'uptime_seconds': uptime.total_seconds(),
                'uptime_formatted': str(uptime),
                'total_requests': self.request_count,
                'server_start_time': self.server_start_time.isoformat(),
                'current_time': datetime.now().isoformat(),
                'pipeline_info': self.pipeline.get_pipeline_info()
            })
        
        @self.app.route('/model_info', methods=['GET'])
        def model_info():
            """Model information endpoint."""
            info = {
                'model_loaded': self.pipeline.model_loaded,
                'config': {
                    'channels': self.config['model']['chans'],
                    'samples_per_epoch': self.config['model']['samples'],
                    'classes': self.config['model']['nb_classes'],
                    'sampling_rate': self.config['eeg']['sampling_rate'],
                    'epoch_length': self.config['eeg']['epoch_length']
                },
                'command_mapping': self.config['commands']['class_mapping'],
                'motor_imagery_channels': self.config['eeg']['motor_imagery_channels']
            }
            
            if self.pipeline.model_loaded:
                info['model_summary'] = self.pipeline.eegnet.get_model_summary()
            
            return jsonify(info)
        
        @self.app.route('/performance', methods=['GET'])
        def performance():
            """Performance statistics endpoint."""
            stats = self.pipeline.get_performance_stats()
            
            # Add server-level stats
            stats['server_uptime'] = (datetime.now() - self.server_start_time).total_seconds()
            stats['total_api_requests'] = self.request_count
            
            return jsonify(stats)
        
        @self.app.route('/simulate', methods=['POST'])
        def simulate():
            """
            Simulation endpoint that uses stored data for testing.
            
            Expects JSON with:
            - num_epochs: Number of random epochs to classify (default: 1)
            - target_class: Optional specific class to target
            
            Returns classification results for simulated data.
            """
            try:
                data = request.get_json() if request.is_json else {}
                
                num_epochs = data.get('num_epochs', 1)
                target_class = data.get('target_class', None)
                
                if num_epochs > 10:  # Limit for API safety
                    return jsonify({
                        'error': 'Maximum 10 epochs allowed per simulation request',
                        'status': 'error'
                    }), 400
                
                # Load simulation data if not already loaded
                if self.pipeline.streamer.eeg_data is None:
                    self.pipeline.streamer.load_simulation_data()
                
                results = []
                
                for i in range(num_epochs):
                    # Get random epoch
                    epoch, true_label = self.pipeline.streamer.get_random_epoch(
                        target_class=target_class
                    )
                    
                    # Classify
                    result = self.pipeline.process_single_epoch(epoch, log_result=False)
                    result['true_label'] = int(true_label)
                    result['epoch_id'] = i
                    
                    results.append(result)
                
                return jsonify({
                    'results': results,
                    'simulation_info': self.pipeline.streamer.get_simulation_info(),
                    'request_id': self.request_count,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                self.logger.error(f"Error in simulation endpoint: {e}")
                return jsonify({
                    'error': str(e),
                    'status': 'error',
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.errorhandler(404)
        def not_found(error):
            """Handle 404 errors."""
            return jsonify({
                'error': 'Endpoint not found',
                'status': 'error',
                'available_endpoints': [
                    '/', '/classify', '/classify_batch', '/status', 
                    '/model_info', '/performance', '/simulate'
                ]
            }), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            """Handle 500 errors."""
            return jsonify({
                'error': 'Internal server error',
                'status': 'error',
                'timestamp': datetime.now().isoformat()
            }), 500
    
    def run(self, host: Optional[str] = None, port: Optional[int] = None, debug: Optional[bool] = None):
        """
        Run the Flask development server.
        
        Args:
            host: Host address (uses config if None)
            port: Port number (uses config if None)
            debug: Debug mode (uses config if None)
        """
        if host is None:
            host = self.api_config['host']
        if port is None:
            port = self.api_config['port']
        if debug is None:
            debug = self.api_config['debug']
        
        self.logger.info(f"Starting BCI API Server on {host}:{port}")
        self.logger.info(f"Model loaded: {self.pipeline.model_loaded}")
        
        try:
            self.app.run(host=host, port=port, debug=debug, threaded=True)
        except KeyboardInterrupt:
            self.logger.info("Server stopped by user")
        except Exception as e:
            self.logger.error(f"Server error: {e}")
    
    def get_app(self):
        """Get Flask app instance for external deployment."""
        return self.app


def create_api_server(config_path: str = "config.yaml") -> BCIAPIServer:
    """
    Factory function to create BCI API server.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured BCIAPIServer instance
    """
    return BCIAPIServer(config_path)


def main():
    """Main function to run the API server."""
    print("BCI Motor Imagery Classification API Server")
    print("=" * 45)
    
    try:
        # Create and run server
        server = create_api_server()
        
        print(f"Server configuration:")
        print(f"  Host: {server.api_config['host']}")
        print(f"  Port: {server.api_config['port']}")
        print(f"  Debug: {server.api_config['debug']}")
        print(f"  Model loaded: {server.pipeline.model_loaded}")
        print()
        
        if not server.pipeline.model_loaded:
            print("⚠️  WARNING: No trained model loaded!")
            print("   Train the model first using: python train_model.py")
            print("   Server will start but classification endpoints will not work.")
            print()
        
        print("Available endpoints:")
        print("  GET  /              - Server information")
        print("  POST /classify      - Single epoch classification")
        print("  POST /classify_batch - Multiple epoch classification")
        print("  GET  /status        - System status")
        print("  GET  /model_info    - Model information")
        print("  GET  /performance   - Performance statistics")
        print("  POST /simulate      - Simulation with stored data")
        print()
        
        print("Starting server... (Press Ctrl+C to stop)")
        server.run()
        
    except Exception as e:
        print(f"Error starting server: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()