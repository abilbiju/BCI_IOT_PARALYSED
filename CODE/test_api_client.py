"""
Test Client for BCI API Server
Phase 1: Core BCI Application (Intent Layer)

This script tests the BCI API server endpoints with various requests
to validate the complete system functionality.
"""

import requests
import json
import numpy as np
import time
from typing import Dict, Any, Optional
import yaml


class BCIAPIClient:
    """
    Test client for BCI API server.
    
    Provides methods to test all API endpoints with various data.
    """
    
    def __init__(self, base_url: str = "http://localhost:5000", config_path: str = "config.yaml"):
        """
        Initialize API client.
        
        Args:
            base_url: Base URL of the API server
            config_path: Path to configuration file
        """
        self.base_url = base_url.rstrip('/')
        
        # Load config for data generation
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
        
        self.sampling_rate = self.config['eeg']['sampling_rate']
        self.epoch_length = self.config['eeg']['epoch_length']
        self.channels = len(self.config['eeg']['motor_imagery_channels'])
        self.samples_per_epoch = int(self.epoch_length * self.sampling_rate)
        
        print(f"BCI API Client initialized")
        print(f"Server URL: {self.base_url}")
        print(f"Expected data format: {self.samples_per_epoch} samples × {self.channels} channels")
    
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make HTTP request to API server.
        
        Args:
            method: HTTP method (GET, POST)
            endpoint: API endpoint
            data: Request data (for POST)
            
        Returns:
            Response JSON or error information
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == 'GET':
                response = requests.get(url, timeout=30)
            elif method.upper() == 'POST':
                response = requests.post(url, json=data, timeout=30)
            else:
                return {'error': f'Unsupported method: {method}'}
            
            # Try to parse JSON response
            try:
                return response.json()
            except json.JSONDecodeError:
                return {
                    'error': 'Invalid JSON response',
                    'status_code': response.status_code,
                    'text': response.text[:200]  # First 200 chars
                }
                
        except requests.exceptions.ConnectionError:
            return {'error': 'Connection failed. Is the server running?'}
        except requests.exceptions.Timeout:
            return {'error': 'Request timeout'}
        except Exception as e:
            return {'error': f'Request failed: {e}'}
    
    def test_home(self) -> bool:
        """Test the home endpoint."""
        print("\n🏠 Testing Home Endpoint")
        print("-" * 30)
        
        response = self._make_request('GET', '/')
        
        if 'error' in response:
            print(f"❌ Error: {response['error']}")
            return False
        
        print(f"✅ Service: {response.get('service', 'Unknown')}")
        print(f"✅ Status: {response.get('status', 'Unknown')}")
        print(f"✅ Endpoints: {list(response.get('endpoints', {}).keys())}")
        
        return True
    
    def test_status(self) -> bool:
        """Test the status endpoint."""
        print("\n📊 Testing Status Endpoint")
        print("-" * 30)
        
        response = self._make_request('GET', '/status')
        
        if 'error' in response:
            print(f"❌ Error: {response['error']}")
            return False
        
        print(f"✅ Status: {response.get('status', 'Unknown')}")
        print(f"✅ Model loaded: {response.get('model_loaded', False)}")
        print(f"✅ Uptime: {response.get('uptime_formatted', 'Unknown')}")
        print(f"✅ Total requests: {response.get('total_requests', 0)}")
        
        return True
    
    def test_model_info(self) -> bool:
        """Test the model info endpoint."""
        print("\n🧠 Testing Model Info Endpoint")
        print("-" * 30)
        
        response = self._make_request('GET', '/model_info')
        
        if 'error' in response:
            print(f"❌ Error: {response['error']}")
            return False
        
        print(f"✅ Model loaded: {response.get('model_loaded', False)}")
        
        config = response.get('config', {})
        print(f"✅ Channels: {config.get('channels', 'Unknown')}")
        print(f"✅ Samples per epoch: {config.get('samples_per_epoch', 'Unknown')}")
        print(f"✅ Classes: {config.get('classes', 'Unknown')}")
        
        command_mapping = response.get('command_mapping', {})
        print(f"✅ Command mapping: {command_mapping}")
        
        return True
    
    def test_performance(self) -> bool:
        """Test the performance endpoint."""
        print("\n📈 Testing Performance Endpoint")
        print("-" * 30)
        
        response = self._make_request('GET', '/performance')
        
        if 'error' in response:
            print(f"❌ Error: {response['error']}")
            return False
        
        print(f"✅ Total classifications: {response.get('total_classifications', 0)}")
        print(f"✅ Server uptime: {response.get('server_uptime', 0):.1f}s")
        print(f"✅ API requests: {response.get('total_api_requests', 0)}")
        
        if response.get('avg_processing_time'):
            print(f"✅ Avg processing time: {response['avg_processing_time']:.3f}s")
        
        return True
    
    def generate_synthetic_eeg(self, noise_level: float = 1.0) -> np.ndarray:
        """
        Generate synthetic EEG data for testing.
        
        Args:
            noise_level: Amplitude of noise to add
            
        Returns:
            Synthetic EEG data of shape (samples, channels)
        """
        # Generate base frequencies (simulate motor imagery rhythms)
        t = np.linspace(0, self.epoch_length, self.samples_per_epoch)
        
        # Create multi-channel EEG-like data
        data = np.zeros((self.samples_per_epoch, self.channels))
        
        for ch in range(self.channels):
            # Base rhythm (around 10 Hz)
            base_freq = 8 + np.random.rand() * 4  # 8-12 Hz
            signal = np.sin(2 * np.pi * base_freq * t)
            
            # Add harmonics
            signal += 0.3 * np.sin(2 * np.pi * base_freq * 2 * t)
            signal += 0.1 * np.sin(2 * np.pi * base_freq * 3 * t)
            
            # Add noise
            noise = np.random.normal(0, noise_level, len(t))
            
            # Channel-specific scaling
            scaling = 0.5 + np.random.rand() * 1.0
            
            data[:, ch] = (signal + noise) * scaling
        
        return data
    
    def test_classify_single(self) -> bool:
        """Test single epoch classification."""
        print("\n🔍 Testing Single Classification Endpoint")
        print("-" * 30)
        
        # Generate synthetic EEG data
        eeg_data = self.generate_synthetic_eeg()
        
        request_data = {
            'data': eeg_data.tolist(),
            'metadata': {
                'test_type': 'synthetic',
                'client': 'test_client'
            }
        }
        
        print(f"Sending EEG data: {eeg_data.shape}")
        
        response = self._make_request('POST', '/classify', request_data)
        
        if 'error' in response:
            print(f"❌ Error: {response['error']}")
            return False
        
        print(f"✅ Command: {response.get('command', 'Unknown')}")
        print(f"✅ Confidence: {response.get('confidence', 0):.3f}")
        print(f"✅ Processing time: {response.get('processing_time', 0):.3f}s")
        print(f"✅ Status: {response.get('status', 'Unknown')}")
        
        return True
    
    def test_classify_batch(self) -> bool:
        """Test batch classification."""
        print("\n📦 Testing Batch Classification Endpoint")
        print("-" * 30)
        
        # Generate multiple synthetic epochs
        batch_size = 3
        batch_data = []
        
        for i in range(batch_size):
            epoch = self.generate_synthetic_eeg(noise_level=0.5 + i * 0.2)
            batch_data.append(epoch.tolist())
        
        request_data = {
            'data': batch_data,
            'metadata': {
                'test_type': 'batch_synthetic',
                'batch_size': batch_size
            }
        }
        
        print(f"Sending batch of {batch_size} epochs")
        
        response = self._make_request('POST', '/classify_batch', request_data)
        
        if 'error' in response:
            print(f"❌ Error: {response['error']}")
            return False
        
        print(f"✅ Total epochs: {response.get('total_epochs', 0)}")
        print(f"✅ Successful: {response.get('successful', 0)}")
        
        results = response.get('results', [])
        for i, result in enumerate(results):
            if result.get('status') != 'error':
                print(f"   Epoch {i}: {result.get('command', 'Unknown')} "
                      f"(confidence: {result.get('confidence', 0):.3f})")
            else:
                print(f"   Epoch {i}: Error - {result.get('error', 'Unknown')}")
        
        return True
    
    def test_simulate(self) -> bool:
        """Test simulation endpoint."""
        print("\n🎮 Testing Simulation Endpoint")
        print("-" * 30)
        
        request_data = {
            'num_epochs': 3,
            'target_class': None  # Random class
        }
        
        print("Requesting simulation with 3 random epochs")
        
        response = self._make_request('POST', '/simulate', request_data)
        
        if 'error' in response:
            print(f"❌ Error: {response['error']}")
            return False
        
        results = response.get('results', [])
        print(f"✅ Received {len(results)} simulation results")
        
        for i, result in enumerate(results):
            true_label = result.get('true_label', 'Unknown')
            command = result.get('command', 'Unknown')
            confidence = result.get('confidence', 0)
            
            print(f"   Epoch {i}: True={true_label}, Predicted={command}, "
                  f"Confidence={confidence:.3f}")
        
        return True
    
    def test_error_handling(self) -> bool:
        """Test API error handling."""
        print("\n⚠️ Testing Error Handling")
        print("-" * 30)
        
        # Test 1: Invalid endpoint
        response = self._make_request('GET', '/invalid_endpoint')
        if response.get('error') == 'Endpoint not found':
            print("✅ 404 error handling works")
        else:
            print(f"❌ Unexpected 404 response: {response}")
            return False
        
        # Test 2: Invalid data format
        invalid_data = {'invalid': 'data'}
        response = self._make_request('POST', '/classify', invalid_data)
        if 'error' in response:
            print("✅ Invalid data error handling works")
        else:
            print(f"❌ Expected error for invalid data: {response}")
            return False
        
        # Test 3: Wrong data shape
        wrong_shape_data = {
            'data': [[1, 2, 3], [4, 5, 6]]  # Too small
        }
        response = self._make_request('POST', '/classify', wrong_shape_data)
        if 'error' in response:
            print("✅ Wrong shape error handling works")
        else:
            print(f"❌ Expected error for wrong shape: {response}")
            return False
        
        return True
    
    def run_load_test(self, num_requests: int = 10) -> bool:
        """Run basic load test."""
        print(f"\n🔥 Running Load Test ({num_requests} requests)")
        print("-" * 30)
        
        start_time = time.time()
        success_count = 0
        response_times = []
        
        for i in range(num_requests):
            # Generate synthetic data
            eeg_data = self.generate_synthetic_eeg(noise_level=0.5)
            
            request_data = {
                'data': eeg_data.tolist(),
                'metadata': {'request_id': i}
            }
            
            # Time the request
            request_start = time.time()
            response = self._make_request('POST', '/classify', request_data)
            request_time = time.time() - request_start
            
            if 'error' not in response:
                success_count += 1
                response_times.append(request_time)
            
            print(f"   Request {i+1}/{num_requests}: "
                  f"{'✅' if 'error' not in response else '❌'} "
                  f"({request_time:.3f}s)")
        
        total_time = time.time() - start_time
        
        print(f"\nLoad Test Results:")
        print(f"✅ Successful requests: {success_count}/{num_requests}")
        print(f"✅ Success rate: {success_count/num_requests*100:.1f}%")
        print(f"✅ Total time: {total_time:.2f}s")
        print(f"✅ Requests per second: {num_requests/total_time:.2f}")
        
        if response_times:
            avg_response_time = np.mean(response_times)
            max_response_time = np.max(response_times)
            print(f"✅ Average response time: {avg_response_time:.3f}s")
            print(f"✅ Maximum response time: {max_response_time:.3f}s")
        
        return success_count > 0
    
    def run_full_test_suite(self) -> None:
        """Run complete test suite."""
        print("🧪 BCI API Server Test Suite")
        print("=" * 50)
        
        tests = [
            ("Home Endpoint", self.test_home),
            ("Status Endpoint", self.test_status),
            ("Model Info Endpoint", self.test_model_info),
            ("Performance Endpoint", self.test_performance),
            ("Single Classification", self.test_classify_single),
            ("Batch Classification", self.test_classify_batch),
            ("Simulation", self.test_simulate),
            ("Error Handling", self.test_error_handling),
            ("Load Test", lambda: self.run_load_test(5))  # 5 requests for demo
        ]
        
        results = []
        
        for test_name, test_func in tests:
            try:
                success = test_func()
                results.append((test_name, success))
                
                if success:
                    print(f"✅ {test_name} PASSED")
                else:
                    print(f"❌ {test_name} FAILED")
                    
            except Exception as e:
                print(f"❌ {test_name} ERROR: {e}")
                results.append((test_name, False))
            
            time.sleep(0.5)  # Small delay between tests
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 Test Results Summary")
        print("=" * 50)
        
        passed = sum(1 for _, success in results if success)
        total = len(results)
        
        for test_name, success in results:
            status = "PASS" if success else "FAIL"
            print(f"   {status:4} {test_name}")
        
        print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        if passed == total:
            print("🎉 All tests passed! API server is working correctly.")
        else:
            print("⚠️  Some tests failed. Check the server and model status.")


def main():
    """Main function to run API tests."""
    print("Starting BCI API Server Tests...")
    print("Make sure the API server is running: python api_server.py")
    print()
    
    try:
        # Initialize client
        client = BCIAPIClient()
        
        # Wait a moment for user to start server if needed
        input("Press Enter when API server is running...")
        
        # Run test suite
        client.run_full_test_suite()
        
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"❌ Error running tests: {e}")


if __name__ == "__main__":
    main()