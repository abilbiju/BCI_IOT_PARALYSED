import os
import numpy as np
from pathlib import Path
import gc
import sys

def load_bci_txt_file(file_path):
    """Load BCI Competition .txt files efficiently"""
    try:
        # Try to load as numeric data first
        data = np.loadtxt(file_path)
        return data
    except:
        try:
            # If that fails, try reading line by line
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Skip header lines and convert to numeric
            numeric_lines = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('|'):
                    try:
                        # Split and convert to float
                        values = [float(x) for x in line.split()]
                        if values:  # Only add non-empty lines
                            numeric_lines.append(values)
                    except ValueError:
                        continue
            
            if numeric_lines:
                return np.array(numeric_lines)
            else:
                return None
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None

def process_gdf_file(file_path):
    """Process GDF files without MNE (basic approach)"""
    try:
        # For GDF files, we'll skip them in this simplified version
        # as they require specific libraries
        print(f"Skipping GDF file (requires MNE): {file_path.name}")
        return None
    except Exception as e:
        print(f"Error with GDF file {file_path}: {e}")
        return None

def preprocess_bci_data(bci_folder_path, output_file):
    """
    Memory-efficient preprocessing of BCI data
    """
    bci_path = Path(bci_folder_path)
    
    # Check if BCI folder exists
    if not bci_path.exists():
        print(f"Error: BCI folder not found at {bci_folder_path}")
        return
    
    print(f"Processing BCI data from: {bci_folder_path}")
    print("=" * 50)
    
    # Collect all processed data
    all_datasets = {}
    total_files = 0
    processed_files = 0
    
    # Process each dataset folder
    for dataset_folder in bci_path.iterdir():
        if dataset_folder.is_dir():
            print(f"\nProcessing dataset: {dataset_folder.name}")
            dataset_data = []
            
            # Process files in this dataset
            for file_path in dataset_folder.iterdir():
                if file_path.is_file():
                    total_files += 1
                    print(f"  Processing: {file_path.name}")
                    
                    try:
                        data = None
                        
                        if file_path.suffix.lower() == '.txt':
                            data = load_bci_txt_file(file_path)
                        elif file_path.suffix.lower() == '.gdf':
                            data = process_gdf_file(file_path)
                        else:
                            print(f"    Skipping unsupported file: {file_path.name}")
                            continue
                        
                        if data is not None:
                            file_info = {
                                'filename': file_path.name,
                                'data': data,
                                'shape': data.shape,
                                'dataset': dataset_folder.name
                            }
                            dataset_data.append(file_info)
                            processed_files += 1
                            print(f"    Loaded: {data.shape}")
                            
                            # Clear memory
                            del data
                            gc.collect()
                        
                    except Exception as e:
                        print(f"    Error processing {file_path.name}: {e}")
                        continue
            
            if dataset_data:
                all_datasets[dataset_folder.name] = dataset_data
    
    # Save results
    print(f"\n" + "=" * 50)
    print(f"Processing Summary:")
    print(f"Total files found: {total_files}")
    print(f"Successfully processed: {processed_files}")
    print(f"Datasets found: {len(all_datasets)}")
    
    # Save to numpy format (more memory efficient than CSV)
    output_path = output_file.replace('.csv', '.npz').replace('.xlsx', '.npz')
    
    try:
        # Prepare data for saving
        save_dict = {}
        
        for dataset_name, files_data in all_datasets.items():
            print(f"\nDataset {dataset_name}:")
            for i, file_info in enumerate(files_data):
                key = f"{dataset_name}_{file_info['filename']}"
                save_dict[key] = file_info['data']
                print(f"  {file_info['filename']}: {file_info['shape']}")
        
        # Save all data
        np.savez_compressed(output_path, **save_dict)
        print(f"\n✅ Data saved to: {output_path}")
        
        # Print file size
        file_size = os.path.getsize(output_path)
        print(f"File size: {file_size / (1024*1024):.2f} MB")
        
    except Exception as e:
        print(f"Error saving data: {e}")
        
        # Fallback: save individual files
        print("Attempting to save individual dataset files...")
        for dataset_name, files_data in all_datasets.items():
            try:
                dataset_dict = {}
                for file_info in files_data:
                    key = file_info['filename'].replace('.txt', '').replace('.gdf', '')
                    dataset_dict[key] = file_info['data']
                
                dataset_output = f"{dataset_name}_processed.npz"
                np.savez_compressed(dataset_output, **dataset_dict)
                print(f"  Saved: {dataset_output}")
                
            except Exception as e2:
                print(f"  Failed to save {dataset_name}: {e2}")
    
    print(f"\n🎉 Processing complete!")
    return all_datasets

if __name__ == "__main__":
    # Set paths
    bci_folder = "BCI"  # Adjust path as needed
    output_file = "bci_preprocessed_data"  # Will be saved as .npz
    
    print("BCI Data Preprocessor - Memory Efficient Version")
    print("=" * 50)
    
    # Check available memory and warn user
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"Available memory: {memory.available / (1024**3):.1f} GB")
        if memory.percent > 80:
            print("⚠️  Warning: High memory usage detected")
    except ImportError:
        print("Install psutil for memory monitoring: pip install psutil")
    
    # Run preprocessing
    try:
        result = preprocess_bci_data(bci_folder, output_file)
        
        if result:
            print("\n✅ SUCCESS: Data preprocessing completed!")
            print("Next steps:")
            print("1. Use the processed .npz files with numpy.load()")
            print("2. Or run our optimized processor: python3 minimal_processor.py")
        else:
            print("\n❌ No data was processed successfully")
            
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
    except MemoryError:
        print("\n❌ Memory error - try processing smaller chunks")
        print("Alternative: Use minimal_processor.py which is more memory efficient")
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        print("Alternative: Use minimal_processor.py which handles errors better")