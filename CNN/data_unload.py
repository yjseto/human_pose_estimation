import os
import subprocess
import time
import urllib.request
import zipfile

def setup_coco(base_dir="coco_dataset"):
    """Complete setup for COCO dataset with both training and validation sets"""
    
    print("=== Setting up COCO dataset ===\n")
    
    # 1. Create base directory
    print("1. Creating dataset directory...")
    os.makedirs(base_dir, exist_ok=True)
    os.chdir(base_dir)
    
    # 2. Install required packages
    print("\n2. Installing required packages...")
    try:
        subprocess.check_call(['pip', 'install', 'pycocotools', 'albumentations'])
    except subprocess.CalledProcessError as e:
        print(f"Failed to install packages: {str(e)}")
        return False
    
    # 3. Download datasets
    print("\n3. Downloading datasets...")
    files_to_download = {
        'train2017.zip': 'http://images.cocodataset.org/zips/train2017.zip',
        'val2017.zip': 'http://images.cocodataset.org/zips/val2017.zip',
        'annotations_trainval2017.zip': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    }
    
    for filename, url in files_to_download.items():
        if os.path.exists(filename):
            print(f"  ✓ {filename} already exists, skipping...")
            continue
            
        print(f"  Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, filename)
        except Exception as e:
            print(f"  ✗ Failed to download {filename}: {str(e)}")
            return False
    
    # 4. Unzip files
    print("\n4. Extracting files...")
    for filename in files_to_download.keys():
        print(f"  Extracting {filename}...")
        try:
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall('.')
        except Exception as e:
            print(f"  ✗ Failed to extract {filename}: {str(e)}")
            return False
    
    # 5. Verify setup
    print("\n5. Verifying setup...")
    required_paths = [
        'train2017',
        'val2017',
        'annotations',
        'annotations/person_keypoints_train2017.json',
        'annotations/person_keypoints_val2017.json'
    ]
    
    all_exists = True
    for path in required_paths:
        if os.path.exists(path):
            print(f"  ✓ Found {path}")
        else:
            print(f"  ✗ Missing {path}")
            all_exists = False
    
    if all_exists:
        print("\n=== Setup completed successfully! ===")
        print("\nDataset statistics:")
        try:
            train_images = len(os.listdir('train2017'))
            val_images = len(os.listdir('val2017'))
            print(f"Training images: {train_images:,}")
            print(f"Validation images: {val_images:,}")
        except Exception as e:
            print("Could not count images:", str(e))
    else:
        print("\n✗ Setup incomplete. Please check the errors above.")
    
    return all_exists

if __name__ == "__main__":
    setup_coco()