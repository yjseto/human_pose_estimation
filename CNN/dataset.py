import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add CUDA check
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class COCOSinglePersonDataset(Dataset):
    def __init__(self, coco_path, annotation_path, transform=None, min_keypoints=5):
        self.coco_path = coco_path
        self.coco = COCO(annotation_path)
        self.transform = transform
        self.min_keypoints = min_keypoints

        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]

        self.colors = {
            'nose': 'red',
            'eyes': 'blue',
            'ears': 'green',
            'shoulders': 'yellow',
            'elbows': 'purple',
            'wrists': 'orange',
            'hips': 'cyan',
            'knees': 'magenta',
            'ankles': 'brown'
        }

        self.skeleton = [
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_elbow'),
            ('right_shoulder', 'right_elbow'),
            ('left_elbow', 'left_wrist'),
            ('right_elbow', 'right_wrist'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            ('left_hip', 'left_knee'),
            ('right_hip', 'right_knee'),
            ('left_knee', 'left_ankle'),
            ('right_knee', 'right_ankle')
        ]

        # Initialize lists to store valid image and annotation IDs
        self.coco_path = coco_path
        self.coco = COCO(annotation_path)
        self.transform = transform
        self.min_keypoints = min_keypoints

        # Initialize lists to store valid image and annotation IDs
        self.img_ids = []
        self.ann_ids = []

        # Get person category ID
        self.cat_ids = self.coco.getCatIds(catNms=['person'])
        all_img_ids = self.coco.getImgIds(catIds=self.cat_ids)

        # Filter for single person images with good keypoints
        for img_id in all_img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids)
            if len(ann_ids) == 1:  # Single person
                ann = self.coco.loadAnns(ann_ids[0])[0]
                num_visible_kpts = sum(1 for v in ann['keypoints'][2::3] if v > 0)
                if num_visible_kpts >= min_keypoints:
                    self.img_ids.append(img_id)
                    self.ann_ids.append(ann_ids[0])

        print(f"Found {len(self.img_ids)} single person images with good keypoints")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        ann_id = self.ann_ids[idx]

        # Load image
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.coco_path, img_info['file_name'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load annotation and get keypoints
        ann = self.coco.loadAnns(ann_id)[0]
        keypoints = np.array(ann['keypoints'], dtype=np.float32).reshape(-1, 3)

        if self.transform:
            # Store visibility for later
            visibility = keypoints[:, 2:3].copy()
            
            # Prepare keypoints for transformation
            keypoints_for_transform = []
            visible_indices = []
            
            # Only transform visible keypoints
            for i, (x, y, v) in enumerate(keypoints):
                if v > 0:
                    keypoints_for_transform.append([x, y])
                    visible_indices.append(i)
            
            # Apply transformation
            transformed = self.transform(
                image=image,
                keypoints=keypoints_for_transform if keypoints_for_transform else [[0, 0]]  # Dummy point if none visible
            )
            
            # Get transformed image and keypoints
            image = transformed['image']
            
            # Create new keypoints array with same shape as original
            transformed_keypoints = np.zeros((17, 2), dtype=np.float32)
            
            if keypoints_for_transform:  # If there were visible keypoints
                transformed_points = np.array(transformed['keypoints'])
                for idx, transformed_idx in enumerate(visible_indices):
                    transformed_keypoints[transformed_idx] = transformed_points[idx]
            
            # Combine with visibility
            keypoints = np.concatenate([transformed_keypoints, visibility], axis=1)

        return image, torch.tensor(keypoints, dtype=torch.float32)

def get_transforms(is_train=True, input_size=(256, 256)):
    """
    Create transform pipeline with proper keypoint handling
    """
    if is_train:
        transform = A.Compose([
            A.Resize(height=input_size[0], width=input_size[1], always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    else:
        transform = A.Compose([
            A.Resize(height=input_size[0], width=input_size[1], always_apply=True),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    return transform

# Test function
def test_dataset_sample(dataset, idx=0):
    """
    Test a single sample from the dataset
    """
    try:
        image, keypoints = dataset[idx]
        print(f"Image shape: {image.shape}")
        print(f"Keypoints shape: {keypoints.shape}")
        print(f"\nNumber of visible keypoints: {(keypoints[:, 2] > 0).sum()}")
        print("\nSample keypoint coordinates:")
        for i, (x, y, v) in enumerate(keypoints):
            if v > 0:
                print(f"{dataset.keypoint_names[i]}: ({x:.1f}, {y:.1f})")
                
        # Visualize
        visualize_sample(dataset, idx, show_labels=True)
        
    except Exception as e:
        print(f"Error testing sample: {e}")


# Test the dataset
def test_dataset(dataset, num_samples=1):
    """
    Test the dataset by loading and visualizing samples
    """
    for i in range(num_samples):
        try:
            image, keypoints = dataset[i]
            print(f"Sample {i}:")
            print(f"Image shape: {image.shape}")
            print(f"Keypoints shape: {keypoints.shape}")
            print("Keypoint statistics:")
            print(f"Min: {keypoints.min():.2f}")
            print(f"Max: {keypoints.max():.2f}")
            print(f"Mean: {keypoints.mean():.2f}")
            print("-" * 40)

            # Visualize the sample
            visualize_sample(dataset, i)

        except Exception as e:
            print(f"Error processing sample {i}: {e}")

def visualize_sample(dataset, idx, show_labels=False):
    """Visualize a sample with keypoints"""
    image, keypoints = dataset[idx]

    if torch.is_tensor(image):
        # Convert tensor to numpy and denormalize
        image = image.numpy().transpose(1, 2, 0)
        image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        image = np.clip(image, 0, 1)

    # Create figure
    plt.figure(figsize=(12, 8))
    plt.imshow(image)

    # Plot keypoints
    for i, (x, y, v) in enumerate(keypoints):
        if v > 0:  # Keypoint is visible
            keypoint_name = dataset.keypoint_names[i]
            color = dataset.colors.get(keypoint_name.split('_')[-1], 'white')
            plt.plot(x, y, 'o', color=color, markersize=8)

            if show_labels:
                plt.text(x+5, y+5, keypoint_name, color='white',
                        bbox=dict(facecolor='black', alpha=0.7))

    # Draw skeleton
    for (kp1_name, kp2_name) in dataset.skeleton:
        idx1 = dataset.keypoint_names.index(kp1_name)
        idx2 = dataset.keypoint_names.index(kp2_name)
        if keypoints[idx1][2] > 0 and keypoints[idx2][2] > 0:
            plt.plot([keypoints[idx1][0], keypoints[idx2][0]],
                    [keypoints[idx1][1], keypoints[idx2][1]],
                    'w-', alpha=0.7)

    plt.title(f'Sample {idx}')
    plt.axis('off')
    plt.show()



if __name__ == "__main__":
    # Update paths for local directory
    base_dir = "coco_dataset"  # Or whatever directory you downloaded COCO to
    
    train_dataset = COCOSinglePersonDataset(
        coco_path=os.path.join(base_dir, 'train2017'),
        annotation_path=os.path.join(base_dir, 'annotations', 'person_keypoints_train2017.json'),
        transform=get_transforms(is_train=True, input_size=(256, 256))
    )

    val_dataset = COCOSinglePersonDataset(
        coco_path=os.path.join(base_dir, 'val2017'),
        annotation_path=os.path.join(base_dir, 'annotations', 'person_keypoints_val2017.json'),
        transform=get_transforms(is_train=False, input_size=(256, 256))
    )

    print("\nTraining samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    # Test a few samples
    print("\nTesting dataset...")
    test_dataset(train_dataset, num_samples=2)  