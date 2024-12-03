import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataset import COCOSinglePersonDataset, get_transforms
from train import HeatmapPoseModel, INPUT_SIZE, HEATMAP_SIZE

def load_trained_model(model_path='best_pose_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HeatmapPoseModel().to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model

def visualize_predictions(model, dataset, num_samples=5, save_dir='prediction_results'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create directory for saving results if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Define colors for different body parts
    colors = {
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

    # Define connections for skeleton
    skeleton = [
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

    for idx in range(min(num_samples, len(dataset))):
        image, true_keypoints = dataset[idx]
        
        # Get model prediction
        with torch.no_grad():
            input_image = image.unsqueeze(0).to(device)
            heatmaps = model(input_image)
            
            # Convert heatmaps to keypoint coordinates
            predicted_keypoints = []
            confidence_scores = []
            for h in heatmaps[0]:  # First image in batch
                h = h.cpu().numpy()
                confidence = h.max()
                loc = np.unravel_index(h.argmax(), h.shape)
                y, x = loc[0], loc[1]
                
                # Scale coordinates back to original size
                x = x * INPUT_SIZE[1] / HEATMAP_SIZE[1]
                y = y * INPUT_SIZE[0] / HEATMAP_SIZE[0]
                predicted_keypoints.append([x, y])
                confidence_scores.append(confidence)
            
            predicted_keypoints = torch.tensor(predicted_keypoints)

        # Create figure
        plt.figure(figsize=(20, 10))
        
        # Plot original image with true keypoints
        plt.subplot(1, 2, 1)
        if torch.is_tensor(image):
            img_np = image.numpy().transpose(1, 2, 0)
            img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_np = np.clip(img_np, 0, 1)
        plt.imshow(img_np)
        
        # Plot true keypoints and skeleton
        for i, (x, y, v) in enumerate(true_keypoints):
            if v > 0:  # If keypoint is visible
                keypoint_name = dataset.keypoint_names[i]
                color = colors.get(keypoint_name.split('_')[-1], 'white')
                plt.plot(x, y, 'o', color=color, markersize=8, label=f'True {keypoint_name}')
        
        # Draw true skeleton
        for (kp1_name, kp2_name) in skeleton:
            idx1 = dataset.keypoint_names.index(kp1_name)
            idx2 = dataset.keypoint_names.index(kp2_name)
            if true_keypoints[idx1][2] > 0 and true_keypoints[idx2][2] > 0:
                plt.plot([true_keypoints[idx1][0], true_keypoints[idx2][0]],
                        [true_keypoints[idx1][1], true_keypoints[idx2][1]],
                        'white', alpha=0.6)
        
        plt.title('Ground Truth')
        
        # Plot original image with predicted keypoints
        plt.subplot(1, 2, 2)
        plt.imshow(img_np)
        
        # Plot predicted keypoints and skeleton
        for i, (x, y) in enumerate(predicted_keypoints):
            keypoint_name = dataset.keypoint_names[i]
            color = colors.get(keypoint_name.split('_')[-1], 'white')
            conf = confidence_scores[i]
            plt.plot(x, y, 'o', color=color, markersize=8, 
                    alpha=min(1.0, conf * 2.0),  # Adjust visibility based on confidence
                    label=f'Pred {keypoint_name} ({conf:.2f})')
        
        # Draw predicted skeleton
        for (kp1_name, kp2_name) in skeleton:
            idx1 = dataset.keypoint_names.index(kp1_name)
            idx2 = dataset.keypoint_names.index(kp2_name)
            conf = min(confidence_scores[idx1], confidence_scores[idx2])
            plt.plot([predicted_keypoints[idx1][0], predicted_keypoints[idx2][0]],
                    [predicted_keypoints[idx1][1], predicted_keypoints[idx2][1]],
                    'white', alpha=min(0.6, conf))
        
        plt.title('Model Predictions')
        
        # Save the figure
        plt.savefig(os.path.join(save_dir, f'prediction_{idx}.png'))
        plt.close()
        
        print(f"Saved prediction visualization {idx+1}/{num_samples}")

if __name__ == "__main__":
    # Load validation dataset
    base_dir = "coco_dataset"
    val_dataset = COCOSinglePersonDataset(
        coco_path=os.path.join(base_dir, 'val2017'),
        annotation_path=os.path.join(base_dir, 'annotations', 'person_keypoints_val2017.json'),
        transform=get_transforms(is_train=False, input_size=INPUT_SIZE)
    )
    
    # Load trained model
    model = load_trained_model('best_pose_model.pth')
    
    # Visualize predictions
    print("\nGenerating prediction visualizations...")
    visualize_predictions(model, val_dataset, num_samples=10)  # Change number of samples as needed