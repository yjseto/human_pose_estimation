import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataset import COCOSinglePersonDataset, get_transforms
from train import HeatmapPoseModel, INPUT_SIZE, HEATMAP_SIZE
from sklearn.metrics import confusion_matrix
import seaborn as sns

def load_trained_model(model_path='best_pose_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HeatmapPoseModel().to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model

def compute_detailed_metrics(model, dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Different PCK thresholds
    thresholds = [0.05, 0.1, 0.15, 0.2]  # 5%, 10%, 15%, 20% of image size
    
    # Initialize metrics dictionaries
    keypoint_metrics = {i: {
        'correct': {t: 0 for t in thresholds},
        'total': 0,
        'distances': [],
        'name': dataset.keypoint_names[i]
    } for i in range(17)}
    
    # Body part groups for grouped analysis
    body_parts = {
        'Face': ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear'],
        'Arms': ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'],
        'Legs': ['left_knee', 'right_knee', 'left_ankle', 'right_ankle'],
        'Torso': ['left_hip', 'right_hip']
    }
    
    total_visible = 0
    
for idx in range(len(dataset)):
    image, true_keypoints = dataset[idx]
    
    with torch.no_grad():
        input_image = image.unsqueeze(0).to(device)
        heatmaps = model(input_image)
        
        # Get predictions
        predicted_keypoints = []
        for h in heatmaps[0]:
            h = h.cpu().numpy()
            loc = np.unravel_index(h.argmax(), h.shape)
            y, x = loc[0], loc[1]
            x = x * INPUT_SIZE[1] / HEATMAP_SIZE[1]
            y = y * INPUT_SIZE[0] / HEATMAP_SIZE[0]
            predicted_keypoints.append([x, y])
        
        predicted_keypoints = torch.tensor(predicted_keypoints)
        
        # Compute metrics for each keypoint
        for kp_idx, (pred, true) in enumerate(zip(predicted_keypoints, true_keypoints)):
            if true[2] > 0:  # If keypoint is visible
                dist = torch.sqrt(((pred - true[:2]) ** 2).sum())
                normalized_dist = dist / INPUT_SIZE[0]
                
                # Update metrics
                keypoint_metrics[kp_idx]['distances'].append(normalized_dist.item())
                keypoint_metrics[kp_idx]['total'] += 1
                
                # Check against each threshold
                for t in thresholds:
                    if normalized_dist < t:
                        keypoint_metrics[kp_idx]['correct'][t] += 1
                
                total_visible += 1
    
    # Compute and format results
    results = {
        'Overall_PCK': {},
        'Per_Keypoint_PCK': {},
        'Body_Part_PCK': {},
        'MPJPE': {},
        'Distribution_Stats': {}
    }
    
    # Overall PCK at different thresholds
    for t in thresholds:
        total_correct = sum(metrics['correct'][t] for metrics in keypoint_metrics.values())
        results['Overall_PCK'][f'PCK@{t}'] = total_correct / total_visible * 100
    
    # Per-keypoint PCK and MPJPE
    for kp_idx, metrics in keypoint_metrics.items():
        kp_name = metrics['name']
        if metrics['total'] > 0:
            # PCK at different thresholds
            pck_values = {t: metrics['correct'][t] / metrics['total'] * 100 
                         for t in thresholds}
            results['Per_Keypoint_PCK'][kp_name] = pck_values
            
            # MPJPE (Mean Per Joint Position Error)
            mpjpe = np.mean(metrics['distances']) * INPUT_SIZE[0]  # Convert back to pixels
            results['MPJPE'][kp_name] = mpjpe
    
    # Body part group analysis
    for part_name, keypoints in body_parts.items():
        part_correct = {t: 0 for t in thresholds}
        part_total = 0
        part_distances = []
        
        for kp_name in keypoints:
            kp_idx = dataset.keypoint_names.index(kp_name)
            metrics = keypoint_metrics[kp_idx]
            
            for t in thresholds:
                part_correct[t] += metrics['correct'][t]
            part_total += metrics['total']
            part_distances.extend(metrics['distances'])
        
        if part_total > 0:
            results['Body_Part_PCK'][part_name] = {
                f'PCK@{t}': part_correct[t] / part_total * 100 
                for t in thresholds
            }
            results['MPJPE'][f'{part_name}_avg'] = np.mean(part_distances) * INPUT_SIZE[0]
    
    return results

def plot_metrics(results):
    # Plot PCK curves
    plt.figure(figsize=(15, 10))
    
    # Overall PCK
    plt.subplot(2, 2, 1)
    thresholds = [float(k.split('@')[1]) for k in results['Overall_PCK'].keys()]
    values = list(results['Overall_PCK'].values())
    plt.plot(thresholds, values, '-o')
    plt.title('Overall PCK Curve')
    plt.xlabel('Threshold')
    plt.ylabel('PCK (%)')
    plt.grid(True)
    
    # Body Part PCK
    plt.subplot(2, 2, 2)
    for part, metrics in results['Body_Part_PCK'].items():
        values = list(metrics.values())
        plt.plot(thresholds, values, '-o', label=part)
    plt.title('PCK by Body Part')
    plt.xlabel('Threshold')
    plt.ylabel('PCK (%)')
    plt.legend()
    plt.grid(True)
    
    # MPJPE by keypoint
    plt.subplot(2, 2, 3)
    keypoints = list(results['MPJPE'].keys())
    mpjpe_values = list(results['MPJPE'].values())
    plt.bar(range(len(keypoints)), mpjpe_values)
    plt.xticks(range(len(keypoints)), keypoints, rotation=45)
    plt.title('MPJPE by Keypoint')
    plt.ylabel('Mean Error (pixels)')
    plt.tight_layout()
    
    plt.show()

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
    
    # Compute detailed metrics
    print("\nComputing detailed metrics...")
    results = compute_detailed_metrics(model, val_dataset)
    
    # Print results
    print("\nOverall Performance:")
    print("-" * 50)
    for metric, value in results['Overall_PCK'].items():
        print(f"{metric}: {value:.2f}%")
    
    print("\nPerformance by Body Part:")
    print("-" * 50)
    for part, metrics in results['Body_Part_PCK'].items():
        print(f"\n{part}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.2f}%")
        print(f"  MPJPE: {results['MPJPE'][f'{part}_avg']:.2f} pixels")
    
    # Plot visualizations
    plot_metrics(results)