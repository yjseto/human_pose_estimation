import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import COCOSinglePersonDataset, get_transforms  # Import from your dataset.py
from tqdm import tqdm 

# Constants
INPUT_SIZE = (256, 256)
HEATMAP_SIZE = (64, 64)
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 0.001

# Check CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def create_target_heatmaps(keypoints, input_size=(256, 256), heatmap_size=(64, 64), sigma=2):
    """Convert keypoint coordinates to target heatmaps"""
    batch_size = keypoints.shape[0]
    num_keypoints = keypoints.shape[1]
    height, width = heatmap_size

    heatmaps = torch.zeros((batch_size, num_keypoints, height, width))

    for b in range(batch_size):
        for k in range(num_keypoints):
            if keypoints[b, k, 2] > 0:  # If keypoint is visible
                # Scale coordinates from input size to heatmap size
                x = keypoints[b, k, 0] * width / input_size[1]
                y = keypoints[b, k, 1] * height / input_size[0]

                # Create 2D gaussian
                xx, yy = torch.meshgrid(torch.arange(width), torch.arange(height), indexing='xy')
                xx = xx.float()
                yy = yy.float()

                gaussian = torch.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
                heatmaps[b, k] = gaussian

    return heatmaps

class HeatmapPoseModel(nn.Module):
    def __init__(self, num_keypoints=17):
        super().__init__()

        # Feature extraction backbone
        self.features = nn.Sequential(
            # Initial convolution (256 -> 128)
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 128 -> 64

            # Feature extraction blocks
            self._make_layer(64, 128, 3),
            self._make_layer(128, 256, 3),
            self._make_layer(256, 512, 3),
        )

        # Final layers to produce heatmaps
        self.heatmap_layers = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, num_keypoints, kernel_size=1)
        )

    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(blocks-1):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)  # Output size: B x 512 x 64 x 64
        x = self.heatmap_layers(x)  # Output size: B x num_keypoints x 64 x 64
        return x

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):  # Added epoch parameter
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}')
    for images, keypoints in pbar:
        images = images.to(device)
        target_heatmaps = create_target_heatmaps(keypoints,
                                               input_size=INPUT_SIZE,
                                               heatmap_size=HEATMAP_SIZE).to(device)

        optimizer.zero_grad()
        pred_heatmaps = model(images)
        loss = criterion(pred_heatmaps, target_heatmaps)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return running_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, keypoints in dataloader:
            images = images.to(device)
            target_heatmaps = create_target_heatmaps(keypoints,
                                                   input_size=INPUT_SIZE,
                                                   heatmap_size=HEATMAP_SIZE).to(device)

            pred_heatmaps = model(images)
            loss = criterion(pred_heatmaps, target_heatmaps)

            running_loss += loss.item()

    return running_loss / len(dataloader)

def train_model(train_loader, val_loader, num_epochs=NUM_EPOCHS):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HeatmapPoseModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)  # Added epoch
        val_loss = validate(model, val_loader, criterion, device)

        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_pose_model.pth')

        print('-' * 40)

    return model

if __name__ == "__main__":

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
    # Create datasets
    base_dir = "coco_dataset"
    
    train_dataset = COCOSinglePersonDataset(
        coco_path=os.path.join(base_dir, 'train2017'),
        annotation_path=os.path.join(base_dir, 'annotations', 'person_keypoints_train2017.json'),
        transform=get_transforms(is_train=True, input_size=INPUT_SIZE)
    )

    val_dataset = COCOSinglePersonDataset(
        coco_path=os.path.join(base_dir, 'val2017'),
        annotation_path=os.path.join(base_dir, 'annotations', 'person_keypoints_val2017.json'),
        transform=get_transforms(is_train=False, input_size=INPUT_SIZE)
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        pin_memory=True,
        num_workers=6  # Adjust based on your CPU
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=6,
        pin_memory=True
    )

    # Train the model
    print("Starting training...")
    model = train_model(train_loader, val_loader)