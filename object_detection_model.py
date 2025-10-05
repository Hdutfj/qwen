"""
Object Detection Model for Home Objects - YOLO-style Implementation
This file contains an improved model architecture and training pipeline for object detection
with bounding boxes instead of just image classification.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from PIL import Image
import json
import warnings
from typing import Tuple, List, Optional, Dict
import time
import logging
from pathlib import Path
import glob
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Define the classes for home objects
HOME_OBJECTS = [
    'chair', 'sofa', 'bed', 'dining_table', 'toilet', 
    'tv', 'laptop', 'mouse', 'oven', 'toaster', 
    'refrigerator', 'book', 'clock', 'vase', 'window'
]

NUM_CLASSES = len(HOME_OBJECTS)

class YOLOHead(nn.Module):
    """YOLO-style detection head"""
    def __init__(self, in_channels, num_classes, num_anchors=3):
        super(YOLOHead, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        out_channels = num_anchors * (5 + num_classes)  # 5 = x, y, w, h, confidence
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, out_channels, kernel_size=1)
        )
    
    def forward(self, x):
        return self.conv(x)

class YOLOBackbone(nn.Module):
    """YOLO-style backbone with feature pyramid"""
    def __init__(self, num_classes, num_anchors=3):
        super(YOLOBackbone, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Feature extraction backbone
        # Input: 416x416x3
        self.backbone = nn.Sequential(
            # Block 1: 416x416 -> 208x208
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 416 -> 208
            
            # Block 2: 208x208 -> 104x104
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 208 -> 104
            
            # Block 3: 104x104 -> 52x52
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 104 -> 52
            
            # Block 4: 52x52 -> 26x26
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 52 -> 26
            
            # Block 5: 26x26 -> 13x13
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 26 -> 13
        )
        
        # YOLO head
        self.yolo_head = YOLOHead(512, num_classes, num_anchors)
        
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Apply the detection head
        output = self.yolo_head(features)
        
        return output

class HomeObjectDetectionModel(nn.Module):
    """
    Object detection model for home objects using YOLO-style architecture
    """
    def __init__(self, num_classes=NUM_CLASSES, num_anchors=3):
        super(HomeObjectDetectionModel, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.grid_size = 13  # For 416 / 32 = 13
        
        # Define anchors (normalized w, h) - can be tuned based on dataset
        self.anchors = torch.tensor([[0.1, 0.1], [0.3, 0.3], [0.5, 0.5]])
        
        # Use YOLO backbone
        self.backbone = YOLOBackbone(num_classes, num_anchors)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.backbone(x)

    def predict(self, x, conf_thresh=0.5, nms_thresh=0.4):
        """
        Perform object detection and return bounding boxes
        """
        # Forward pass
        outputs = self.forward(x)
        
        # Process outputs to get bounding boxes
        batch_size, _, grid_h, grid_w = outputs.shape  # grid_h, grid_w should be 13, 13
        anchor_step = 5 + self.num_classes  # x, y, w, h, conf, class_probs
        
        # Reshape to separate anchors, boxes, class scores
        outputs = outputs.view(batch_size, self.num_anchors, anchor_step, grid_h, grid_w)
        outputs = outputs.permute(0, 1, 3, 4, 2).contiguous()
        
        # Extract box coordinates and confidence
        box_xy = torch.sigmoid(outputs[..., :2])  # x, y (center coordinates)
        box_wh = torch.exp(outputs[..., 2:4])  # w, h (exponential for scale)
        conf = torch.sigmoid(outputs[..., 4])     # confidence
        cls_scores = torch.softmax(outputs[..., 5:], dim=-1)  # class scores
        
        # Apply anchors to wh
        anchors = self.anchors.to(box_wh.device)
        box_wh *= anchors.view(1, self.num_anchors, 1, 1, 2)
        
        # Apply confidence threshold
        mask = conf > conf_thresh
        
        # Collect detections
        detections = []
        for b in range(batch_size):
            batch_detections = []
            for a in range(self.num_anchors):
                for i in range(grid_h):
                    for j in range(grid_w):
                        if mask[b, a, i, j]:
                            # Calculate actual box coordinates (normalized 0-1)
                            x = (box_xy[b, a, i, j, 0] + j) / grid_w
                            y = (box_xy[b, a, i, j, 1] + i) / grid_h
                            w = box_wh[b, a, i, j, 0]
                            h = box_wh[b, a, i, j, 1]
                            
                            # Get class prediction
                            cls_prob, cls_id = torch.max(cls_scores[b, a, i, j], dim=0)
                            
                            confidence = conf[b, a, i, j]
                            
                            batch_detections.append([
                                x.item(), y.item(), w.item(), h.item(),
                                confidence.item(), cls_id.item()
                            ])
            
            # Apply NMS
            batch_detections = self._nms(batch_detections, nms_thresh)
            detections.append(batch_detections)
        
        return detections

    def _nms(self, boxes, nms_thresh):
        """Non-Maximum Suppression"""
        if len(boxes) == 0:
            return []
        
        # Convert to tensors
        boxes_tensor = torch.tensor(boxes)
        x1 = boxes_tensor[:, 0] - boxes_tensor[:, 2] / 2  # xmin
        y1 = boxes_tensor[:, 1] - boxes_tensor[:, 3] / 2  # ymin
        x2 = boxes_tensor[:, 0] + boxes_tensor[:, 2] / 2  # xmax
        y2 = boxes_tensor[:, 1] + boxes_tensor[:, 3] / 2  # ymax
        scores = boxes_tensor[:, 4]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort(descending=True)
        
        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                i = order.item()
                keep.append(i)
                break
            
            i = order[0].item()
            keep.append(i)
            
            # Get intersection coordinates
            xx1 = torch.max(x1[i], x1[order[1:]])
            yy1 = torch.max(y1[i], y1[order[1:]])
            xx2 = torch.min(x2[i], x2[order[1:]])
            yy2 = torch.min(y2[i], y2[order[1:]])
            
            # Compute intersection area
            w = torch.clamp(xx2 - xx1, min=0)
            h = torch.clamp(yy2 - yy1, min=0)
            inter = w * h
            
            # Compute IoU
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            # Keep boxes with IoU less than threshold
            idx = (iou <= nms_thresh).nonzero().squeeze()
            if idx.numel() == 0:
                break
            order = order[idx + 1]
        
        return [boxes[i] for i in keep]

def draw_bounding_boxes(image, detections, class_names=HOME_OBJECTS):
    """
    Draw bounding boxes on an image
    """
    from PIL import ImageDraw, ImageFont
    
    # Convert to PIL image if it's a tensor
    if isinstance(image, torch.Tensor):
        # Denormalize and convert to PIL
        image = image.cpu().detach()
        image = image.permute(1, 2, 0)  # CHW to HWC
        # Denormalize assuming ImageNet normalization
        image = image * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
        image = torch.clamp(image, 0, 1)
        image = (image * 255).byte()
        image = Image.fromarray(image.numpy())
    
    draw = ImageDraw.Draw(image)
    
    # Draw each detection
    for det in detections:
        x_center, y_center, width, height, conf, cls_id = det
        x1 = int((x_center - width/2) * image.width)
        y1 = int((y_center - height/2) * image.height)
        x2 = int((x_center + width/2) * image.width)
        y2 = int((y_center + height/2) * image.height)
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        
        # Draw label
        label = f"{class_names[int(cls_id)]}: {conf:.2f}"
        draw.text((x1, y1 - 10), label, fill="red")
    
    return image

class HomeObjectsDetectionDataset(Dataset):
    """
    Custom dataset for home object detection with bounding box annotations
    Loads annotations from .txt files if available, otherwise creates synthetic
    """
    def __init__(self, images_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.transform = transform
        
        # Get all image files
        self.image_paths = []
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        for ext in valid_extensions:
            self.image_paths.extend(self.images_dir.glob(f"**/*{ext}"))  # Include subdirs
        
        self.image_paths = sorted(self.image_paths)  # For consistency
        logger.info(f"Found {len(self.image_paths)} images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Error opening {img_path}: {e}. Using dummy image.")
            image = Image.new('RGB', (416, 416), color='gray')
        
        bounding_boxes = []
        
        # Try to load annotation file
        ann_path = img_path.with_suffix('.txt')
        if ann_path.exists():
            try:
                with open(ann_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cls_id = int(parts[0])
                            x = float(parts[1])
                            y = float(parts[2])
                            w = float(parts[3])
                            h = float(parts[4])
                            if 0 <= cls_id < NUM_CLASSES and 0 < w <= 1 and 0 < h <= 1:
                                bounding_boxes.append([x, y, w, h, cls_id])
            except Exception as e:
                logger.warning(f"Error reading annotation {ann_path}: {e}")
        
        # If no annotations found, create synthetic for demonstration
        if not bounding_boxes:
            num_objects = random.randint(1, 3)
            for _ in range(num_objects):
                cls_id = random.randint(0, NUM_CLASSES - 1)
                w = random.uniform(0.1, 0.4)
                h = random.uniform(0.1, 0.4)
                x = random.uniform(w / 2, 1 - w / 2)
                y = random.uniform(h / 2, 1 - h / 2)
                bounding_boxes.append([x, y, w, h, cls_id])
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            default_transform = transforms.Compose([
                transforms.Resize((416, 416)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image = default_transform(image)
        
        # Return image and list of boxes (variable length)
        boxes = torch.tensor(bounding_boxes, dtype=torch.float32) if bounding_boxes else torch.zeros((0, 5))
        return image, boxes

def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    boxes = [item[1] for item in batch]
    return images, boxes

def get_detection_transforms():
    """Get data transforms for object detection"""
    train_transform = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def yolo_loss(predictions, targets, model):
    """
    YOLO loss function (improved to handle variable boxes and proper components)
    """
    device = predictions.device
    lambda_coord = 5.0
    lambda_noobj = 0.5
    batch_size, _, grid_h, grid_w = predictions.shape
    num_anchors = model.num_anchors
    num_classes = model.num_classes
    anchors = model.anchors.to(device)
    
    # Reshape predictions
    anchor_step = 5 + num_classes
    pred = predictions.view(batch_size, num_anchors, anchor_step, grid_h, grid_w)
    pred = pred.permute(0, 1, 3, 4, 2).contiguous()
    
    # Extract components
    pred_xy = torch.sigmoid(pred[..., :2])  # x, y
    pred_wh = pred[..., 2:4]                # w, h (log scale)
    pred_conf = torch.sigmoid(pred[..., 4]) # confidence
    pred_cls = torch.softmax(pred[..., 5:], dim=-1)  # class probabilities
    
    # Initialize targets
    obj_mask = torch.zeros(batch_size, num_anchors, grid_h, grid_w, device=device)
    noobj_mask = torch.ones(batch_size, num_anchors, grid_h, grid_w, device=device)
    target_xy = torch.zeros(batch_size, num_anchors, grid_h, grid_w, 2, device=device)
    target_wh = torch.zeros(batch_size, num_anchors, grid_h, grid_w, 2, device=device)
    target_cls = torch.zeros(batch_size, num_anchors, grid_h, grid_w, dtype=torch.long, device=device) - 1  # -1 for no obj
    
    for b in range(batch_size):
        if len(targets[b]) == 0:
            continue
        gt_boxes = targets[b].to(device)
        for gt in gt_boxes:
            gt_x, gt_y, gt_w, gt_h, gt_cls = gt
            if gt_w <= 0 or gt_h <= 0:
                continue
            
            cell_x = int(gt_x * grid_w)
            cell_y = int(gt_y * grid_h)
            gt_offset_x = gt_x * grid_w - cell_x
            gt_offset_y = gt_y * grid_h - cell_y
            
            # Find best anchor
            best_iou = 0
            best_a = 0
            for a in range(num_anchors):
                anchor_w, anchor_h = anchors[a]
                min_w = min(gt_w, anchor_w)
                min_h = min(gt_h, anchor_h)
                inter = min_w * min_h
                union = gt_w * gt_h + anchor_w * anchor_h - inter + 1e-16
                iou = inter / union
                if iou > best_iou:
                    best_iou = iou
                    best_a = a
            
            # Assign if better than threshold or always assign to best
            obj_mask[b, best_a, cell_y, cell_x] = 1
            noobj_mask[b, best_a, cell_y, cell_x] = 0
            target_xy[b, best_a, cell_y, cell_x] = torch.tensor([gt_offset_x, gt_offset_y], device=device)
            target_wh[b, best_a, cell_y, cell_x] = torch.tensor([torch.log(gt_w / anchors[best_a][0] + 1e-16),
                                                                  torch.log(gt_h / anchors[best_a][1] + 1e-16)], device=device)
            target_cls[b, best_a, cell_y, cell_x] = int(gt_cls)
    
    # Losses
    obj_indices = obj_mask > 0
    noobj_indices = noobj_mask > 0
    
    # Coord losses
    loss_xy = lambda_coord * torch.mean(obj_mask * ((pred_xy - target_xy) ** 2).sum(-1))
    loss_wh = lambda_coord * torch.mean(obj_mask * ((pred_wh - target_wh) ** 2).sum(-1))
    
    # Confidence losses
    loss_conf_obj = torch.mean(obj_mask * ((pred_conf - 1) ** 2))
    loss_conf_noobj = lambda_noobj * torch.mean(noobj_mask * (pred_conf ** 2))
    
    # Class loss (using cross-entropy like)
    loss_cls = 0
    if (target_cls >= 0).any():
        valid_cls = target_cls >= 0
        selected_pred_cls = pred_cls[valid_cls]
        selected_target_cls = target_cls[valid_cls]
        loss_cls = -torch.mean(torch.log(selected_pred_cls[torch.arange(selected_pred_cls.size(0)), selected_target_cls] + 1e-16))
    
    total_loss = loss_xy + loss_wh + loss_conf_obj + loss_conf_noobj + loss_cls
    return total_loss

def train_detection_model(model, train_loader, val_loader, num_epochs=20, learning_rate=0.001, save_path='best_detection_model.pth'):
    """
    Train the object detection model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model.to(device)
    
    # Loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Training metrics
    train_losses = []
    
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            # Targets are list of tensors, move to device
            targets = [t.to(device) for t in targets]
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = yolo_loss(outputs, targets, model)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        logger.info(f"  Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, save_path.replace('.pth', f'_epoch_{epoch+1}.pth'))
        
        # Update learning rate
        scheduler.step(avg_loss)
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'class_names': HOME_OBJECTS,
        'anchors': model.anchors
    }, save_path)
    
    logger.info(f"Training completed! Model saved as {save_path}")
    return train_losses

def create_synthetic_detection_dataset(dataset_dir: str, num_images: int = 200, img_size=416):
    """
    Create synthetic dataset for object detection with bounding box annotations
    Now also creates .txt annotation files in YOLO format
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        dataset_path = Path(dataset_dir)
        dataset_path.mkdir(exist_ok=True)
        
        logger.info(f"Creating synthetic detection dataset with {num_images} images...")
        
        for i in range(num_images):
            # Create a synthetic image
            img = Image.new('RGB', (img_size, img_size), 
                            color=(random.randint(50, 200), 
                                   random.randint(50, 200), 
                                   random.randint(50, 200)))
            
            draw = ImageDraw.Draw(img)
            
            # Add random objects/bounding boxes
            num_objects = random.randint(1, 3)
            annotations = []
            
            for _ in range(num_objects):
                # Random bounding box (pixels)
                x1 = random.randint(50, img_size - 50)
                y1 = random.randint(50, img_size - 50)
                x2 = min(x1 + random.randint(30, 150), img_size - 10)
                y2 = min(y1 + random.randint(30, 150), img_size - 10)
                
                # Random color for the object
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                
                # Draw rectangle as "object"
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                
                # Random class
                cls_id = random.randint(0, len(HOME_OBJECTS)-1)
                class_name = HOME_OBJECTS[cls_id]
                
                # Draw label
                try:
                    font = ImageFont.load_default()
                    draw.text((x1, y1 - 10), class_name, fill=color, font=font)
                except:
                    draw.text((x1, y1 - 10), class_name, fill=color)
                
                # Compute normalized YOLO format
                x_center = ((x1 + x2) / 2) / img_size
                y_center = ((y1 + y2) / 2) / img_size
                width = (x2 - x1) / img_size
                height = (y2 - y1) / img_size
                
                annotations.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            # Add random shapes for variety (no annotations for noise)
            for _ in range(random.randint(2, 6)):
                shape_type = random.choice(['rectangle', 'ellipse'])
                x1 = random.randint(0, img_size - 50)
                y1 = random.randint(0, img_size - 50)
                x2 = min(x1 + random.randint(20, 60), img_size)
                y2 = min(y1 + random.randint(20, 60), img_size)
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                
                if shape_type == 'rectangle':
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                else:
                    draw.ellipse([x1, y1, x2, y2], outline=color, width=2)
            
            # Save image
            img_path = dataset_path / f"synthetic_detection_{i:04d}.jpg"
            img.save(img_path, 'JPEG', quality=85)
            
            # Save annotations
            ann_path = img_path.with_suffix('.txt')
            with open(ann_path, 'w') as f:
                f.write('\n'.join(annotations))
        
        logger.info(f"Synthetic detection dataset created at {dataset_path} with annotations")
        return True
        
    except ImportError:
        logger.error("PIL not available for creating synthetic images")
        return False
    except Exception as e:
        logger.error(f"Error creating synthetic dataset: {e}")
        return False

def main():
    logger.info("Home Objects Detection Model - YOLO-style Implementation")
    logger.info("=" * 60)
    
    # Configuration
    dataset_dir = "./detection_dataset"
    batch_size = 8  # Smaller batch size for detection
    num_epochs = 10
    learning_rate = 0.001
    
    try:
        # Initialize model
        model = HomeObjectDetectionModel(num_classes=NUM_CLASSES)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model initialized with {total_params:,} parameters")
        
        # Create synthetic dataset if directory empty
        dataset_path = Path(dataset_dir)
        if not dataset_path.exists() or len(list(dataset_path.glob('*.jpg'))) == 0:
            logger.info("Creating synthetic detection dataset...")
            success = create_synthetic_detection_dataset(dataset_dir, num_images=200)
            if not success:
                logger.error("Failed to create synthetic dataset")
                return
        
        # Get transforms
        train_transform, val_transform = get_detection_transforms()
        
        # Load datasets
        train_dataset = HomeObjectsDetectionDataset(dataset_dir, transform=train_transform)
        val_dataset = HomeObjectsDetectionDataset(dataset_dir, transform=val_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
        
        logger.info(f"Dataset loaded successfully:")
        logger.info(f"  Training samples: {len(train_dataset)}")
        logger.info(f"  Batch size: {batch_size}")
        
        # Train model
        train_losses = train_detection_model(
            model, train_loader, val_loader, num_epochs, learning_rate
        )
        
        # Save final model
        model_save_path = Path('home_objects_detection_model.pth')
        torch.save(model.state_dict(), model_save_path)
        logger.info(f"Final model saved as '{model_save_path}'")
        
        # Also create a copy for API compatibility
        expected_path = Path('home_objects_cnn.pth')
        if not expected_path.exists():
            import shutil
            shutil.copy(model_save_path, expected_path)
            logger.info(f"Also saved model as '{expected_path}' for API compatibility")
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()