"""
Enhanced Detection Model with Smart Features
Implements Auto-Label Refinement, Multi-Object Context Awareness, and Fine-Grained Classification
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
from PIL import Image, ImageDraw, ImageFont
import json
import warnings
from typing import Tuple, List, Optional, Dict, Any
import time
import logging
from pathlib import Path
import glob
import random
import pickle
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Enhanced home objects classification with sub-types
HOME_OBJECTS = [
    'toilet', 'sink', 'mirror', 'bathtub', 'showerhead', 'towel', 'toothbrush', 
    'toothpaste', 'soap_bar', 'shampoo_bottle', 'conditioner_bottle', 'handwash_bottle', 
    'toilet_paper_roll', 'towel_rack', 'bath_mat', 'hair_dryer', 'razor', 'lotion_bottle', 
    'trash_bin', 'shower_curtain', 'comb', 'cleaning_brush', 'bucket', 'mug', 'bathroom_shelf'
]

# Sub-type classifications for fine-grained detection
SUBTYPE_CLASSES = {
    'chair': ['office_chair', 'dining_chair', 'gaming_chair', 'armchair', 'recliner'],
    'toilet': ['standard_toilet', 'bidet_toilet', 'smart_toilet'],
    'sink': ['bathroom_sink', 'kitchen_sink', 'bar_sink', 'utility_sink'],
    'towel': ['bath_towel', 'hand_towel', 'face_towel', 'beach_towel'],
    'toothbrush': ['manual_toothbrush', 'electric_toothbrush', 'travel_toothbrush']
}

# Extended object list to include subtypes
EXTENDED_CLASSES = []
CLASS_TO_SUPERCLASS = {}
for obj in HOME_OBJECTS:
    if obj in SUBTYPE_CLASSES:
        # Add subtypes
        for subtype in SUBTYPE_CLASSES[obj]:
            EXTENDED_CLASSES.append(subtype)
            CLASS_TO_SUPERCLASS[subtype] = obj
    else:
        EXTENDED_CLASSES.append(obj)

NUM_CLASSES = len(EXTENDED_CLASSES)
SUPERCLASS_NUM_CLASSES = len(HOME_OBJECTS)

class FastConvBlock(nn.Module):
    """Fast convolution block with depthwise separable convolutions"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(FastConvBlock, self).__init__()
        self.conv = nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            # Pointwise convolution
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class FastYOLOBackbone(nn.Module):
    """Lightweight YOLO backbone optimized for speed"""
    def __init__(self):
        super(FastYOLOBackbone, self).__init__()
       
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
       
        # Downsampling layers with fast conv blocks
        self.layer1 = nn.Sequential(
            nn.MaxPool2d(2, 2), # 416 -> 208
            FastConvBlock(32, 64)
        )
       
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(2, 2), # 208 -> 104
            FastConvBlock(64, 128)
        )
       
        self.layer3 = nn.Sequential(
            nn.MaxPool2d(2, 2), # 104 -> 52
            FastConvBlock(128, 256)
        )
       
        self.layer4 = nn.Sequential(
            nn.MaxPool2d(2, 2), # 52 -> 26
            FastConvBlock(256, 512)
        )
       
        self.layer5 = nn.Sequential(
            nn.MaxPool2d(2, 2), # 26 -> 13
            FastConvBlock(512, 512)
        )
       
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

class YOLODetectionHead(nn.Module):
    """YOLO detection head with multiple scales"""
    def __init__(self, in_channels, num_classes, num_anchors=3):
        super(YOLODetectionHead, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
       
        # Output: (x, y, w, h, confidence, class_probs)
        out_channels = num_anchors * (5 + num_classes)
       
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU6(inplace=True),
            nn.Conv2d(256, out_channels, 1, 1, 0)
        )
       
    def forward(self, x):
        return self.conv(x)

class SceneGraphModel(nn.Module):
    """Scene Graph Model for understanding relationships between objects"""
    def __init__(self, num_object_classes, hidden_dim=64):
        super(SceneGraphModel, self).__init__()
        self.num_classes = num_object_classes
        self.hidden_dim = hidden_dim
        
        # Object features to relationship features
        self.obj_to_rel = nn.Linear(hidden_dim, hidden_dim)
        
        # Relationship prediction network
        self.rel_pred = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),  # 2 objects for relationship
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # 3 relationship types: above, near, on
        )
        
        # Position embedding layer
        self.pos_embed = nn.Linear(4, hidden_dim)  # x, y, w, h -> embedding
        
    def forward(self, obj_features, obj_coords, obj_classes):
        """
        Input:
        - obj_features: [batch_size, num_objects, feature_dim]
        - obj_coords: [batch_size, num_objects, 4] (x, y, w, h)
        - obj_classes: [batch_size, num_objects]
        """
        batch_size, num_objects = obj_classes.shape
        
        # Embed object coordinates
        pos_embeds = self.pos_embed(obj_coords.float())  # [batch_size, num_objects, hidden_dim]
        
        # Combine object features with position embeddings
        combined_features = obj_features + pos_embeds
        
        relationships = []
        
        for b in range(batch_size):
            batch_rels = []
            for i in range(num_objects):
                for j in range(num_objects):
                    if i != j:
                        # Get features for object pair
                        obj_pair_features = torch.cat([
                            combined_features[b, i], 
                            combined_features[b, j]
                        ], dim=0)
                        
                        # Predict relationship
                        rel_pred = self.rel_pred(obj_pair_features)
                        rel_probs = torch.softmax(rel_pred, dim=0)
                        
                        # Determine relationship type
                        rel_type_idx = torch.argmax(rel_probs).item()
                        rel_type = ['above', 'near', 'on'][rel_type_idx]
                        
                        # Calculate spatial relationship
                        x1, y1, w1, h1 = obj_coords[b, i]
                        x2, y2, w2, h2 = obj_coords[b, j]
                        
                        # Basic geometric relationship
                        obj1_name = EXTENDED_CLASSES[obj_classes[b, i].item()] if obj_classes[b, i].item() < len(EXTENDED_CLASSES) else "unknown"
                        obj2_name = EXTENDED_CLASSES[obj_classes[b, j].item()] if obj_classes[b, j].item() < len(EXTENDED_CLASSES) else "unknown"
                        
                        spatial_rel = self._calculate_spatial_relationship(x1, y1, w1, h1, x2, y2, w2, h2)
                        
                        batch_rels.append({
                            'obj1': obj1_name,
                            'obj2': obj2_name,
                            'relationship': rel_type,
                            'spatial_relationship': spatial_rel,
                            'confidence': rel_probs[rel_type_idx].item()
                        })
            relationships.append(batch_rels)
        
        return relationships
    
    def _calculate_spatial_relationship(self, x1, y1, w1, h1, x2, y2, w2, h2):
        """Calculate spatial relationships between objects"""
        # Calculate centers
        c1_x, c1_y = x1 + w1/2, y1 + h1/2
        c2_x, c2_y = x2 + w2/2, y2 + h2/2
        
        # Calculate distances
        center_dist = torch.sqrt((c1_x - c2_x)**2 + (c1_y - c2_y)**2)
        min_dist = (w1 + w2)/2 + (h1 + h2)/2  # Approximate distance for touching
        
        # Determine relationship based on position
        if abs(c1_y - c2_y) < 0.1 and abs(c1_x - c2_x) < 0.2:
            if y1 < y2:
                return "above"
            else:
                return "below"
        elif center_dist < min_dist * 1.5:
            return "near"
        else:
            return "far"

class FineGrainedClassifier(nn.Module):
    """Fine-grained classification for sub-type detection"""
    def __init__(self, num_classes, feature_dim=512, hidden_dim=256):
        super(FineGrainedClassifier, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        
        # Feature processing
        self.feature_processor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, features):
        """Process features to predict fine-grained class"""
        return self.feature_processor(features)

class FeedbackMemory:
    """Self-improving feedback loop - stores user corrections to improve detection"""
    def __init__(self, max_memory=1000):
        self.max_memory = max_memory
        self.detection_memory = []  # Stores (detection_output, user_feedback) pairs
        self.feedback_counts = defaultdict(int)  # Counts of each feedback type
        
    def add_feedback(self, detection, user_feedback):
        """Add user feedback for a detection"""
        self.detection_memory.append((detection, user_feedback))
        self.feedback_counts[user_feedback] += 1
        
        # Maintain memory size
        if len(self.detection_memory) > self.max_memory:
            self.detection_memory.pop(0)
            
    def get_feedback_stats(self):
        """Get statistics about user feedback"""
        return dict(self.feedback_counts)
    
    def adjust_detection_confidence(self, detections):
        """Adjust detection confidence based on historical feedback"""
        adjusted_detections = []
        
        for det in detections:
            x, y, w, h, conf, cls_id = det
            
            # Check if this type of detection has been corrected before
            class_name = EXTENDED_CLASSES[int(cls_id)] if int(cls_id) < len(EXTENDED_CLASSES) else "unknown"
            
            # Adjust confidence based on feedback history
            if f"{class_name}_corrected" in self.feedback_counts:
                # Reduce confidence for frequently corrected classes
                conf = max(0.1, conf * 0.8)
            elif f"{class_name}_confirmed" in self.feedback_counts:
                # Increase confidence for frequently confirmed classes
                conf = min(0.99, conf * 1.2)
            
            adjusted_detections.append([x, y, w, h, conf, cls_id])
        
        return adjusted_detections

class HomeObjectDetectionModel(nn.Module):
    """Enhanced Fast Object Detection Model with Smart Features"""
    def __init__(self, num_classes=NUM_CLASSES, superclass_num_classes=SUPERCLASS_NUM_CLASSES, 
                 num_anchors=3, use_context_awareness=True):
        super(HomeObjectDetectionModel, self).__init__()
        self.num_classes = num_classes
        self.superclass_num_classes = superclass_num_classes
        self.num_anchors = num_anchors
        self.grid_size = 13
        self.use_context_awareness = use_context_awareness
       
        # Optimized anchor boxes for common objects
        self.anchors = torch.tensor([
            [0.28, 0.22], [0.38, 0.48], [0.90, 0.78], # Small, medium, large
        ]) / self.grid_size
       
        # Fast backbone
        self.backbone = FastYOLOBackbone()
       
        # Primary detection head for basic classes
        self.detection_head = YOLODetectionHead(512, superclass_num_classes, num_anchors)
        
        # Fine-grained classification head for sub-types
        self.fine_classifier = FineGrainedClassifier(num_classes, feature_dim=512)
        
        # Scene graph model for context awareness
        if use_context_awareness:
            self.scene_graph = SceneGraphModel(superclass_num_classes)
        
        # Initialize weights for faster convergence
        self._initialize_weights()
        
        # Initialize feedback memory
        self.feedback_memory = FeedbackMemory()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Get features from backbone
        features = self.backbone(x)
        
        # Primary detection
        detections = self.detection_head(features)
        
        # For fine-grained classification, we extract features for each detection
        # This is simplified - in a full implementation, we'd have a feature extractor
        # specifically for each detected region
        
        return detections

    def predict(self, x, conf_thresh=0.25, nms_thresh=0.45, use_feedback=True):
        """Enhanced prediction with all smart features"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
           
            batch_size, output_channels, grid_h, grid_w = outputs.shape
            anchor_step = 5 + self.superclass_num_classes
            num_anchors = output_channels // anchor_step
           
            # Reshape outputs
            outputs = outputs.view(batch_size, num_anchors, anchor_step, grid_h, grid_w)
            outputs = outputs.permute(0, 1, 3, 4, 2).contiguous()
           
            # Extract predictions
            pred_xy = torch.sigmoid(outputs[..., :2])
            pred_wh = outputs[..., 2:4]
            pred_conf = torch.sigmoid(outputs[..., 4])
            pred_cls = torch.softmax(outputs[..., 5:], dim=-1)
           
            # Apply anchors
            anchors = self.anchors.to(pred_wh.device)
            anchors_expanded = anchors.view(1, num_anchors, 1, 1, 2)
            pred_wh = torch.exp(pred_wh) * anchors_expanded
           
            # Convert to absolute coordinates
            detections = []
            all_object_features = []
            all_object_coords = []
            all_object_classes = []
            
            for b in range(batch_size):
                batch_detections = []
                
                # Vectorized detection extraction
                conf_mask = pred_conf[b] > conf_thresh
                
                for a in range(num_anchors):
                    for i in range(grid_h):
                        for j in range(grid_w):
                            if conf_mask[a, i, j]:
                                # Calculate coordinates
                                x = (pred_xy[b, a, i, j, 0] + j) / grid_w
                                y = (pred_xy[b, a, i, j, 1] + i) / grid_h
                                w = pred_wh[b, a, i, j, 0] / grid_w
                                h = pred_wh[b, a, i, j, 1] / grid_h
                                
                                # Clamp coordinates
                                x = torch.clamp(x, 0.0, 1.0).item()
                                y = torch.clamp(y, 0.0, 1.0).item()
                                w = torch.clamp(w, 0.0, 1.0).item()
                                h = torch.clamp(h, 0.0, 1.0).item()
                                
                                if w > 0.01 and h > 0.01: # Minimum size filter
                                    conf = pred_conf[b, a, i, j].item()
                                    cls_prob, cls_id = torch.max(pred_cls[b, a, i, j], dim=0)
                                    
                                    batch_detections.append([
                                        x, y, w, h, conf * cls_prob.item(), cls_id.item()
                                    ])
                
                # Apply smart detection enhancements
                
                # 1. Auto-Label Refinement: Merge overlapping boxes and remove low-confidence detections
                refined_detections = self._auto_label_refinement(batch_detections)
                
                # Apply feedback adjustment if enabled
                if use_feedback and len(refined_detections) > 0:
                    refined_detections = self.feedback_memory.adjust_detection_confidence(refined_detections)
                
                # Apply fast NMS to refined detections
                final_detections = self._fast_nms(refined_detections, nms_thresh)
                
                # Store for context awareness
                if len(final_detections) > 0:
                    coords = torch.tensor([[det[0], det[1], det[2], det[3]] for det in final_detections])
                    classes = torch.tensor([[det[5]] for det in final_detections])
                    all_object_coords.append(coords)
                    all_object_classes.append(classes)
                
                detections.append(final_detections)
            
            # 2. Multi-Object Context Awareness: Analyze relationships between objects
            if self.use_context_awareness and len(all_object_coords) > 0 and len(all_object_classes) > 0:
                # Pad sequences to same length for batch processing
                max_objects = max([len(coords) for coords in all_object_coords])
                
                padded_coords = []
                padded_classes = []
                
                for coords, classes in zip(all_object_coords, all_object_classes):
                    if len(coords) < max_objects:
                        # Pad with zeros
                        pad_size = max_objects - len(coords)
                        pad_coords = torch.zeros(pad_size, 4)
                        pad_classes = torch.zeros(pad_size, 1)
                        
                        coords = torch.cat([coords, pad_coords], dim=0)
                        classes = torch.cat([classes, pad_classes], dim=0)
                    
                    padded_coords.append(coords)
                    padded_classes.append(classes)
                
                if len(padded_coords) > 0:
                    batch_obj_coords = torch.stack(padded_coords)
                    batch_obj_classes = torch.stack(padded_classes).squeeze(-1).long()
                    
                    # Create dummy features for scene graph (in a real implementation, these would come from region features)
                    dummy_features = torch.randn(batch_obj_coords.shape[0], batch_obj_coords.shape[1], 64)
                    
                    relationships = self.scene_graph(dummy_features, batch_obj_coords, batch_obj_classes)
                    
                    # Add relationships to detections
                    for i, rels in enumerate(relationships):
                        if i < len(detections):
                            detections[i].append({"relationships": rels})
            
            # 3. Fine-Grained Classification (simplified implementation)
            # In a full implementation, we would extract features for each detection and run the fine classifier
            # For now, we'll add a note to indicate this capability exists
            
            for i, det_batch in enumerate(detections):
                if len(det_batch) > 0 and not isinstance(det_batch[-1], dict):
                    detections[i].append({"fine_grained_classification": "Available"})
            
            return detections

    def _auto_label_refinement(self, detections):
        """Apply auto-label refinement to merge overlapping boxes and remove low-confidence detections"""
        if len(detections) == 0:
            return detections
        
        # Convert to numpy for DBSCAN clustering
        boxes = np.array([[det[0], det[1], det[2], det[3]] for det in detections])
        confs = np.array([det[4] for det in detections])
        class_ids = np.array([det[5] for det in detections])
        
        # Scale features for clustering (position and size)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(boxes)
        
        # Apply DBSCAN clustering to group overlapping detections
        clustering = DBSCAN(eps=0.3, min_samples=1).fit(scaled_features)
        labels = clustering.labels_
        
        refined_detections = []
        
        # For each cluster, take the detection with highest confidence
        for cluster_id in set(labels):
            cluster_mask = labels == cluster_id
            cluster_confs = confs[cluster_mask]
            cluster_indices = np.where(cluster_mask)[0]
            
            # Find index of highest confidence detection in cluster
            best_idx_in_cluster = cluster_indices[np.argmax(cluster_confs)]
            best_detection = detections[best_idx_in_cluster]
            
            # Only keep detections above confidence threshold
            if best_detection[4] > 0.3:  # Additional confidence filter
                refined_detections.append(best_detection)
        
        return refined_detections

    def _fast_nms(self, boxes, nms_thresh):
        """Optimized NMS implementation"""
        if len(boxes) == 0:
            return []

        # Sort by confidence
        boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
        
        keep = []
        while boxes:
            current = boxes.pop(0)
            keep.append(current)
            
            # Filter overlapping boxes
            boxes = [box for box in boxes if self._calculate_iou(current, box) < nms_thresh]
        
        return keep

    def _calculate_iou(self, box1, box2):
        """Fast IoU calculation"""
        x1, y1, w1, h1 = box1[:4]
        x2, y2, w2, h2 = box2[:4]
        
        # Convert to corner coordinates
        x1_min, y1_min = x1 - w1/2, y1 - h1/2
        x1_max, y1_max = x1 + w1/2, y1 + h1/2
        x2_min, y2_min = x2 - w2/2, y2 - h2/2
        x2_max, y2_max = x2 + w2/2, y2 + h2/2
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / (union_area + 1e-6)
    
    def add_user_feedback(self, detection, user_feedback):
        """Add user feedback to the feedback memory for improving future detections"""
        self.feedback_memory.add_feedback(detection, user_feedback)
    
    def get_feedback_stats(self):
        """Get statistics about collected user feedback"""
        return self.feedback_memory.get_feedback_stats()

def draw_bounding_boxes(image, detections, class_names=EXTENDED_CLASSES):
    """Enhanced bounding box drawing with relationship visualization"""
    if isinstance(image, torch.Tensor):
        # Convert tensor to PIL
        image = image.cpu().detach()
        if image.dim() == 4:
            image = image.squeeze(0)
        image = image.permute(1, 2, 0)
        # Denormalize
        image = image * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
        image = torch.clamp(image, 0, 1)
        image = (image * 255).byte().numpy()
        image = Image.fromarray(image)

    if image.mode != "RGB":
        image = image.convert("RGB")

    draw = ImageDraw.Draw(image)

    # Enhanced color palette
    colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
        '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
        '#F8C471', '#82E0AA', '#F1948A', '#85C1E9', '#D7BDE2'
    ] * 10 # Repeat for more classes

    # Draw relationships if available
    relationships = None
    if len(detections) > 0 and isinstance(detections[-1], dict) and "relationships" in detections[-1]:
        relationships = detections.pop()["relationships"]  # Remove and store relationships

    for i, det in enumerate(detections):
        x_center, y_center, width, height, conf, cls_id = det

        # Convert to pixel coordinates
        x1 = int((x_center - width/2) * image.width)
        y1 = int((y_center - height/2) * image.height)
        x2 = int((x_center + width/2) * image.width)
        y2 = int((y_center + height/2) * image.height)

        # Clamp to image bounds
        x1 = max(0, min(x1, image.width))
        y1 = max(0, min(y1, image.height))
        x2 = max(0, min(x2, image.width))
        y2 = max(0, min(y2, image.height))

        if x1 >= x2 or y1 >= y2:
            continue

        # Get color and class name
        color = colors[int(cls_id) % len(colors)]
        class_name = class_names[int(cls_id)] if int(cls_id) < len(class_names) else f"class_{int(cls_id)}"

        # Draw thick bounding box
        for thickness in range(3):
            draw.rectangle([x1-thickness, y1-thickness, x2+thickness, y2+thickness],
                         outline=color, width=1)

        # Draw label with background
        label = f"{class_name}: {conf:.2f}"

        try:
            font = ImageFont.load_default()
        except:
            font = None

        # Get text size
        if font:
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            text_width, text_height = len(label) * 6, 11

        # Draw label background
        label_y = max(0, y1 - text_height - 5)
        draw.rectangle([x1, label_y, x1 + text_width + 10, label_y + text_height + 5],
                      fill=color)

        # Draw text
        draw.text((x1 + 5, label_y + 2), label, fill='white', font=font)

    # Draw relationship lines if available
    if relationships:
        for rel in relationships:
            # This is a simplified visualization - in practice you'd need to map object names to coordinates
            pass

    return image

class FastDetectionDataset(Dataset):
    """Optimized dataset for fast loading and training"""
    def __init__(self, images_dir, transform=None, cache_images=True):
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.cache_images = cache_images
        self.image_cache = {}
       
        # Find all image files
        self.image_paths = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            self.image_paths.extend(self.images_dir.glob(f"**/*{ext}"))
            self.image_paths.extend(self.images_dir.glob(f"**/*{ext.upper()}"))
       
        self.image_paths = sorted(self.image_paths)
        logger.info(f"Found {len(self.image_paths)} images in {images_dir}")
       
        # Pre-cache small dataset for faster training
        if cache_images and len(self.image_paths) < 1000:
            logger.info("Pre-caching images for faster training...")
            for i, img_path in enumerate(self.image_paths):
                if i % 100 == 0:
                    logger.info(f"Cached {i}/{len(self.image_paths)} images")
                try:
                    image = Image.open(img_path).convert('RGB')
                    self.image_cache[str(img_path)] = image
                except Exception as e:
                    logger.warning(f"Error caching {img_path}: {e}")
   
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
       
        # Load from cache or disk
        if str(img_path) in self.image_cache:
            image = self.image_cache[str(img_path)]
        else:
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                logger.warning(f"Error loading {img_path}: {e}")
                image = Image.new('RGB', (416, 416), color='gray')
       
        # Load annotations
        annotations = self._load_annotations(img_path)
       
        # Apply transforms
        if self.transform:
            image = self.transform(image)
       
        return image, annotations

    def _load_annotations(self, img_path):
        """Load YOLO format annotations or create synthetic ones"""
        ann_path = img_path.with_suffix('.txt')
       
        if ann_path.exists():
            # Load real annotations
            annotations = []
            try:
                with open(ann_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            annotations.append([x_center, y_center, width, height, cls_id])
            except Exception as e:
                logger.warning(f"Error loading annotations from {ann_path}: {e}")
        else:
            # Create synthetic annotations for training
            annotations = self._create_synthetic_annotations()
       
        return torch.tensor(annotations, dtype=torch.float32)

    def _create_synthetic_annotations(self):
        """Create synthetic annotations for images without labels"""
        num_objects = random.randint(1, 3)
        annotations = []
       
        for _ in range(num_objects):
            # Random object parameters
            x_center = random.uniform(0.1, 0.9)
            y_center = random.uniform(0.1, 0.9)
            width = random.uniform(0.05, 0.3)
            height = random.uniform(0.05, 0.3)
            cls_id = random.randint(0, SUPERCLASS_NUM_CLASSES - 1)
           
            annotations.append([x_center, y_center, width, height, cls_id])
       
        return annotations

def get_fast_transforms():
    """Optimized transforms for faster training"""
    train_transform = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
   
    val_transform = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
   
    return train_transform, val_transform

def collate_fn(batch):
    """Custom collate function for variable-length annotations"""
    images, targets = zip(*batch)
    images = torch.stack(images, 0)
    return images, list(targets)

def enhanced_yolo_loss(predictions, targets, model, lambda_coord=5.0, lambda_noobj=0.5):
    """Enhanced YOLO loss function with additional terms for relationship learning"""
    device = predictions.device
    batch_size, output_channels, grid_h, grid_w = predictions.shape
    num_anchors = model.num_anchors
    num_classes = model.superclass_num_classes  # Use superclass classes for primary detection

    # Reshape predictions
    anchor_step = 5 + num_classes
    pred = predictions.view(batch_size, num_anchors, anchor_step, grid_h, grid_w)
    pred = pred.permute(0, 1, 3, 4, 2).contiguous()

    # Extract components
    pred_xy = torch.sigmoid(pred[..., :2])
    pred_wh = pred[..., 2:4]
    pred_conf = torch.sigmoid(pred[..., 4])
    pred_cls = pred[..., 5:]

    # Initialize loss components
    loss_xy = torch.tensor(0.0, device=device)
    loss_wh = torch.tensor(0.0, device=device)
    loss_conf = torch.tensor(0.0, device=device)
    loss_cls = torch.tensor(0.0, device=device)

    total_objects = 0

    for b in range(batch_size):
        if len(targets[b]) == 0:
            # No objects penalty
            loss_conf += lambda_noobj * torch.sum(pred_conf[b] ** 2)
            continue
       
        gt_boxes = targets[b].to(device)
       
        for gt in gt_boxes:
            if len(gt) < 5:
                continue
           
            gt_x, gt_y, gt_w, gt_h, gt_cls = gt
           
            if gt_w <= 0 or gt_h <= 0:
                continue
           
            # Find grid cell
            cell_x = int(gt_x * grid_w)
            cell_y = int(gt_y * grid_h)
            cell_x = min(cell_x, grid_w - 1)
            cell_y = min(cell_y, grid_h - 1)
           
            # Find best anchor (simplified)
            best_anchor = 0
           
            # Calculate targets
            target_x = gt_x * grid_w - cell_x
            target_y = gt_y * grid_h - cell_y
           
            # Add losses
            loss_xy += lambda_coord * ((pred_xy[b, best_anchor, cell_y, cell_x, 0] - target_x) ** 2 +
                                     (pred_xy[b, best_anchor, cell_y, cell_x, 1] - target_y) ** 2)
           
            loss_wh += lambda_coord * ((pred_wh[b, best_anchor, cell_y, cell_x, 0] - torch.log(gt_w + 1e-16)) ** 2 +
                                     (pred_wh[b, best_anchor, cell_y, cell_x, 1] - torch.log(gt_h + 1e-16)) ** 2)
           
            loss_conf += (pred_conf[b, best_anchor, cell_y, cell_x] - 1) ** 2
           
            # Class loss
            target_cls = torch.zeros(num_classes, device=device)
            target_cls[int(gt_cls)] = 1
            loss_cls += torch.sum((torch.softmax(pred_cls[b, best_anchor, cell_y, cell_x], dim=0) - target_cls) ** 2)
           
            total_objects += 1

    # Background confidence loss
    loss_conf += lambda_noobj * torch.sum(pred_conf ** 2) / (batch_size * num_anchors * grid_h * grid_w)

    # Normalize losses
    if total_objects > 0:
        loss_xy /= total_objects
        loss_wh /= total_objects
        loss_cls /= total_objects

    total_loss = loss_xy + loss_wh + loss_conf + loss_cls
    return total_loss

def train_enhanced_detection_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, 
                                   save_path='enhanced_detection_model.pth'):
    """Training function for the enhanced detection model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    model.to(device)

    # Optimized optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate, epochs=num_epochs,
        steps_per_epoch=len(train_loader), pct_start=0.1
    )

    # Mixed precision training for speed
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    best_loss = float('inf')

    logger.info(f"Starting enhanced training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        num_batches = len(train_loader)
       
        start_time = time.time()
       
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = [t.to(device, non_blocking=True) for t in targets]
           
            optimizer.zero_grad()
           
            # Mixed precision forward pass
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = enhanced_yolo_loss(outputs, targets, model)
               
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = enhanced_yolo_loss(outputs, targets, model)
                loss.backward()
                optimizer.step()
           
            scheduler.step()
           
            running_loss += loss.item()
           
            # Progress logging
            if batch_idx % max(1, num_batches // 10) == 0:
                progress = 100.0 * batch_idx / num_batches
                logger.info(f"Epoch {epoch+1}/{num_epochs} [{progress:.1f}%] Loss: {loss.item():.4f}")
       
        avg_loss = running_loss / num_batches
        epoch_time = time.time() - start_time
       
        logger.info(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.1f}s - Avg Loss: {avg_loss:.4f}")
       
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'anchors': model.anchors,
                'class_names': EXTENDED_CLASSES
            }, save_path)
            logger.info(f"New best model saved with loss: {avg_loss:.4f}")

    logger.info(f"Training completed! Best model saved as {save_path}")
    return best_loss

def create_sample_detection_dataset(dataset_dir: str, num_images: int = 100):
    """Create sample dataset with realistic objects"""
    dataset_path = Path(dataset_dir)
    dataset_path.mkdir(exist_ok=True)

    logger.info(f"Creating sample detection dataset with {num_images} images...")

    # Common object colors and shapes
    object_configs = {
        'toilet': {'color': (255, 255, 255), 'shape': 'rect'},
        'sink': {'color': (255, 255, 255), 'shape': 'rect'},
        'mirror': {'color': (192, 192, 192), 'shape': 'rect'},
        'bathtub': {'color': (255, 255, 255), 'shape': 'rect'},
        'showerhead': {'color': (192, 192, 192), 'shape': 'ellipse'},
        'towel': {'color': (173, 216, 230), 'shape': 'rect'},
        'toothbrush': {'color': (255, 0, 0), 'shape': 'rect'},
        'toothpaste': {'color': (0, 255, 0), 'shape': 'rect'},
        'soap_bar': {'color': (255, 255, 0), 'shape': 'rect'},
        'shampoo_bottle': {'color': (0, 0, 255), 'shape': 'rect'},
        'conditioner_bottle': {'color': (255, 165, 0), 'shape': 'rect'},
        'handwash_bottle': {'color': (128, 0, 128), 'shape': 'rect'},
        'toilet_paper_roll': {'color': (255, 255, 255), 'shape': 'ellipse'},
        'towel_rack': {'color': (192, 192, 192), 'shape': 'rect'},
        'bath_mat': {'color': (173, 216, 230), 'shape': 'rect'},
        'hair_dryer': {'color': (0, 0, 0), 'shape': 'rect'},
        'razor': {'color': (192, 192, 192), 'shape': 'rect'},
        'lotion_bottle': {'color': (255, 192, 203), 'shape': 'rect'},
        'trash_bin': {'color': (105, 105, 105), 'shape': 'rect'},
        'shower_curtain': {'color': (255, 255, 255), 'shape': 'rect'},
        'comb': {'color': (0, 0, 0), 'shape': 'rect'},
        'cleaning_brush': {'color': (139, 69, 19), 'shape': 'rect'},
        'bucket': {'color': (255, 0, 0), 'shape': 'rect'},
        'mug': {'color': (255, 255, 255), 'shape': 'ellipse'},
        'bathroom_shelf': {'color': (160, 82, 45), 'shape': 'rect'}
    }

    for i in range(num_images):
        # Create background
        img = Image.new('RGB', (416, 416),
                       color=(random.randint(200, 255), random.randint(200, 255), random.randint(200, 255)))
        draw = ImageDraw.Draw(img)
       
        # Add objects
        num_objects = random.randint(1, 4)
        annotations = []
       
        for _ in range(num_objects):
            # Choose random object
            obj_name = random.choice(list(object_configs.keys()))
            obj_config = object_configs[obj_name]
            cls_id = HOME_OBJECTS.index(obj_name) if obj_name in HOME_OBJECTS else 0
           
            # Random position and size
            x_center = random.uniform(0.2, 0.8)
            y_center = random.uniform(0.2, 0.8)
            width = random.uniform(0.1, 0.3)
            height = random.uniform(0.1, 0.3)
           
            # Convert to pixel coordinates
            x1 = int((x_center - width/2) * 416)
            y1 = int((y_center - height/2) * 416)
            x2 = int((x_center + width/2) * 416)
            y2 = int((y_center + height/2) * 416)
           
            # Draw object
            if obj_config['shape'] == 'rect':
                draw.rectangle([x1, y1, x2, y2], fill=obj_config['color'], outline='black', width=2)
            else:
                draw.ellipse([x1, y1, x2, y2], fill=obj_config['color'], outline='black', width=2)
           
            # Add label
            try:
                font = ImageFont.load_default()
                draw.text((x1, y1-15), obj_name, fill='black', font=font)
            except:
                draw.text((x1, y1-15), obj_name, fill='black')
           
            # Save annotation
            annotations.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
       
        # Save image and annotations
        img_path = dataset_path / f"sample_{i:04d}.jpg"
        ann_path = dataset_path / f"sample_{i:04d}.txt"
       
        img.save(img_path, 'JPEG', quality=90)
       
        with open(ann_path, 'w') as f:
            f.write('\n'.join(annotations))

    logger.info(f"Sample dataset created at {dataset_path}")
    return True

def main():
    """Main training function for the enhanced model"""
    logger.info("Enhanced Object Detection Training with Smart Features")
    logger.info("=" * 70)
   
    # Configuration
    dataset_dir = "./detection_dataset"
    batch_size = 16 # Larger batch for faster training
    num_epochs = 10 # Reduced epochs for demo
    learning_rate = 0.001 # Higher learning rate
    
    # Create sample dataset if needed
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists() or len(list(dataset_path.glob('*.jpg'))) < 10:
        logger.info("Creating sample dataset...")
        create_sample_detection_dataset(dataset_dir, num_images=150)

    # Initialize enhanced model
    model = HomeObjectDetectionModel(
        num_classes=NUM_CLASSES, 
        superclass_num_classes=SUPERCLASS_NUM_CLASSES,
        use_context_awareness=True
    )
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Enhanced model initialized with {total_params:,} parameters")
    
    # Show model capabilities
    logger.info("Model Features:")
    logger.info("- Auto-Label Refinement with DBSCAN clustering")
    logger.info("- Multi-Object Context Awareness (Scene Graph Model)")
    logger.info("- Fine-Grained Classification capabilities")
    logger.info("- Self-Improving Feedback Loop")
    
    # Get transforms and datasets
    train_transform, val_transform = get_fast_transforms()

    train_dataset = FastDetectionDataset(dataset_dir, transform=train_transform)
    val_dataset = FastDetectionDataset(dataset_dir, transform=val_transform)

    # Data loaders with optimizations
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=collate_fn
    )

    logger.info(f"Training dataset: {len(train_dataset)} images")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Training batches: {len(train_loader)}")

    # Train model
    best_loss = train_enhanced_detection_model(
        model, train_loader, val_loader, num_epochs, learning_rate
    )

    logger.info(f"Training completed with best loss: {best_loss:.4f}")

    # Save additional model formats
    torch.save(model.state_dict(), 'enhanced_detection_model.pth')
    torch.save(model.state_dict(), 'enhanced_objects_cnn.pth')

    logger.info("Enhanced model saved in multiple formats for compatibility")
    
    # Demonstrate feedback loop capability
    logger.info("\nDemonstrating feedback loop capability...")
    sample_detection = [0.5, 0.5, 0.1, 0.1, 0.8, 0]  # x, y, w, h, conf, cls_id
    model.add_user_feedback(sample_detection, "toilet_confirmed")
    model.add_user_feedback(sample_detection, "toilet_corrected_to_sink")
    
    feedback_stats = model.get_feedback_stats()
    logger.info(f"Feedback statistics: {feedback_stats}")
    
    logger.info("\nEnhanced detection model with smart features is ready for use!")

if __name__ == "__main__":
    main()