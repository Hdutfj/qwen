"""
AI 3D Scene Mapper - From 2D images to interactive 3D environments
This module implements depth estimation, 3D reconstruction, and scene visualization
"""
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw
import cv2
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
import logging

# Handle transformers import gracefully
try:
    from transformers import pipeline, AutoFeatureExtractor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    pipeline = None
    AutoFeatureExtractor = None
    print("Warning: transformers library not available. Depth estimation will use dummy implementation.")

logger = logging.getLogger(__name__)

@dataclass
class Object3DInfo:
    """3D information for each detected object"""
    class_name: str
    bbox: List[float]  # [x1, y1, x2, y2]
    center_3d: List[float]  # [x, y, z] in 3D space
    dimensions_3d: List[float]  # [width, height, depth] in 3D space
    confidence: float


class DepthEstimator:
    """Depth estimation using transformer-based models"""
    
    def __init__(self, model_name: str = "Intel/dpt-hybrid-midas"):
        """
        Initialize depth estimation model
        Options: 
        - "Intel/dpt-hybrid-midas" (DPT with MiDaS backbone)
        - "LiheYoung/depth_anything_vitl14" (Depth Anything)
        """
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available. Using dummy depth estimation.")
            self.model_available = False
            return
            
        logger.info(f"Initializing depth estimation model: {model_name}")
        
        try:
            # Load feature extractor and depth estimation pipeline
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
            self.depth_estimator = pipeline(
                "depth-estimation",
                model=model_name,
                feature_extractor=model_name
            )
            self.model_available = True
            logger.info("Depth estimation model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading depth estimation model: {e}")
            self.model_available = False
            raise
    
    def estimate_depth(self, image: Image.Image) -> np.ndarray:
        """
        Estimate depth map from input image
        Returns: depth map as numpy array
        """
        if not TRANSFORMERS_AVAILABLE or not self.model_available:
            # Return a simple simulated depth map
            # In a real implementation, you might want to implement basic depth estimation
            # or use a different approach when transformers is not available
            height, width = image.height, image.width
            # Create a simple gradient depth map (farther = brighter)
            x = np.linspace(0, 1, width)
            y = np.linspace(0, 1, height)
            X, Y = np.meshgrid(x, y)
            # Simulate depth with center being closer (convex surface effect)
            depth_map = 0.3 + 0.7 * np.sqrt((X - 0.5)**2 + (Y - 0.5)**2)
            return depth_map.astype(np.float32)
        
        try:
            # Get the depth prediction
            outputs = self.depth_estimator(image)
            depth_map = outputs["depth"]
            
            # Convert PIL image to numpy array
            depth_array = np.array(depth_map)
            
            return depth_array
        except Exception as e:
            logger.error(f"Error in depth estimation: {e}")
            # Return a dummy depth map in case of error
            return np.ones((image.height, image.width), dtype=np.float32) * 0.5


class PointCloudGenerator:
    """Generate 3D point cloud from depth map and camera parameters"""
    
    def __init__(self):
        # Default camera intrinsic parameters (can be calibrated per use case)
        self.camera_matrix = np.array([
            [500, 0, 320],   # fx, 0, cx
            [0, 500, 240],   # 0, fy, cy
            [0, 0, 1]        # 0, 0, 1
        ])
    
    def generate_point_cloud(self, depth_map: np.ndarray, image: Image.Image) -> np.ndarray:
        """
        Generate 3D point cloud from depth map
        Returns: point cloud as numpy array (H, W, 3) where each pixel has (x, y, z) coordinates
        """
        height, width = depth_map.shape
        
        # Generate coordinate grids
        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
        
        # Convert to homogeneous coordinates
        x_coords = x_coords.astype(np.float32)
        y_coords = y_coords.astype(np.float32)
        
        # Apply camera intrinsics to get 3D coordinates
        # z is the depth value
        z_coords = depth_map.astype(np.float32)
        
        # Calculate X and Y coordinates using inverse camera matrix
        inv_fx = 1.0 / self.camera_matrix[0, 0]
        inv_fy = 1.0 / self.camera_matrix[1, 1]
        center_x = self.camera_matrix[0, 2]
        center_y = self.camera_matrix[1, 2]
        
        x_coords = (x_coords - center_x) * z_coords * inv_fx
        y_coords = (y_coords - center_y) * z_coords * inv_fy
        
        # Stack to create 3D point cloud
        point_cloud = np.stack((x_coords, y_coords, z_coords), axis=-1)
        
        return point_cloud

    def extract_object_point_cloud(self, point_cloud: np.ndarray, bbox: List[int]) -> np.ndarray:
        """
        Extract point cloud for a specific object within its bounding box
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure coordinates are within bounds
        h, w = point_cloud.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        
        if x1 >= x2 or y1 >= y2:
            return np.array([])
        
        # Extract the region of interest
        object_points = point_cloud[y1:y2, x1:x2, :]
        
        # Reshape to (N, 3) where N is number of points
        object_points = object_points.reshape(-1, 3)
        
        # Remove any invalid points (e.g., with depth of 0)
        valid_mask = object_points[:, 2] > 0
        object_points = object_points[valid_mask]
        
        return object_points

    def compute_object_center_from_point_cloud(self, object_points: np.ndarray) -> List[float]:
        """
        Compute the 3D center of an object from its point cloud
        """
        if len(object_points) == 0:
            return [0.0, 0.0, 0.0]
        
        # Calculate mean of all points
        center = np.mean(object_points, axis=0)
        return center.tolist()

    def compute_object_dimensions_from_point_cloud(self, object_points: np.ndarray) -> List[float]:
        """
        Compute 3D dimensions of an object from its point cloud
        """
        if len(object_points) == 0:
            return [0.0, 0.0, 0.0]
        
        # Calculate min and max in each dimension
        min_vals = np.min(object_points, axis=0)
        max_vals = np.max(object_points, axis=0)
        
        # Calculate dimensions (width, height, depth)
        dimensions = (max_vals - min_vals).tolist()
        
        return dimensions


class Object3DAnchorer:
    """Anchors detected objects in 3D space based on depth information"""
    
    def __init__(self):
        self.depth_estimator = DepthEstimator()
        self.point_cloud_generator = PointCloudGenerator()
    
    def compute_object_depth(self, depth_map: np.ndarray, bbox: List[int]) -> float:
        """Compute average depth of an object within its bounding box"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure coordinates are within bounds
        h, w = depth_map.shape
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        
        if x1 >= x2 or y1 >= y2:
            return float(np.mean(depth_map))
        
        # Get depth values within the bounding box
        object_depth = depth_map[y1:y2, x1:x2]
        
        # Return median depth to be more robust to outliers
        return float(np.median(object_depth))
    
    def estimate_object_dimensions(self, bbox: List[int], depth: float, image_size: Tuple[int, int]) -> List[float]:
        """Estimate 3D dimensions of an object based on its 2D bounding box and depth"""
        x1, y1, x2, y2 = bbox
        width_2d = x2 - x1
        height_2d = y2 - y1
        
        # These are normalized estimates - in a real application, 
        # you would use calibrated camera parameters and known object scales
        # For now, we'll use a simple proportion based approach
        
        image_width, image_height = image_size
        
        # Normalize the 2D dimensions
        norm_width = width_2d / image_width
        norm_height = height_2d / image_height
        
        # Estimate real-world dimensions based on depth (simplified)
        # In a real implementation, you would use known object dimensions or calibration
        scale_factor = depth / 10.0  # Adjust based on actual scene scale
        
        # These are estimated values - actual implementation would require 
        # either known object sizes or additional information
        width_3d = norm_width * 2.0 * scale_factor  # meters
        height_3d = norm_height * 2.0 * scale_factor  # meters
        depth_3d = norm_width * 0.5 * scale_factor   # depth assuming typical object thickness
        
        return [width_3d, height_3d, depth_3d]
    
    def anchor_objects_in_3d(self, 
                           image: Image.Image, 
                           detections: List[Dict],
                           depth_map: Optional[np.ndarray] = None) -> List[Object3DInfo]:
        """
        Anchor detected objects in 3D space using point cloud information
        
        Args:
            image: Input image
            detections: List of detections from object detection model
            depth_map: Precomputed depth map (if None, will be computed)
            
        Returns:
            List of Object3DInfo with 3D positions and dimensions
        """
        if depth_map is None:
            depth_map = self.depth_estimator.estimate_depth(image)
        
        # Generate full point cloud
        point_cloud = self.point_cloud_generator.generate_point_cloud(depth_map, image)
        
        object_3d_list = []
        
        for det in detections:
            bbox = det.get("bbox")
            if bbox is None:
                continue
            
            # Extract point cloud for this specific object
            object_points = self.point_cloud_generator.extract_object_point_cloud(point_cloud, bbox)
            
            if len(object_points) > 0:
                # Use actual point cloud data for more accurate 3D positioning
                center_3d = self.point_cloud_generator.compute_object_center_from_point_cloud(object_points)
                dimensions_3d = self.point_cloud_generator.compute_object_dimensions_from_point_cloud(object_points)
            else:
                # Fallback to 2D-based estimation if point cloud extraction failed
                obj_depth = self.compute_object_depth(depth_map, bbox)
                dimensions_3d = self.estimate_object_dimensions(bbox, obj_depth, image.size)
                
                # Calculate center in 3D space
                x1, y1, x2, y2 = bbox
                center_x_2d = (x1 + x2) / 2
                center_y_2d = (y1 + y2) / 2
                
                center_3d = [
                    (center_x_2d - image.size[0]/2) / (image.size[0]/2) * obj_depth * 0.1,  # x
                    (center_y_2d - image.size[1]/2) / (image.size[1]/2) * obj_depth * 0.1,  # y
                    obj_depth  # z
                ]
            
            object_3d_info = Object3DInfo(
                class_name=det.get("class", "unknown"),
                bbox=bbox,
                center_3d=center_3d,
                dimensions_3d=dimensions_3d,
                confidence=det.get("confidence", 0.0)
            )
            
            object_3d_list.append(object_3d_info)
        
        return object_3d_list


class Scene3DVisualizer:
    """Visualizes 3D scene with objects positioned in 3D space"""
    
    def __init__(self):
        self.objects_3d = []
    
    def visualize_3d_scene(self, objects_3d: List[Object3DInfo], output_path: str = None):
        """
        Create 3D visualization of the scene
        This is a basic implementation - in practice you'd use Three.js or similar for interactive visualization
        """
        if not objects_3d:
            logger.warning("No 3D objects to visualize")
            return None
        
        # Create a 3D scatter plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract coordinates for plotting
        xs = [obj.center_3d[0] for obj in objects_3d]
        ys = [obj.center_3d[1] for obj in objects_3d]
        zs = [obj.center_3d[2] for obj in objects_3d]  # Use depth as z-coordinate
        colors = [self._get_color_for_class(obj.class_name) for obj in objects_3d]
        
        # Create scatter plot
        scatter = ax.scatter(xs, ys, zs, c=colors, s=100, alpha=0.7)
        
        # Add labels for each object
        for obj in objects_3d:
            ax.text(obj.center_3d[0], obj.center_3d[1], obj.center_3d[2], 
                   obj.class_name, fontsize=9)
        
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters - depth)')
        ax.set_title('3D Scene Reconstruction')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"3D visualization saved to {output_path}")
        
        return fig
    
    def generate_3d_scene_data(self, objects_3d: List[Object3DInfo]) -> Dict:
        """
        Generate the 3D scene data in a format suitable for Three.js visualization
        """
        scene_data = {
            "objects_3d": [
                {
                    "class": obj.class_name,
                    "position_3d": obj.center_3d,
                    "dimensions_3d": obj.dimensions_3d,
                    "confidence": obj.confidence
                }
                for obj in objects_3d
            ]
        }
        return scene_data
    
    def _get_color_for_class(self, class_name: str) -> str:
        """Get a distinct color for each class"""
        color_map = {
            'toilet': '#FF6B6B',
            'sink': '#4ECDC4', 
            'mirror': '#45B7D1',
            'bathtub': '#96CEB4',
            'showerhead': '#FFEAA7',
            'towel': '#DDA0DD',
            'toothbrush': '#98D8C8',
            'toothpaste': '#F7DC6F',
            'soap_bar': '#BB8FCE',
            'shampoo_bottle': '#85C1E9',
            'conditioner_bottle': '#F8C471',
            'handwash_bottle': '#82E0AA',
            'toilet_paper_roll': '#F1948A',
            'towel_rack': '#85C1E9',
            'bath_mat': '#D7BDE2',
            'hair_dryer': '#F9E79F',
            'razor': '#FADBD8',
            'lotion_bottle': '#D5DBDB',
            'trash_bin': '#F6DDCC',
            'shower_curtain': '#E8DAEF',
            'comb': '#D4E6F1',
            'cleaning_brush': '#F9EBEA',
            'bucket': '#EAF2F8',
            'mug': '#FADBD8',
            'bathroom_shelf': '#D6EAF8'
        }
        return color_map.get(class_name, '#95A5A6')  # Default gray color


class AI3DSceneMapper:
    """Main class that integrates all components of the 3D scene mapper"""
    
    def __init__(self):
        self.object_anchorer = Object3DAnchorer()
        self.visualizer = Scene3DVisualizer()
    
    def create_3d_scene(self, 
                       image: Image.Image, 
                       detections: List[Dict],
                       output_path: str = None) -> Dict:
        """
        Create a 3D reconstruction of the scene from 2D image and detections
        
        Args:
            image: Input image
            detections: List of object detections from 2D detector
            output_path: Path to save 3D visualization
            
        Returns:
            Dictionary containing 3D scene information
        """
        logger.info(f"Creating 3D scene from image with {len(detections)} detections")
        
        # Anchor objects in 3D space
        objects_3d = self.object_anchorer.anchor_objects_in_3d(image, detections)
        
        # Create 3D visualization
        vis_path = None
        if output_path:
            vis_path = f"{output_path}_3d_visualization.png"
            self.visualizer.visualize_3d_scene(objects_3d, vis_path)
        
        # Generate 3D scene data for web visualization
        scene_data = self.visualizer.generate_3d_scene_data(objects_3d)
        
        # Prepare result
        result = {
            "objects_3d": [
                {
                    "class": obj.class_name,
                    "bbox_2d": obj.bbox,
                    "position_3d": obj.center_3d,
                    "dimensions_3d": obj.dimensions_3d,
                    "confidence": obj.confidence
                }
                for obj in objects_3d
            ],
            "scene_data": scene_data,  # For web-based 3D visualization
            "visualization_path": vis_path,
            "object_count": len(objects_3d),
            "success": True
        }
        
        logger.info(f"3D scene created successfully with {len(objects_3d)} objects")
        return result


# Example usage and testing function
def test_3d_scene_mapper():
    """Test function to demonstrate the 3D scene mapper"""
    from object_detection_model import HomeObjectDetectionModel, draw_bounding_boxes
    
    # Create a mapper instance
    mapper = AI3DSceneMapper()
    
    # Create a test image with synthetic objects
    test_img = Image.new('RGB', (416, 416), color='lightblue')
    draw = ImageDraw.Draw(test_img)
    
    # Draw some simple objects
    draw.rectangle([100, 100, 200, 200], fill='red', outline='black', width=3)
    draw.ellipse([250, 150, 350, 250], fill='green', outline='black', width=3)
    draw.rectangle([50, 300, 150, 380], fill='blue', outline='black', width=3)
    
    # Test detections (simulated)
    test_detections = [
        {
            "class": "toilet",
            "bbox": [100, 100, 200, 200],
            "confidence": 0.9
        },
        {
            "class": "sink", 
            "bbox": [250, 150, 350, 250],
            "confidence": 0.85
        },
        {
            "class": "mirror",
            "bbox": [50, 300, 150, 380], 
            "confidence": 0.8
        }
    ]
    
    # Create 3D scene
    result = mapper.create_3d_scene(
        image=test_img,
        detections=test_detections,
        output_path="test_output"
    )
    
    print("3D Scene Creation Result:")
    print(f"Success: {result['success']}")
    print(f"Objects in 3D: {result['object_count']}")
    print(f"Visualization saved to: {result['visualization_path']}")
    
    for obj in result['objects_3d']:
        print(f"  {obj['class']}: pos={obj['position_3d']}, dims={obj['dimensions_3d']}")


if __name__ == "__main__":
    test_3d_scene_mapper()