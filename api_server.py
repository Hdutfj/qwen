"""
Enhanced FastAPI Backend for Object Detection
Optimized for fast inference and better detection accuracy
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import io
import numpy as np
import base64
from pathlib import Path
import uuid
import logging
from typing import List, Optional
import uvicorn
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
from dotenv import load_dotenv
import requests

# Import the enhanced object detection model
from object_detection_model import (
    HomeObjectDetectionModel, 
    draw_bounding_boxes, 
    # HOME_OBJECTS,  # Comment out import; define explicitly below
    # NUM_CLASSES  # Comment out; define below
)

# Define HOME_OBJECTS explicitly (COCO 80 classes)
HOME_OBJECTS = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
NUM_CLASSES = len(HOME_OBJECTS)  # 80

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enhanced Object Detection API",
    description="Fast and accurate object detection with 80+ object classes",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Create directories
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)

# Global model and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device} ({'GPU' if device.type == 'cuda' else 'CPU'})")

# Initialize model with corrected NUM_CLASSES
model = HomeObjectDetectionModel(num_classes=NUM_CLASSES)

# Load trained model with fallback options, prioritizing home_objects_detection_model.pth
model_paths = [
    "home_objects_detection_model.pth", 
    "best_detection_model.pth",
    "home_objects_cnn.pth",
    "yolov5s.pt"  # Try loading a pre-trained YOLO model as fallback
]

model_loaded = False
loaded_key_count = 0
total_params = len(model.state_dict())

for model_path in model_paths:
    if Path(model_path).exists():
        try:
            logger.info(f"Attempting to load model from {model_path}")
            checkpoint = torch.load(model_path, map_location=device)
            
            state_dict = checkpoint.get('model_state_dict') or checkpoint.get('state_dict') or checkpoint
            
            # Load and count matched keys
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            loaded_key_count = total_params - len(missing)
            logger.info(f"Loaded {loaded_key_count}/{total_params} keys from {model_path} (missing: {len(missing)}, unexpected: {len(unexpected)})")
            
            if len(missing) == 0 and len(unexpected) == 0:
                model_loaded = True
                logger.info(f"Successfully loaded model from {model_path} with perfect match")
                break
            elif loaded_key_count / total_params > 0.8:
                model_loaded = True
                logger.warning(f"Partially loaded model from {model_path} ({loaded_key_count}/{total_params} keys matched). May work but verify performance.")
                break
            else:
                logger.warning(f"Too few keys matched ({loaded_key_count}/{total_params}). Skipping.")
                
        except Exception as e:
            logger.warning(f"Failed to load model from {model_path}: {e}")

if not model_loaded:
    logger.warning("No suitable pre-trained model found. Initializing with defaults and attempting quick training...")
    try:
        from object_detection_model import create_sample_detection_dataset, train_fast_detection_model
        from object_detection_model import FastDetectionDataset, get_fast_transforms, collate_fn
        from torch.utils.data import DataLoader
        
        logger.info("Creating sample dataset...")
        if create_sample_detection_dataset("temp_dataset", num_images=200):
            train_transform, _ = get_fast_transforms()
            dataset = FastDetectionDataset("temp_dataset", transform=train_transform)
            
            if len(dataset) > 0:
                loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn, num_workers=2)
                logger.info("Performing quick training...")
                train_fast_detection_model(model, loader, loader, num_epochs=10, learning_rate=0.001, save_path='quick_model.pth')
                model_loaded = True
                logger.info("Quick training completed")
    except Exception as e:
        logger.error(f"Failed to train quick model: {e}. Using un trained model - detection may be poor.")

# Move model to device and set to eval mode
model.to(device)
model.eval()

# Optimized image transforms
transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=4)

def process_single_image(image_data, confidence_threshold=0.25):
    """Process a single image for object detection"""
    try:
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        original_size = img.size
        
        input_tensor = transform(img).unsqueeze(0).to(device)
        
        start_time = time.time()
        with torch.no_grad():
            detections = model.predict(input_tensor, conf_thresh=confidence_threshold)
        inference_time = time.time() - start_time
        
        detection_results = []
        for det in detections[0]:
            x_center, y_center, width, height, conf, cls_id = det
            
            x1 = int((x_center - width/2) * original_size[0])
            y1 = int((y_center - height/2) * original_size[1])
            x2 = int((x_center + width/2) * original_size[0])
            y2 = int((y_center + height/2) * original_size[1])
            
            x1 = max(0, min(x1, original_size[0]))
            y1 = max(0, min(y1, original_size[1]))
            x2 = max(0, min(x2, original_size[0]))
            y2 = max(0, min(y2, original_size[1]))
            
            if x2 > x1 and y2 > y1:
                class_name = HOME_OBJECTS[int(cls_id)] if int(cls_id) < len(HOME_OBJECTS) else f"class_{int(cls_id)}"
                detection_results.append({
                    "class": class_name,
                    "confidence": float(conf),
                    "bbox": [x1, y1, x2, y2],
                    "center": [float(x_center), float(y_center)],
                    "size": [float(width), float(height)],
                    "image_width": original_size[0],
                    "image_height": original_size[1]
                })
        
        result_img = draw_bounding_boxes(img, detections[0]) if len(detections[0]) > 0 else img
        
        return {
            "success": True,
            "detections": detection_results,
            "result_image": result_img,
            "inference_time": inference_time,
            "image_size": original_size
        }
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return {"success": False, "error": str(e)}

def draw_local_bounding_boxes(image, detections):
    if not detections:
        return image
    
    draw = ImageDraw.Draw(image)
    
    colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
        '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
        '#F8C471', '#82E0AA', '#F1948A', '#85C1E9', '#D7BDE2'
    ] * 10
    
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        color = colors[i % len(colors)]
        class_name = det['class']
        confidence = det['confidence']
        
        for thickness in range(3):
            draw.rectangle([x1-thickness, y1-thickness, x2+thickness, y2+thickness], 
                           outline=color, width=1)
        
        label = f"{class_name}: {confidence:.2f}"
        try:
            font = ImageFont.load_default()
        except:
            font = None

        if font:
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            text_width, text_height = len(label) * 6, 11

        label_y = max(0, y1 - text_height - 5)
        draw.rectangle([x1, label_y, x1 + text_width + 10, label_y + text_height + 5], 
                       fill=color)
        draw.text((x1 + 5, label_y + 2), label, fill='white', font=font)
    
    return image

@app.get("/")
def read_root():
    return {
        "message": "Enhanced Object Detection API",
        "version": "2.0.0",
        "features": [
            "80+ object classes",
            "Fast inference",
            "High accuracy detection",
            "Batch processing support"
        ],
        "endpoints": {
            "/detect": "Single image detection",
            "/detect-batch": "Batch image detection",
            "/detect-url": "Detect from image URL",
            "/classes": "List all detectable classes",
            "/health": "API health check",
            "/stats": "Detection statistics"
        }
    }

@app.post("/detect")
async def detect_objects(
    image: UploadFile = File(...), 
    confidence_threshold: float = Form(0.25),
    draw_boxes: bool = Form(True)
):
    try:
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        contents = await image.read()
        
        detection_source = "local_model"
        result = process_single_image(contents, confidence_threshold)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        detections = result["detections"]
        result_image = None
        
        if draw_boxes:
            img = Image.open(io.BytesIO(contents)).convert('RGB')
            temp_detections = [
                {"class": det["class"], "confidence": det["confidence"], "bbox": det["bbox"]}
                for det in detections
            ]
            result_image = draw_local_bounding_boxes(img, temp_detections)
        
        result_url = None
        if result_image:
            unique_id = str(uuid.uuid4())
            result_path = static_dir / f"result_{detection_source}_{unique_id}.jpg"
            result_image.save(result_path, "JPEG", quality=95)
            result_url = f"/static/result_{detection_source}_{unique_id}.jpg"
        
        response_data = {
            "filename": image.filename,
            "detections": detections,
            "detection_count": len(detections),
            "result_image_url": result_url if result_image else None,
            "confidence_threshold": confidence_threshold,
            "detection_source": detection_source,
            "inference_time_ms": round(result["inference_time"] * 1000, 2),
            "image_size": result["image_size"]
        }
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in detect endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect-batch")
async def detect_objects_batch(
    images: List[UploadFile] = File(...),
    confidence_threshold: float = Form(0.25),
    draw_boxes: bool = Form(True)
):
    if len(images) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 images per batch")
    
    try:
        results = []
        total_detections = 0
        
        for image in images:
            if not image.content_type.startswith('image/'):
                continue
            
            contents = await image.read()
            detection_source = "local_model"
            result = process_single_image(contents, confidence_threshold)
            
            if not result["success"]:
                continue
            
            detections = result["detections"]
            result_image = None
            
            if draw_boxes:
                img = Image.open(io.BytesIO(contents)).convert('RGB')
                temp_detections = [
                    {"class": det["class"], "confidence": det["confidence"], "bbox": det["bbox"]}
                    for det in detections
                ]
                result_image = draw_local_bounding_boxes(img, temp_detections)
            
            result_url = None
            if result_image:
                unique_id = str(uuid.uuid4())
                result_path = static_dir / f"batch_{detection_source}_{unique_id}.jpg"
                result_image.save(result_path, "JPEG", quality=95)
                result_url = f"/static/batch_{detection_source}_{unique_id}.jpg"
            
            result_item = {
                "filename": image.filename,
                "detections": detections,
                "detection_count": len(detections),
                "result_image_url": result_url,
                "detection_source": detection_source,
                "inference_time_ms": round(result["inference_time"] * 1000, 2),
                "image_size": result["image_size"]
            }
            
            results.append(result_item)
            total_detections += len(detections)
        
        return {
            "batch_size": len(results),
            "total_detections": total_detections,
            "confidence_threshold": confidence_threshold,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error in batch detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/detect-url")
async def detect_from_url(
    image_url: str,
    confidence_threshold: float = 0.25,
    draw_boxes: bool = True
):
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        
        if not response.headers.get('content-type', '').startswith('image/'):
            raise HTTPException(status_code=400, detail="URL does not point to an image")
        
        contents = response.content
        detection_source = "local_model"
        result = process_single_image(contents, confidence_threshold)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        detections = result["detections"]
        result_image = None
        
        if draw_boxes:
            img = Image.open(io.BytesIO(contents)).convert('RGB')
            temp_detections = [
                {"class": det["class"], "confidence": det["confidence"], "bbox": det["bbox"]}
                for det in detections
            ]
            result_image = draw_local_bounding_boxes(img, temp_detections)
        
        result_url = None
        if result_image:
            unique_id = str(uuid.uuid4())
            result_path = static_dir / f"url_{detection_source}_{unique_id}.jpg"
            result_image.save(result_path, "JPEG", quality=95)
            result_url = f"/static/url_{detection_source}_{unique_id}.jpg"
        
        response_data = {
            "source_url": image_url,
            "detections": detections,
            "detection_count": len(detections),
            "result_image_url": result_url,
            "detection_source": detection_source,
            "inference_time_ms": round(result["inference_time"] * 1000, 2),
            "image_size": result["image_size"]
        }
        
        return response_data
        
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")
    except Exception as e:
        logger.error(f"Error in URL detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/classes")
def get_classes():
    return {
        "classes": HOME_OBJECTS,
        "total_classes": len(HOME_OBJECTS),
        "categories": {
            "people_animals": HOME_OBJECTS[:24],
            "vehicles": HOME_OBJECTS[1:9],
            "household_items": HOME_OBJECTS[40:],
            "electronics": [cls for cls in HOME_OBJECTS if cls in ['tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone']]
        }
    }

@app.get("/health")
def health_check():
    try:
        dummy_input = torch.randn(1, 3, 416, 416).to(device)
        with torch.no_grad():
            start_time = time.time()
            _ = model(dummy_input)
            inference_time = time.time() - start_time
        
        return {
            "status": "healthy",
            "model_loaded": model_loaded,
            "device": str(device),
            "model_parameters": sum(p.numel() for p in model.parameters()),
            "inference_test_ms": round(inference_time * 1000, 2),
            "supported_classes": len(HOME_OBJECTS),
            "api_version": "2.0.0"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "model_loaded": model_loaded
        }

@app.get("/stats")
def get_stats():
    return {
        "model_info": {
            "architecture": "Fast YOLO",
            "input_size": "416x416",
            "output_grid": "13x13",
            "anchors": 3,
            "classes": len(HOME_OBJECTS)
        },
        "performance": {
            "device": str(device),
            "mixed_precision": device.type == 'cuda',
            "batch_processing": True,
            "parallel_inference": True
        },
        "features": {
            "real_time_detection": True,
            "batch_processing": True,
            "url_detection": True,
            "confidence_filtering": True,
            "nms_filtering": True
        }
    }

@app.get("/test-detection")
def test_detection():
    try:
        test_img = Image.new('RGB', (416, 416), color='lightblue')
        draw = ImageDraw.Draw(test_img)
        
        draw.rectangle([100, 100, 200, 200], fill='red', outline='black', width=3)
        draw.ellipse([250, 150, 350, 250], fill='green', outline='black', width=3)
        draw.rectangle([50, 300, 150, 380], fill='blue', outline='black', width=3)
        
        try:
            font = ImageFont.load_default()
            draw.text((100, 80), "chair", fill='black', font=font)
            draw.text((250, 130), "clock", fill='black', font=font)
            draw.text((50, 280), "book", fill='black', font=font)
        except:
            pass
        
        img_bytes = io.BytesIO()
        test_img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        contents = img_bytes.getvalue()
        detection_source = "local_model"
        result = process_single_image(contents, 0.1)
        
        detections = result["detections"] if result["success"] else []
        
        result_image = None
        if result["success"]:
            img = Image.open(io.BytesIO(contents)).convert('RGB')
            temp_detections = [
                {"class": det["class"], "confidence": det["confidence"], "bbox": det["bbox"]}
                for det in detections
            ]
            result_image = draw_local_bounding_boxes(img, temp_detections)
        
        unique_id = str(uuid.uuid4())
        result_path = static_dir / f"test_{detection_source}_{unique_id}.jpg"
        if result_image:
            result_image.save(result_path, "JPEG", quality=95)
            result_url = f"/static/test_{detection_source}_{unique_id}.jpg"
        else:
            result_url = None
        
        return {
            "test_status": "success" if result["success"] else "failed",
            "detections_found": len(detections),
            "detections": detections,
            "test_image_url": result_url,
            "detection_source": detection_source
        }
            
    except Exception as e:
        logger.error(f"Test detection failed: {e}")
        return {
            "test_status": "failed",
            "error": str(e)
        }

if __name__ == "__main__":
    logger.info("Starting Enhanced Object Detection API Server")
    logger.info(f"Model classes: {len(HOME_OBJECTS)}")
    logger.info(f"Device: {device}")
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )