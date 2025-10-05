"""
FastAPI Backend for Home Object Detection
This server handles image uploads, runs object detection, and returns processed images with bounding boxes.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np
import base64
from pathlib import Path
import uuid
import logging
from typing import List, Optional
import uvicorn

# Import the object detection model
from object_detection_model import (
    HomeObjectDetectionModel, 
    draw_bounding_boxes, 
    HOME_OBJECTS,
    NUM_CLASSES
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Home Object Detection API",
    description="API for detecting home objects in images with bounding boxes",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory for serving images
app.mount("/static", StaticFiles(directory="static"), name="static")

# Create static directory if it doesn't exist
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HomeObjectDetectionModel(num_classes=NUM_CLASSES)

# Try to load the trained model, if not available, create a new one
model_paths = ["home_objects_detection_model.pth", "home_objects_cnn.pth"]
model_loaded = False

for model_path in model_paths:
    if Path(model_path).exists():
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            logger.info(f"Model loaded from {model_path}")
            model_loaded = True
            break
        except Exception as e:
            logger.warning(f"Could not load model from {model_path}: {e}")

if not model_loaded:
    logger.warning("No model file found. Using untrained model.")

model.to(device)
model.eval()

# Image transformation for inference
transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.get("/")
def read_root():
    return {
        "message": "Home Object Detection API",
        "endpoints": {
            "/detect": "POST endpoint for object detection",
            "/detect-multiple": "POST endpoint for multiple image detection"
        }
    }

@app.post("/detect")
async def detect_objects(image: UploadFile = File(...), confidence_threshold: float = Form(0.5)):
    """
    Detect objects in a single image and return the image with bounding boxes.
    """
    try:
        # Read and validate image
        contents = await image.read()
        try:
            img = Image.open(io.BytesIO(contents)).convert('RGB')
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Transform image for model
        input_tensor = transform(img).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            detections = model.predict(input_tensor, conf_thresh=confidence_threshold)
        
        # Draw bounding boxes on the original image
        result_img = draw_bounding_boxes(img, detections[0])
        
        # Save result image
        unique_id = str(uuid.uuid4())
        result_path = static_dir / f"result_{unique_id}.jpg"
        result_img.save(result_path, "JPEG", quality=95)
        
        # Prepare detection results for response
        detection_results = []
        for det in detections[0]:  # detections[0] because batch size is 1
            x_center, y_center, width, height, conf, cls_id = det
            x1 = int((x_center - width/2) * img.width)
            y1 = int((y_center - height/2) * img.height)
            x2 = int((x_center + width/2) * img.width)
            y2 = int((y_center + height/2) * img.height)
            
            detection_results.append({
                "class": HOME_OBJECTS[int(cls_id)],
                "confidence": float(conf),
                "bbox": [x1, y1, x2, y2]  # [x_min, y_min, x_max, y_max]
            })
        
        return {
            "result_image_url": f"/static/result_{unique_id}.jpg",
            "detections": detection_results,
            "image_size": [img.width, img.height]
        }
    
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/detect-multiple")
async def detect_objects_multiple(images: List[UploadFile] = File(...), confidence_threshold: float = Form(0.5)):
    """
    Detect objects in multiple images and return results for each.
    """
    results = []
    
    for image in images:
        try:
            # Read and validate image
            contents = await image.read()
            try:
                img = Image.open(io.BytesIO(contents)).convert('RGB')
            except Exception:
                results.append({
                    "filename": image.filename,
                    "error": "Invalid image file"
                })
                continue
            
            # Transform image for model
            input_tensor = transform(img).unsqueeze(0).to(device)
            
            # Run inference
            with torch.no_grad():
                detections = model.predict(input_tensor, conf_thresh=confidence_threshold)
            
            # Draw bounding boxes on the original image
            result_img = draw_bounding_boxes(img, detections[0])
            
            # Save result image
            unique_id = str(uuid.uuid4())
            result_path = static_dir / f"result_{unique_id}.jpg"
            result_img.save(result_path, "JPEG", quality=95)
            
            # Prepare detection results
            detection_results = []
            for det in detections[0]:  # detections[0] because batch size is 1
                x_center, y_center, width, height, conf, cls_id = det
                x1 = int((x_center - width/2) * img.width)
                y1 = int((y_center - height/2) * img.height)
                x2 = int((x_center + width/2) * img.width)
                y2 = int((y_center + height/2) * img.height)
                
                detection_results.append({
                    "class": HOME_OBJECTS[int(cls_id)],
                    "confidence": float(conf),
                    "bbox": [x1, y1, x2, y2]  # [x_min, y_min, x_max, y_max]
                })
            
            results.append({
                "filename": image.filename,
                "result_image_url": f"/static/result_{unique_id}.jpg",
                "detections": detection_results,
                "image_size": [img.width, img.height]
            })
            
        except Exception as e:
            logger.error(f"Error processing image {image.filename}: {e}")
            results.append({
                "filename": image.filename,
                "error": f"Error processing image: {str(e)}"
            })
    
    return {"results": results}

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": "home_objects_detection_model.pth" in [f.name for f in Path(".").iterdir() if f.is_file()]}

# Additional utility endpoints
@app.get("/classes")
def get_classes():
    """Get the list of detectable classes"""
    return {"classes": HOME_OBJECTS, "count": len(HOME_OBJECTS)}

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Set to False in production
    )