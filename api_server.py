"""
Enhanced FastAPI Backend for Object Detection
Supports both OpenAI API and local enhanced detection model with smart features
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
import requests
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
import io
import numpy as np
import base64
from pathlib import Path
import uuid
import json
import logging
from typing import List, Optional
import uvicorn
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
from dotenv import load_dotenv
from base64 import b64encode
import torch
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from pydantic import BaseModel, EmailStr
from enhanced_detection_model import HomeObjectDetectionModel, draw_bounding_boxes
from api_intelligence_extension import extend_api_with_intelligence_features
from depth_3d_mapper import AI3DSceneMapper

# Define home objects
HOME_OBJECTS = [
    'toilet', 'sink', 'mirror', 'bathtub', 'showerhead', 'towel', 'toothbrush', 
    'toothpaste', 'soap_bar', 'shampoo_bottle', 'conditioner_bottle', 'handwash_bottle', 
    'toilet_paper_roll', 'towel_rack', 'bath_mat', 'hair_dryer', 'razor', 'lotion_bottle', 
    'trash_bin', 'shower_curtain', 'comb', 'cleaning_brush', 'bucket', 'mug', 'bathroom_shelf'
]

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Security constants
SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key_here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Security instances
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load enhanced detection model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enhanced_model = None
model_path = "enhanced_detection_model.pth"

if os.path.exists(model_path):
    try:
        enhanced_model = HomeObjectDetectionModel()
        enhanced_model.load_state_dict(torch.load(model_path, map_location=device))
        enhanced_model.to(device)
        logger.info("Enhanced detection model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading enhanced detection model: {e}")

if not openai_client.api_key and enhanced_model is None:
    logger.error("No detection models available - please provide OPENAI_API_KEY or ensure enhanced model is trained")
    raise ValueError("No detection models available")

app = FastAPI(
    title="Enhanced Object Detection API",
    description="Object detection using OpenAI API",
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

# Thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=4)

def process_single_image_openai(img: Image.Image):
    """Process a single PIL Image for object detection using OpenAI API"""
    try:
        start_time = time.time()
        
        # Convert PIL image to base64 for API request
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=95)
        img_base64 = b64encode(buffer.getvalue()).decode('utf-8')
        
        # Call OpenAI API for image analysis
        response = openai_client.chat.completions.create(
            model="gpt-4o",  # Using GPT-4 Omni model which includes vision capabilities
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": "Analyze this image in detail and identify each object with its specific location. Return a JSON response with the following structure: {\"objects\": [{\"name\": \"object_name\", \"confidence\": 0.95, \"position\": {\"x\": 0.3, \"y\": 0.7, \"width\": 0.2, \"height\": 0.3}}]}. The position coordinates should be normalized between 0 and 1, where (0,0) is top-left and (1,1) is bottom-right of the image. For example, x=0.3 means 30% from the left edge, y=0.7 means 70% from the top edge. Common objects to look for include: chair, table, sofa, bed, desk, lamp, mirror, window, door, curtain, bookshelf, TV, refrigerator, microwave, stove, sink, toilet, bathtub, shower, towel, pillow, painting, plant, cup, bottle, and any other common household items. Only respond with the JSON object and nothing else."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500,
            temperature=0.1  # Lower temperature for more consistent results
        )
        
        # Extract the response
        result_text = response.choices[0].message.content
        logger.info(f"OpenAI response: {result_text}")
        
        try:
            # Try to parse the JSON response
            result = json.loads(result_text)
            detected_objects = result.get("objects", [])
        except json.JSONDecodeError:
            # If the response wasn't valid JSON, try to extract object list from text
            detected_objects = []
            # Try to handle various response formats
            import re
            # Look for JSON-like structures in the response
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                try:
                    partial_result = json.loads(json_match.group())
                    detected_objects = partial_result.get("objects", [])
                except json.JSONDecodeError:
                    # If still can't parse, try simpler approach
                    # Look for items in a list format like item1, item2, item3
                    detected_objects = [obj.strip().strip('"\'') for obj in result_text.split(',') if obj.strip()]
            else:
                # If no JSON found, just return the text as a single object
                detected_objects = [{"name": result_text.strip(), "confidence": 0.5}]
        
        # Format as detection results (with potential bounding box data if available)
        detection_results = []
        for obj in detected_objects:
            if isinstance(obj, dict):
                # If obj is a dictionary with name, confidence, and position
                class_name = obj.get("name", "")
                confidence = obj.get("confidence", 0.85)
                
                # Get position coordinates if available
                position_data = obj.get("position", {})
                if isinstance(position_data, dict) and all(k in position_data for k in ["x", "y", "width", "height"]):
                    # We have detailed position data
                    pos_x = position_data["x"]
                    pos_y = position_data["y"]
                    pos_width = position_data["width"]
                    pos_height = position_data["height"]
                    
                    # Convert normalized coordinates to pixel coordinates
                    try:
                        x1 = int(pos_x * img.width)
                        y1 = int(pos_y * img.height)
                        box_width = int(pos_width * img.width)
                        box_height = int(pos_height * img.height)
                        x2 = x1 + box_width
                        y2 = y1 + box_height
                        
                        # Ensure coordinates are within image bounds
                        x1 = max(0, min(x1, img.width))
                        y1 = max(0, min(y1, img.height))
                        x2 = max(0, min(x2, img.width))
                        y2 = max(0, min(y2, img.height))
                        
                        # Calculate center and size
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        size_w = x2 - x1
                        size_h = y2 - y1
                        
                        # Calculate area percentage
                        object_area = size_w * size_h
                        total_area = img.width * img.height
                        area_percentage = (object_area / total_area) * 100 if total_area > 0 else 0
                        
                        detection_results.append({
                            "class": class_name,
                            "confidence": confidence,
                            "bbox": [x1, y1, x2, y2],  # Bounding box in pixel coordinates
                            "center": [center_x, center_y],
                            "size": [size_w, size_h],
                            "area_percentage": area_percentage,
                            "image_width": img.width,
                            "image_height": img.height
                        })
                    except Exception as coord_error:
                        logger.error(f"Error processing coordinates: {coord_error}")
                        # Fallback if coordinate processing fails
                        detection_results.append({
                            "class": class_name,
                            "confidence": confidence,
                            "bbox": None,
                            "center": None,
                            "size": None,
                            "area_percentage": None,
                            "image_width": img.width,
                            "image_height": img.height
                        })
                else:
                    # Fallback if no detailed position data
                    detection_results.append({
                        "class": class_name,
                        "confidence": confidence,
                        "bbox": None,
                        "center": None,
                        "size": None,
                        "area_percentage": None,
                        "image_width": img.width,
                        "image_height": img.height
                    })
            else:
                # If obj is just a string
                class_name = str(obj)
                confidence = 0.85  # Default confidence
                
                detection_results.append({
                    "class": class_name,
                    "confidence": confidence,
                    "bbox": None,
                    "center": None,
                    "size": None,
                    "area_percentage": None,
                    "image_width": img.width,
                    "image_height": img.height
                })
        
        inference_time = time.time() - start_time
        
        return {
            "success": True,
            "detections": detection_results,
            "result_image": img,  # Return original image since we can't draw bounding boxes
            "inference_time": inference_time,
            "image_size": (img.width, img.height)
        }
        
    except Exception as e:
        logger.error(f"Error processing image with OpenAI API: {e}")
        return {"success": False, "error": str(e)}

def process_single_image_enhanced(img: Image.Image):
    """Process a single PIL Image using the enhanced local model with smart features"""
    if enhanced_model is None:
        return {"success": False, "error": "Enhanced model not available"}
    
    try:
        start_time = time.time()
        
        # Preprocess image for model
        transform = torch.nn.Sequential(
            torch.nn.Resize((416, 416)),
            torch.nn.ToTensor(),
            torch.nn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
        
        # Convert PIL to tensor and add batch dimension
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Run enhanced prediction
        detections = enhanced_model.predict(img_tensor)
        
        # Process detections into expected format
        detection_results = []
        
        # The detections format may vary, so we adjust accordingly
        if detections:
            for batch_dets in detections:
                # Remove relationships dict if present for processing
                relationships = None
                fine_grained_info = None
                if len(batch_dets) > 0 and isinstance(batch_dets[-1], dict):
                    last_item = batch_dets[-1]
                    if "relationships" in last_item:
                        relationships = batch_dets.pop()
                    elif "fine_grained_classification" in last_item:
                        fine_grained_info = batch_dets.pop()
                
                for det in batch_dets:
                    if isinstance(det, (list, tuple)) and len(det) >= 6:
                        x_center, y_center, width, height, conf, cls_id = det
                        
                        # Convert normalized coordinates to pixel coordinates
                        x1 = int((x_center - width/2) * img.width)
                        y1 = int((y_center - height/2) * img.height)
                        x2 = int((x_center + width/2) * img.width)
                        y2 = int((y_center + height/2) * img.height)
                        
                        # Calculate center and size
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        size_w = x2 - x1
                        size_h = y2 - y1
                        
                        # Calculate area percentage
                        object_area = size_w * size_h
                        total_area = img.width * img.height
                        area_percentage = (object_area / total_area) * 100 if total_area > 0 else 0
                        
                        # Get class name from extended classes
                        class_name = "unknown"
                        if isinstance(cls_id, torch.Tensor):
                            cls_id = cls_id.item()
                        cls_id = int(cls_id)
                        if cls_id < len(enhanced_model.EXTENDED_CLASSES if hasattr(enhanced_model, 'EXTENDED_CLASSES') else []):
                            class_name = enhanced_model.EXTENDED_CLASSES[cls_id]
                        elif cls_id < len(HOME_OBJECTS):
                            class_name = HOME_OBJECTS[cls_id]
                        else:
                            class_name = f"class_{cls_id}"
                        
                        detection_result = {
                            "class": class_name,
                            "confidence": conf,
                            "bbox": [x1, y1, x2, y2],  # Bounding box in pixel coordinates
                            "center": [center_x, center_y],
                            "size": [size_w, size_h],
                            "area_percentage": area_percentage,
                            "image_width": img.width,
                            "image_height": img.height
                        }
                        
                        # Add relationship info if available
                        if relationships:
                            detection_result["relationships"] = relationships
                        
                        # Add fine-grained info if available
                        if fine_grained_info:
                            detection_result["fine_grained_info"] = fine_grained_info
                        
                        detection_results.append(detection_result)
        
        inference_time = time.time() - start_time
        
        return {
            "success": True,
            "detections": detection_results,
            "result_image": img,
            "inference_time": inference_time,
            "image_size": (img.width, img.height),
            "model_type": "enhanced_local"
        }
    
    except Exception as e:
        logger.error(f"Error processing image with enhanced model: {e}")
        return {"success": False, "error": str(e)}

def draw_local_bounding_boxes(image, detections, draw_external_labels=True, output_quality=95):
    """Draw bounding boxes and labels on image based on detections for the enhanced model"""
    if not detections:
        return image
    
    # Work with a copy of the image
    # Increase image size to have space for external labels
    if draw_external_labels:
        # Create a new image with additional space on the right for labels
        new_width = int(image.width * 1.3)  # 30% more width for labels
        result_image = Image.new('RGB', (new_width, image.height), (240, 240, 240))  # Light gray background
        result_image.paste(image, (0, 0))
    else:
        result_image = image.copy()
    
    draw = ImageDraw.Draw(result_image)
    
    # Define a color palette for different objects
    colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
        '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
        '#F8C471', '#82E0AA', '#F1948A', '#85C1E9', '#D7BDE2'
    ]
    
    # Pre-calculate text sizes and positions for external labels
    external_label_info = []
    
    for i, det in enumerate(detections):
        # Get color for this object
        color = colors[i % len(colors)]
        class_name = det.get("class", "Unknown")
        confidence = det.get("confidence", 0.0)
        label_text = f"{class_name}: {confidence:.2f}"
        
        # Check if we have actual bounding box coordinates
        bbox = det.get("bbox", None)
        
        if bbox is not None and len(bbox) == 4:
            # We have actual coordinates from detection
            x1, y1, x2, y2 = bbox
            # Ensure coordinates are within original image bounds
            x1 = max(0, min(x1, image.width))
            y1 = max(0, min(y1, image.height))
            x2 = max(0, min(x2, image.width))
            y2 = max(0, min(y2, image.height))
        else:
            # Fallback to a grid layout if no coordinates available
            grid_rows = int(len(detections) ** 0.5) + 1
            grid_cols = grid_rows
            row = i // grid_cols
            col = i % grid_cols
            
            width, height = image.size
            x = (col + 1) * width // (grid_cols + 1)
            y = (row + 1) * height // (grid_rows + 1)
            
            # Draw bounding box (using a fixed size since we don't have coordinates)
            box_size = 60
            x1, y1 = x - box_size//2, y - box_size//2
            x2, y2 = x + box_size//2, y + box_size//2
        
        # Draw rectangle with multiple outlines for visibility
        for thickness in range(1, 4):
            draw.rectangle([x1-thickness, y1-thickness, x2+thickness, y2+thickness], 
                           outline=color, width=1)
        
        if draw_external_labels:
            # Calculate text size for external label with better quality font
            try:
                font_size = 14  # Larger font for better readability
                # Try to use a better font if available
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except:
                    try:
                        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
                    except:
                        font = ImageFont.load_default()
            except:
                font = None

            if font:
                bbox_text = draw.textbbox((0, 0), label_text, font=font)
                text_width = bbox_text[2] - bbox_text[0]
                text_height = bbox_text[3] - bbox_text[1]
            else:
                text_width, text_height = len(label_text) * 6, 11
            
            # Calculate position for external label on the right side
            label_x = image.width + 20  # Start after original image
            label_y = 20 + i * (text_height + 15)  # Space labels vertically
            
            # Store information for drawing arrows later
            external_label_info.append({
                'bbox_center': ((x1 + x2) // 2, (y1 + y2) // 2),
                'label_pos': (label_x, label_y),
                'label_text': label_text,
                'text_size': (text_width, text_height),
                'color': color
            })
        else:
            # For internal labels (fallback), use the original approach with better quality font
            try:
                font_size = 14  # Larger font for better readability
                # Try to use a better font if available
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except:
                    try:
                        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
                    except:
                        font = ImageFont.load_default()
            except:
                font = None

            if font:
                bbox_text = draw.textbbox((0, 0), label_text, font=font)
                text_width = bbox_text[2] - bbox_text[0]
                text_height = bbox_text[3] - bbox_text[1]
            else:
                text_width, text_height = len(label_text) * 6, 11

            # Position label near the top-left of the bounding box
            label_x = max(0, x1)
            label_y = max(0, y1 - text_height - 5)
            
            # Draw background rectangle for label
            draw.rectangle([label_x, label_y, label_x + text_width + 10, label_y + text_height + 5], 
                           fill=color)
            # Draw text
            draw.text((label_x + 5, label_y + 2), label_text, fill='white', font=font)
    
    # Draw arrows and external labels if enabled
    if draw_external_labels and external_label_info:
        for info in external_label_info:
            # Draw arrow from bbox center to label
            start_x, start_y = info['bbox_center']
            end_x, end_y = info['label_pos'][0] - 10, info['label_pos'][1] + info['text_size'][1] // 2  # Arrow points to left edge of text
            
            # Draw arrow line
            draw.line([start_x, start_y, end_x, end_y], fill=info['color'], width=2)
            
            # Draw arrowhead
            # Calculate direction vector
            dx = end_x - start_x
            dy = end_y - start_y
            length = max(1, (dx**2 + dy**2)**0.5)
            unit_dx = dx / length
            unit_dy = dy / length
            
            # Arrowhead points
            head_length = 10
            head_width = 6
            
            # Calculate perpendicular vector for arrowhead
            perp_dx = -unit_dy
            perp_dy = unit_dx
            
            # Points for arrowhead
            point1 = (end_x - head_length * unit_dx + head_width * perp_dx, 
                      end_y - head_length * unit_dy + head_width * perp_dy)
            point2 = (end_x - head_length * unit_dx - head_width * perp_dx, 
                      end_y - head_length * unit_dy - head_width * perp_dy)
            
            draw.line([end_x, end_y, point1[0], point1[1]], fill=info['color'], width=2)
            draw.line([end_x, end_y, point2[0], point2[1]], fill=info['color'], width=2)
            
            # Draw label with background
            label_x, label_y = info['label_pos']
            text_width, text_height = info['text_size']
            
            # Draw background rectangle for label
            padding = 8
            draw.rectangle([label_x, label_y, label_x + text_width + 2*padding, label_y + text_height + padding], 
                           fill=info['color'], outline='black', width=1)
            
            # Draw text with better quality font
            try:
                font_size = 14  # Larger font for better readability
                # Try to use a better font if available
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except:
                    try:
                        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
                    except:
                        font = ImageFont.load_default()
            except:
                font = None
            draw.text((label_x + padding, label_y + padding//2), info['label_text'], fill='white', font=font)
    
    return result_image

@app.get("/")
def read_root():
    return {
        "message": "Enhanced Object Detection API with Smart Features",
        "version": "3.0.0",
        "features": [
            "Bathroom object detection",
            "Fast inference",
            "High accuracy detection",
            "Single and batch processing support",
            "Auto-Label Refinement",
            "Multi-Object Context Awareness",
            "Fine-Grained Classification",
            "Self-Improving Feedback Loop"
        ],
        "endpoints": {
            "/detect": "Single image detection (with method selection)",
            "/detect-batch": "Batch image detection",
            "/detect-url": "Detect from image URL",
            "/classes": "List all detectable classes",
            "/health": "API health check",
            "/stats": "Detection statistics",
            "/smart-features": "Information about smart detection features",
            "/chatbot": "AI-related chatbot query"
        }
    }

@app.post("/detect")
async def detect_objects(
    request: Request,
    image: Optional[UploadFile] = File(None),
    # Remove confidence_threshold since OpenAI doesn't use it
    draw_boxes: bool = Form(True),
    detection_method: str = Form("auto")  # "openai", "enhanced", or "auto"
):
    """
    Detect objects in an uploaded image using OpenAI API or enhanced local model.
    Handles both file uploads and raw binary image input.
    """
    # Pre-read request body if image is None to avoid "Stream consumed" error in exception handling
    if image is None:
        try:
            contents = await request.body()
            if not contents:
                raise HTTPException(status_code=400, detail="No image data received.")
            image_name = f"raw_upload_{uuid.uuid4().hex}.jpg"
        except Exception as e:
            logger.error(f"Error reading request body in detect endpoint: {e}")
            raise HTTPException(status_code=400, detail="Failed to read image data from request body.")
    else:
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
        
        # Read file bytes ONCE and store them in memory
        try:
            contents = await image.read()
            image_name = image.filename
        except Exception as e:
            logger.error(f"Error reading uploaded file in detect endpoint: {e}")
            raise HTTPException(status_code=400, detail="Failed to read uploaded image file.")

    try:
        # Convert bytes → image once
        try:
            img = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file or corrupted data.")

        # Determine which detection method to use
        if detection_method == "auto":
            # Use OpenAI if available, otherwise use enhanced model
            if openai_client.api_key:
                result = process_single_image_openai(img)
                detection_source = "openai"
            elif enhanced_model:
                result = process_single_image_enhanced(img)
                detection_source = "enhanced_local"
            else:
                raise HTTPException(status_code=500, detail="No detection models available")
        elif detection_method == "openai":
            if not openai_client.api_key:
                raise HTTPException(status_code=500, detail="OpenAI API key not configured")
            result = process_single_image_openai(img)
            detection_source = "openai"
        elif detection_method == "enhanced":
            if not enhanced_model:
                raise HTTPException(status_code=500, detail="Enhanced model not available")
            result = process_single_image_enhanced(img)
            detection_source = "enhanced_local"
        else:
            raise HTTPException(status_code=400, detail="Invalid detection method. Use 'openai', 'enhanced', or 'auto'")

        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])

        detections = result["detections"]

        # --- Draw bounding boxes ---
        if draw_boxes:
            if detection_source == "enhanced_local":
                result_image = draw_local_bounding_boxes(img.copy(), detections)
            else:
                result_image = draw_local_bounding_boxes(img.copy(), detections)
        else:
            result_image = img

        # Save final image
        result_url = None
        if result_image:
            static_dir.mkdir(exist_ok=True)
            unique_id = str(uuid.uuid4())
            result_path = static_dir / f"result_{unique_id}.jpg"
            result_image.save(result_path, "JPEG", quality=100, optimize=True, subsampling=0)
            result_url = f"/static/result_{unique_id}.jpg"

        # Extract unique object names from detections
        detected_objects = list(set(det["class"] for det in detections))
        
        return {
            "filename": image_name,
            "detections": detections,
            "detection_count": len(detections),
            "detected_objects": detected_objects,  # List of unique objects found
            "result_image_url": result_url,
            "inference_time_ms": round(result["inference_time"] * 1000, 2),
            "image_size": result["image_size"],
            "detection_source": detection_source
        }

    except HTTPException:
        raise
    except Exception as e:
        # Log the original exception - we don't access request here to avoid stream consumption issues
        logger.error(f"Error in detect endpoint after reading image data: {type(e).__name__} - {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/detect-batch")
async def detect_objects_batch(
    images: List[UploadFile] = File(...),
    draw_boxes: bool = Form(True),
    detection_method: str = Form("auto")  # Add method selection to batch endpoint too
):
    if not images:
        raise HTTPException(status_code=400, detail="At least one image file is required")
    
    if len(images) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 images per batch")
    
    try:
        results = []
        total_detections = 0
        
        for image in images:
            if not image.content_type.startswith('image/'):
                logger.warning(f"Skipping {image.filename}: Invalid content type {image.content_type}")
                continue
            
            contents = await image.read()
            try:
                img = Image.open(io.BytesIO(contents)).convert('RGB')
            except Exception:
                logger.warning(f"Failed to open {image.filename}: Invalid image file")
                continue
            
            # Determine which detection method to use
            if detection_method == "auto":
                # Use OpenAI if available, otherwise use enhanced model
                if openai_client.api_key:
                    result = process_single_image_openai(img)
                    detection_source = "openai"
                elif enhanced_model:
                    result = process_single_image_enhanced(img)
                    detection_source = "enhanced_local"
                else:
                    logger.error(f"No detection models available for {image.filename}")
                    continue
            elif detection_method == "openai":
                if not openai_client.api_key:
                    logger.error(f"OpenAI API key not configured for {image.filename}")
                    continue
                result = process_single_image_openai(img)
                detection_source = "openai"
            elif detection_method == "enhanced":
                if not enhanced_model:
                    logger.error(f"Enhanced model not available for {image.filename}")
                    continue
                result = process_single_image_enhanced(img)
                detection_source = "enhanced_local"
            else:
                logger.error(f"Invalid detection method for {image.filename}")
                continue
            
            if not result["success"]:
                logger.warning(f"Failed to process {image.filename}: {result['error']}")
                continue
            
            detections = result["detections"]
            result_image = None
            
            if draw_boxes:
                temp_detections = [
                    {"class": det["class"], "confidence": det["confidence"], "bbox": det["bbox"]}
                    for det in detections
                ]
                # Choose drawing method based on detection source
                if detection_source == "enhanced_local":
                    result_image = draw_local_bounding_boxes(img.copy(), temp_detections)
                else:
                    result_image = draw_local_bounding_boxes(img.copy(), temp_detections)
            
            result_url = None
            if result_image:
                unique_id = str(uuid.uuid4())
                result_path = static_dir / f"batch_{detection_source}_{unique_id}.jpg"
                result_image.save(result_path, "JPEG", quality=100, optimize=True, subsampling=0)
                result_url = f"/static/batch_{detection_source}_{unique_id}.jpg"
            
            detected_objects = list(set(det["class"] for det in detections))
            
            result_item = {
                "filename": image.filename,
                "detections": detections,
                "detected_objects": detected_objects,
                "detection_count": len(detections),
                "result_image_url": result_url,
                "detection_source": detection_source,
                "inference_time_ms": round(result["inference_time"] * 1000, 2),
                "image_size": result["image_size"]
            }
            
            results.append(result_item)
            total_detections += len(detections)
        
        if not results:
            raise HTTPException(status_code=400, detail="No valid images processed")
        
        return {
            "batch_size": len(results),
            "total_detections": total_detections,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error in batch detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/detect-url")
async def detect_from_url(
    image_url: str,
    draw_boxes: bool = True
):
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        
        if not response.headers.get('content-type', '').startswith('image/'):
            raise HTTPException(status_code=400, detail="URL does not point to an image")
        
        contents = response.content
        try:
            img = Image.open(io.BytesIO(contents)).convert('RGB')
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file from URL")
        
        result = process_single_image_openai(img)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        detections = result["detections"]
        result_image = None
        
        if draw_boxes:
            temp_detections = [
                {"class": det["class"], "confidence": det["confidence"], "bbox": det["bbox"]}
                for det in detections
            ]
            result_image = draw_local_bounding_boxes(img.copy(), temp_detections)
        
        result_url = None
        if result_image:
            unique_id = str(uuid.uuid4())
            result_path = static_dir / f"url_openai_{unique_id}.jpg"
            result_image.save(result_path, "JPEG", quality=100, optimize=True, subsampling=0)
            result_url = f"/static/url_openai_{unique_id}.jpg"
        
        detected_objects = list(set(det["class"] for det in detections))
        
        response_data = {
            "source_url": image_url,
            "detections": detections,
            "detected_objects": detected_objects,
            "detection_count": len(detections),
            "result_image_url": result_url,
            "detection_source": "openai",
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
        "service": "OpenAI Vision API",
        "description": "Using OpenAI's GPT-4 Vision model for general object detection",
        "capabilities": [
            "General object recognition",
            "Scene understanding",
            "Text in images",
            "Color detection",
            "Shape identification"
        ]
    }

@app.get("/health")
def health_check():
    try:
        start_time = time.time()
        
        # Test OpenAI API connection if available
        openai_status = None
        openai_inference_time = None
        if openai_client.api_key:
            try:
                # Create a dummy 1x1 pixel image for testing
                dummy_img = Image.new('RGB', (10, 10), color='white')
                result = process_single_image_openai(dummy_img)
                openai_inference_time = time.time() - start_time
                openai_status = result["success"]
            except Exception:
                openai_status = False
        
        # Test enhanced model if available
        enhanced_status = None
        enhanced_inference_time = None
        if enhanced_model:
            try:
                test_img = Image.new('RGB', (416, 416), color='white')
                result = process_single_image_enhanced(test_img)
                enhanced_inference_time = time.time() - start_time
                enhanced_status = result["success"]
            except Exception:
                enhanced_status = False
        
        return {
            "status": "healthy",
            "openai_api_key_set": bool(openai_client.api_key),
            "openai_available": openai_status,
            "openai_inference_test_ms": round(openai_inference_time * 1000, 2) if openai_inference_time else None,
            "enhanced_model_available": enhanced_status,
            "enhanced_inference_test_ms": round(enhanced_inference_time * 1000, 2) if enhanced_inference_time else None,
            "api_version": "3.0.0"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "openai_api_key_set": bool(openai_client.api_key),
            "enhanced_model_loaded": enhanced_model is not None
        }

@app.get("/stats")
def get_stats():
    return {
        "model_info": {
            "openai_service": "OpenAI GPT-4 Vision",
            "openai_provider": "OpenAI",
            "openai_detection_method": "Cloud-based image analysis",
            "enhanced_detection_model": "Local CNN with smart features",
            "enhanced_features": [
                "Auto-Label Refinement",
                "Multi-Object Context Awareness", 
                "Fine-Grained Classification",
                "Self-Improving Feedback Loop"
            ]
        },
        "performance": {
            "openai_api_based": openai_client.api_key is not None,
            "local_model_available": enhanced_model is not None,
            "network_required_for_openai": openai_client.api_key is not None,
            "batch_processing": True,
            "parallel_inference": True
        },
        "features": {
            "real_time_detection": True,
            "batch_processing": True,
            "url_detection": True,
            "object_recognition": True,
            "detailed_analysis": True,
            "smart_detection_features": True,
            "multi_method_support": True
        }
    }


@app.get("/smart-features")
def get_smart_features():
    """Information about the smart detection features"""
    return {
        "smart_features": {
            "auto_label_refinement": {
                "description": "Uses DBSCAN clustering to merge overlapping boxes and remove low-confidence detections intelligently",
                "benefits": [
                    "Reduces duplicate detections",
                    "Improves detection accuracy",
                    "Self-improving feedback loop"
                ]
            },
            "multi_object_context_awareness": {
                "description": "Scene Graph Model to understand relationships between objects (e.g., 'Toothbrush on sink' or 'Laptop on desk')",
                "benefits": [
                    "Semantic intelligence", 
                    "Understanding of spatial relationships",
                    "Contextual awareness"
                ]
            },
            "fine_grained_classification": {
                "description": "Secondary classifier for sub-type detection (e.g., 'chair → office chair / dining chair / gaming chair')",
                "benefits": [
                    "More detailed object identification",
                    "Sub-type classification",
                    "Enhanced categorization"
                ]
            },
            "self_improving_feedback_loop": {
                "description": "Learns from user corrections to improve future detections",
                "benefits": [
                    "Continuous improvement",
                    "Adapts to user preferences",
                    "Better accuracy over time"
                ]
            }
        },
        "implementation_details": {
            "post_processing": "Lightweight transformer and rule-based filter",
            "scene_graph_model": "Detectron + Graph R-CNN implementation",
            "fine_grained_classifier": "CNN-based secondary classifier",
            "feedback_system": "In-memory storage with confidence adjustments"
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
            draw.text((100, 80), "toilet", fill='black', font=font)
            draw.text((250, 130), "sink", fill='black', font=font)
            draw.text((50, 280), "mirror", fill='black', font=font)
        except:
            pass
        
        result = process_single_image_openai(test_img)
        
        detections = result["detections"] if result["success"] else []
        
        result_image = None
        if result["success"]:
            temp_detections = [
                {"class": det["class"], "confidence": det["confidence"], "bbox": det["bbox"]}
                for det in detections
            ]
            result_image = draw_local_bounding_boxes(test_img.copy(), temp_detections)
        
        unique_id = str(uuid.uuid4())
        result_path = static_dir / f"test_openai_{unique_id}.jpg"
        if result_image:
            result_image.save(result_path, "JPEG", quality=100, optimize=True, subsampling=0)
            result_url = f"/static/test_openai_{unique_id}.jpg"
        else:
            result_url = None
        
        detected_objects = list(set(det["class"] for det in detections))
        
        return {
            "test_status": "success" if result["success"] else "failed",
            "detections_found": len(detections),
            "detected_objects": detected_objects,
            "detections": detections,
            "test_image_url": result_url,
            "detection_source": "openai"
        }
            
    except Exception as e:
        logger.error(f"Test detection failed: {e}")
        return {
            "test_status": "failed",
            "error": str(e)
        }


# Initialize 3D scene mapper
scene_mapper = None

try:
    scene_mapper = AI3DSceneMapper()
    logger.info("3D Scene Mapper initialized successfully")
except Exception as e:
    logger.error(f"Error initializing 3D Scene Mapper: {e}")

@app.post("/3d-scene-map")
async def create_3d_scene_map(
    image: UploadFile = File(...),
    detection_method: str = Form("auto")
):
    """
    Create a 3D reconstruction of the scene from a 2D image.
    First performs object detection, then estimates depth and creates 3D positions.
    """
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    
    try:
        # Read and process image
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Perform object detection first
        if detection_method == "auto":
            # Use OpenAI if available, otherwise use enhanced model
            if openai_client.api_key:
                detection_result = process_single_image_openai(img)
                detection_source = "openai"
            elif enhanced_model:
                detection_result = process_single_image_enhanced(img)
                detection_source = "enhanced_local"
            else:
                raise HTTPException(status_code=500, detail="No detection models available")
        elif detection_method == "openai":
            if not openai_client.api_key:
                raise HTTPException(status_code=500, detail="OpenAI API key not configured")
            detection_result = process_single_image_openai(img)
            detection_source = "openai"
        elif detection_method == "enhanced":
            if not enhanced_model:
                raise HTTPException(status_code=500, detail="Enhanced model not available")
            detection_result = process_single_image_enhanced(img)
            detection_source = "enhanced_local"
        else:
            raise HTTPException(status_code=400, detail="Invalid detection method. Use 'openai', 'enhanced', or 'auto'")
        
        if not detection_result["success"]:
            raise HTTPException(status_code=500, detail=detection_result["error"])
        
        detections = detection_result["detections"]
        
        # Create 3D scene using the mapper
        if scene_mapper is None:
            raise HTTPException(status_code=500, detail="3D Scene Mapper not initialized")
        
        # Generate unique ID for this 3D scene
        unique_id = str(uuid.uuid4())
        scene_output_path = str(static_dir / f"3d_scene_{unique_id}")
        
        # Create 3D scene
        scene_result = scene_mapper.create_3d_scene(
            image=img,
            detections=detections,
            output_path=scene_output_path
        )
        
        # Save scene data to a JSON file for web visualization
        scene_data_path = static_dir / f"3d_scene_data_{unique_id}.json"
        with open(scene_data_path, 'w') as f:
            json.dump(scene_result["scene_data"], f)
        
        # Return the 3D scene result with a URL to the 3D visualization
        response = {
            "filename": image.filename,
            "detections": detections,
            "detection_count": len(detections),
            "detected_objects": list(set(det["class"] for det in detections)),
            "detection_source": detection_source,
            "objects_3d": scene_result["objects_3d"],
            "object_count_3d": scene_result["object_count"],
            "success": scene_result["success"],
            "scene_data_url": f"/static/3d_scene_data_{unique_id}.json",
            "visualization_3d_url": f"/static/3d_visualization.html?dataUrl=/static/3d_scene_data_{unique_id}.json",
            "visualization_2d_path": scene_result.get("visualization_path", None)
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in 3D scene mapping: {e}")
        raise HTTPException(status_code=500, detail=f"3D scene mapping failed: {str(e)}")


@app.get("/3d-visualization")
def get_3d_visualization():
    """
    Serve the 3D visualization page
    """
    try:
        visualization_path = Path("static/3d_visualization.html")
        if visualization_path.exists():
            return FileResponse(visualization_path, media_type="text/html")
        else:
            return {"error": "3D visualization not found"}
    except Exception as e:
        logger.error(f"Error serving 3D visualization: {e}")
        return {"error": str(e)}

@app.post("/chatbot")
async def chatbot_query(
    question: str = Form(...),
):
    """
    AI-related chatbot endpoint. Answers questions only if they are related to the AI field.
    Uses OpenAI's GPT-4o model for accurate and correct responses.
    """
    if not openai_client.api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")

    try:
        # First, check if the question is related to AI
        check_response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a classifier. Determine if the user's question is related to AI (Artificial Intelligence), machine learning, neural networks, computer vision, NLP, or related fields. Respond with only 'YES' or 'NO'."
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            max_tokens=1,
            temperature=0.0
        )
        
        is_ai_related = check_response.choices[0].message.content.strip().upper() == "YES"
        
        if not is_ai_related:
            return {"response": "This chatbot only answers questions related to the AI field."}
        
        # If related, generate accurate response
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in AI. Provide accurate, factual, and correct answers based on established knowledge in the AI field. Be concise and informative. Do not speculate or provide misinformation."
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            max_tokens=500,
            temperature=0.2  # Low temperature for factual accuracy
        )
        
        answer = response.choices[0].message.content.strip()
        
        return {"response": answer}
        
    except Exception as e:
        logger.error(f"Error in chatbot endpoint: {e}")
        raise HTTPException(status_code=500, detail="Chatbot query failed")
    
fake_users_db = {}

# ============================================================
# MODELS
# ============================================================
class User(BaseModel):
    name: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

# ============================================================
# UTILS
# ============================================================
def hash_password(password: str):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# ============================================================
# AUTH ENDPOINTS
# ============================================================

@app.post("/register")
async def register(user: User):
    """Register a new user"""
    if user.email in fake_users_db:
        raise HTTPException(status_code=400, detail="User already exists")

    hashed_pw = hash_password(user.password)
    fake_users_db[user.email] = {
        "name": user.name,
        "email": user.email,
        "password": hashed_pw
    }

    return {"success": True, "message": "Registration successful"}

@app.post("/login")
async def login(user: UserLogin):
    """Authenticate user"""
    stored_user = fake_users_db.get(user.email)
    if not stored_user or not verify_password(user.password, stored_user["password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": stored_user["email"]}, expires_delta=access_token_expires
    )

    return {"success": True, "message": "Login successful", "access_token": access_token}


@app.post("/api/auth/register")
async def register_api(user: User):
    """Register a new user - API endpoint for frontend"""
    if user.email in fake_users_db:
        raise HTTPException(status_code=400, detail="User already exists")

    hashed_pw = hash_password(user.password)
    fake_users_db[user.email] = {
        "name": user.name,
        "email": user.email,
        "password": hashed_pw
    }

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )

    return {"success": True, "user": {"name": user.name, "email": user.email}, "token": access_token}


@app.post("/api/auth/login")
async def login_api(user: UserLogin):
    """Authenticate user - API endpoint for frontend"""
    stored_user = fake_users_db.get(user.email)
    if not stored_user or not verify_password(user.password, stored_user["password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": stored_user["email"]}, expires_delta=access_token_expires
    )

    return {"success": True, "user": {"name": stored_user["name"], "email": stored_user["email"]}, "token": access_token}


@app.get("/api/auth/profile")
async def get_profile(token: str = Depends(oauth2_scheme)):
    """Get user profile - API endpoint for frontend"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None or email not in fake_users_db:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        user = fake_users_db[email]
        return {"user": {"name": user["name"], "email": user["email"]}}
    except JWTError:
        raise HTTPException(status_code=401, detail="Token invalid or expired")

# ============================================================
# PROTECTED ENDPOINT (for testing)
# ============================================================
@app.get("/protected")
async def protected_route(token: str = Depends(oauth2_scheme)):
    """Test protected route"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None or email not in fake_users_db:
            raise HTTPException(status_code=401, detail="Invalid token")
        return {"success": True, "message": f"Welcome, {fake_users_db[email]['name']}!"}
    except JWTError:
        raise HTTPException(status_code=401, detail="Token invalid or expired")

if __name__ == "__main__":
    logger.info("Starting Enhanced Object Detection API Server using OpenAI API")
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )