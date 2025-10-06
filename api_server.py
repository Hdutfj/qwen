"""
Enhanced FastAPI Backend for Object Detection
Using OpenAI API for object detection
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import requests
from openai import OpenAI
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
from base64 import b64encode

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

if not openai_client.api_key:
    logger.error("OPENAI_API_KEY not found in environment variables")
    raise ValueError("OPENAI_API_KEY is required")

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
        img.save(buffer, format="JPEG", quality=85)
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
                            "text": "Analyze this image and identify all objects present with their approximate locations. Return a JSON response with the following structure: {\"objects\": [{\"name\": \"object_name\", \"confidence\": 0.95, \"position\": \"position_in_image\", \"size_ratio\": 0.2}]}. Position can be 'top-left', 'top-center', 'top-right', 'middle-left', 'center', 'middle-right', 'bottom-left', 'bottom-center', 'bottom-right'. Size_ratio represents approximate space the object occupies (0.0 to 1.0). Common objects to look for include: chair, table, sofa, bed, desk, lamp, mirror, window, door, curtain, bookshelf, TV, refrigerator, microwave, stove, sink, toilet, bathtub, shower, towel, pillow, painting, plant, cup, bottle, and any other common household items. Only respond with the JSON object and nothing else."
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
        
        import json
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
        
        # Format as detection results (without bounding boxes since OpenAI doesn't provide them)
        detection_results = []
        for obj in detected_objects:
            if isinstance(obj, dict):
                # If obj is a dictionary with name, confidence, position, and size_ratio
                class_name = obj.get("name", "")
                confidence = obj.get("confidence", 0.85)
                position = obj.get("position", "unknown")
                size_ratio = obj.get("size_ratio", 0.1)
            else:
                # If obj is just a string
                class_name = str(obj)
                confidence = 0.85  # Default confidence
                position = "unknown"
                size_ratio = 0.1  # Default size ratio
            
            if class_name and class_name.strip():  # Make sure the object name isn't empty
                detection_results.append({
                    "class": class_name,
                    "confidence": confidence,
                    "position": position,
                    "size_ratio": size_ratio,
                    "bbox": None,  # OpenAI API doesn't provide bounding boxes in this usage
                    "center": None,
                    "size": None,
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

def draw_local_bounding_boxes(image, detections):
    """Draw bounding boxes and labels on image based on detections - using position info from OpenAI when available"""
    if not detections:
        return image
    
    # Work with a copy of the image
    result_image = image.copy()
    draw = ImageDraw.Draw(result_image)
    
    # Define position mapping to coordinates
    width, height = image.size
    position_coords = {
        "top-left": (width * 0.2, height * 0.2),
        "top-center": (width * 0.5, height * 0.2),
        "top-right": (width * 0.8, height * 0.2),
        "middle-left": (width * 0.2, height * 0.5),
        "center": (width * 0.5, height * 0.5),
        "middle-right": (width * 0.8, height * 0.5),
        "bottom-left": (width * 0.2, height * 0.8),
        "bottom-center": (width * 0.5, height * 0.8),
        "bottom-right": (width * 0.8, height * 0.8)
    }
    
    # Define a color palette for different objects
    colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
        '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
        '#F8C471', '#82E0AA', '#F1948A', '#85C1E9', '#D7BDE2'
    ]
    
    for i, det in enumerate(detections):
        # Get position info from detection if available
        position_str = det.get("position", "unknown")
        size_ratio = det.get("size_ratio", 0.1)
        
        if position_str in position_coords:
            x, y = position_coords[position_str]
        else:
            # Fallback to grid if position not provided
            grid_rows = int(len(detections) ** 0.5) + 1
            grid_cols = grid_rows
            row = i // grid_cols
            col = i % grid_cols
            x = (col + 1) * width // (grid_cols + 1)
            y = (row + 1) * height // (grid_rows + 1)
        
        # Get color for this object
        color = colors[i % len(colors)]
        class_name = det.get("class", "Unknown")
        
        # Calculate box size based on size_ratio
        box_size = int(60 + (size_ratio * 100))  # Base size of 60 + adjustment for size ratio
        x1, y1 = x - box_size//2, y - box_size//2
        x2, y2 = x + box_size//2, y + box_size//2
        
        # Draw rectangle
        for thickness in range(3):
            draw.rectangle([x1-thickness, y1-thickness, x2+thickness, y2+thickness], 
                           outline=color, width=1)
        
        # Draw label with background
        try:
            font = ImageFont.load_default()
        except:
            font = None

        if font:
            bbox = draw.textbbox((0, 0), class_name, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            text_width, text_height = len(class_name) * 6, 11

        label_y = max(0, y1 - text_height - 5)
        draw.rectangle([x1, label_y, x1 + text_width + 10, label_y + text_height + 5], 
                       fill=color)
        draw.text((x1 + 5, label_y + 2), class_name, fill='white', font=font)
    
    return result_image

@app.get("/")
def read_root():
    return {
        "message": "Enhanced Object Detection API",
        "version": "2.0.0",
        "features": [
            "Bathroom object detection",
            "Fast inference",
            "High accuracy detection",
            "Single and batch processing support"
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
    request: Request,
    image: Optional[UploadFile] = File(None),
    # Remove confidence_threshold since OpenAI doesn't use it
    draw_boxes: bool = Form(True)
):
    """
    Detect objects in an uploaded image using OpenAI API.
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

        # Run model prediction with the PIL Image using OpenAI API
        result = process_single_image_openai(img)
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])

        detections = result["detections"]

        # --- Draw bounding boxes ---
        if draw_boxes:
            result_image = draw_local_bounding_boxes(img.copy(), detections)
        else:
            result_image = img

        # Save final image
        result_url = None
        if result_image:
            static_dir.mkdir(exist_ok=True)
            unique_id = str(uuid.uuid4())
            result_path = static_dir / f"result_{unique_id}.jpg"
            result_image.save(result_path, "JPEG", quality=95)
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
            "image_size": result["image_size"]
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
    draw_boxes: bool = Form(True)
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
                
            result = process_single_image_openai(img)
            
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
                result_image = draw_local_bounding_boxes(img.copy(), temp_detections)
            
            result_url = None
            if result_image:
                unique_id = str(uuid.uuid4())
                result_path = static_dir / f"batch_openai_{unique_id}.jpg"
                result_image.save(result_path, "JPEG", quality=95)
                result_url = f"/static/batch_openai_{unique_id}.jpg"
            
            detected_objects = list(set(det["class"] for det in detections))
            
            result_item = {
                "filename": image.filename,
                "detections": detections,
                "detected_objects": detected_objects,
                "detection_count": len(detections),
                "result_image_url": result_url,
                "detection_source": "openai",
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
            result_image.save(result_path, "JPEG", quality=95)
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
        # Test OpenAI API connection
        start_time = time.time()
        
        # Create a dummy 1x1 pixel image for testing
        dummy_img = Image.new('RGB', (10, 10), color='white')
        result = process_single_image_openai(dummy_img)
        
        inference_time = time.time() - start_time
        
        return {
            "status": "healthy",
            "openai_api_key_set": bool(openai_client.api_key),
            "inference_test_ms": round(inference_time * 1000, 2),
            "api_version": "2.0.0"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "openai_api_key_set": bool(openai_client.api_key)
        }

@app.get("/stats")
def get_stats():
    return {
        "model_info": {
            "service": "OpenAI GPT-4 Vision",
            "api_provider": "OpenAI",
            "detection_method": "Cloud-based image analysis"
        },
        "performance": {
            "api_based": True,
            "network_required": True,
            "batch_processing": True,
            "parallel_inference": True
        },
        "features": {
            "real_time_detection": True,
            "batch_processing": True,
            "url_detection": True,
            "object_recognition": True,
            "detailed_analysis": True
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
            result_image.save(result_path, "JPEG", quality=95)
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

if __name__ == "__main__":
    logger.info("Starting Enhanced Object Detection API Server using OpenAI API")
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )