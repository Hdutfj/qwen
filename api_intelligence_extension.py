"""
Extension module to integrate automation and intelligence features into the existing API
"""

from fastapi import HTTPException, Form
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional
import numpy as np
from PIL import Image
import io
import tempfile
import os
import uuid

from automation_intelligence import AutomationIntelligencePipeline


def extend_api_with_intelligence_features(app, openai_client=None):
    """
    Extend existing API with intelligence features
    """
    from fastapi import UploadFile, File
    from fastapi import Form
    
    intelligence_pipeline = AutomationIntelligencePipeline(openai_client)
    
    @app.post("/generate-caption")
    async def generate_scene_caption(image: UploadFile = File(...)):
        """Generate caption for a scene"""
        try:
            # Read and convert image
            contents = await image.read()
            pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
            
            caption = intelligence_pipeline.scene_describer.generate_caption(pil_image)
            return {"caption": caption, "filename": image.filename}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate caption: {str(e)}")
    
    @app.post("/count-objects")
    async def count_objects_in_image(detections: List[Dict]):
        """Count objects in detections"""
        try:
            counts = intelligence_pipeline.object_counter.count_objects(detections)
            return {"object_counts": counts}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to count objects: {str(e)}")
    
    @app.post("/detect-changes")
    async def detect_changes_between_images(
        before_image: UploadFile = File(..., description="Before renovation image"),
        after_image: UploadFile = File(..., description="After renovation image")
    ):
        """Detect changes between two images"""
        try:
            # Read and convert images
            before_contents = await before_image.read()
            after_contents = await after_image.read()
            
            before_img = Image.open(io.BytesIO(before_contents)).convert("RGB")
            after_img = Image.open(io.BytesIO(after_contents)).convert("RGB")
            
            changes = intelligence_pipeline.compare_images(before_img, after_img)
            return {
                "change_analysis": changes,
                "before_filename": before_image.filename,
                "after_filename": after_image.filename
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to detect changes: {str(e)}")
    
    @app.post("/track-objects-frame")
    async def track_objects_in_frame(
        frame: UploadFile = File(...),
        detections: List[Dict] = Form(...)
    ):
        """Track objects in a single video frame"""
        try:
            # Read and convert frame
            contents = await frame.read()
            frame_np = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))
            
            result = intelligence_pipeline.process_video_frame(frame_np, detections)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to track objects: {str(e)}")
    
    @app.post("/estimate-real-area")
    async def estimate_real_world_area(
        detections: List[Dict],
        reference_object: Dict = Form(...)
    ):
        """Estimate real-world area using reference object"""
        try:
            real_areas = intelligence_pipeline.object_counter.estimate_area_in_real_units(
                detections, reference_object
            )
            return {"real_world_estimates": real_areas}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to estimate real-world area: {str(e)}")
    
    @app.get("/intelligence-features-info")
    async def get_intelligence_features_info():
        """Get information about available intelligence features"""
        return {
            "features": [
                {
                    "name": "Auto-Captioning",
                    "description": "Generate textual descriptions of scenes using Vision-Language models",
                    "endpoint": "/generate-caption",
                    "method": "POST",
                    "parameters": ["image"]
                },
                {
                    "name": "Object Counting & Area Estimation",
                    "description": "Count instances of objects and estimate occupied area",
                    "endpoint": "/count-objects",
                    "method": "POST",
                    "parameters": ["detections"]
                },
                {
                    "name": "Change Detection",
                    "description": "Compare two images to highlight new, missing, or moved objects",
                    "endpoint": "/detect-changes",
                    "method": "POST",
                    "parameters": ["before_image", "after_image"]
                },
                {
                    "name": "Object Tracking",
                    "description": "Track objects across frames using DeepSORT-inspired algorithm",
                    "endpoint": "/track-objects-frame",
                    "method": "POST",
                    "parameters": ["frame", "detections"]
                },
                {
                    "name": "Real-World Area Estimation",
                    "description": "Estimate real-world measurements using reference objects",
                    "endpoint": "/estimate-real-area",
                    "method": "POST",
                    "parameters": ["detections", "reference_object"]
                }
            ],
            "description": "Advanced automation and intelligence features for home object detection system"
        }
    
    return app