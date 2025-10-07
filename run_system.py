"""
System Integration Script for Home Objects Detection System
This script ensures all components are properly set up and integrated.
"""
import os
import sys
from pathlib import Path
import subprocess
import time
import threading
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_model_exists():
    """Check if the trained model exists"""
    model_path = Path("home_objects_detection_model.pth")
    enhanced_model_path = Path("enhanced_detection_model.pth")
    return model_path.exists() or enhanced_model_path.exists()

def train_model_if_needed():
    """Train the models if they don't exist"""
    model_path = Path("home_objects_detection_model.pth")
    enhanced_model_path = Path("enhanced_detection_model.pth")
    
    # Check if any model exists
    if not model_path.exists() and not enhanced_model_path.exists():
        logger.info("No models found. Starting training for basic model...")
        try:
            from object_detection_model import main as train_main
            train_main()
            if model_path.exists():
                logger.info("Basic model training completed successfully!")
            else:
                logger.error("Basic model training may have failed. Model file not found.")
                return False
        except Exception as e:
            logger.error(f"Error during basic model training: {e}")
            return False
    
    # Check if enhanced model exists
    if not enhanced_model_path.exists():
        logger.info("Enhanced model not found. Starting training for enhanced model...")
        try:
            from enhanced_detection_model import main as enhanced_train_main
            enhanced_train_main()
            if enhanced_model_path.exists():
                logger.info("Enhanced model training completed successfully!")
            else:
                logger.error("Enhanced model training may have failed. Model file not found.")
                return False
        except Exception as e:
            logger.error(f"Error during enhanced model training: {e}")
            return False
    else:
        logger.info("Enhanced model already exists. Skipping training.")
    
    return True

def start_api_server():
    """Start the FastAPI server in a separate thread"""
    import uvicorn
    from api_server import app
    
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

def check_api_health():
    """Check if the API server is running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    logger.info("Home Objects Detection System - Integration Script")
    logger.info("=" * 60)
    
    # Step 1: Ensure model exists
    logger.info("Step 1: Checking model...")
    if not train_model_if_needed():
        logger.error("Failed to ensure model exists. Exiting.")
        return
    
    # Step 2: Start API server in a separate thread
    logger.info("Step 2: Starting API server...")
    api_thread = threading.Thread(target=start_api_server, daemon=True)
    api_thread.start()
    
    # Wait for API server to start
    logger.info("Waiting for API server to start...")
    max_wait_time = 30  # seconds
    wait_time = 0
    while wait_time < max_wait_time:
        if check_api_health():
            logger.info("API server is ready!")
            break
        time.sleep(1)
        wait_time += 1
    else:
        logger.warning("API server may not have started properly after waiting.")
    
    # Step 3: Provide instructions for frontend
    logger.info("Step 3: Frontend instructions...")
    logger.info("The API server is running on http://localhost:8000")
    logger.info("\nTo use the React frontend:")
    logger.info("1. Open a new terminal")
    logger.info("2. Navigate to the 'frontend' directory: cd frontend")
    logger.info("3. Install dependencies: npm install")
    logger.info("4. Start the frontend: npm start")
    logger.info("5. Open your browser to http://localhost:3000")
    
    logger.info("\nAPI endpoints:")
    logger.info("- GET  http://localhost:8000/ - API info")
    logger.info("- POST http://localhost:8000/detect - Single image detection")
    logger.info("- POST http://localhost:8000/detect-multiple - Multiple image detection")
    logger.info("- GET  http://localhost:8000/health - Health check")
    logger.info("- GET  http://localhost:8000/classes - List of classes")
    logger.info("- GET  http://localhost:8000/docs - Interactive API docs")
    
    logger.info("\nThe system is now running. Press Ctrl+C to stop.")
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
        logger.info("Please manually stop the frontend if it's running.")

if __name__ == "__main__":
    main()