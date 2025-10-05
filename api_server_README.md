# Home Objects Detection API

This is the FastAPI backend for the home objects detection system. It handles image uploads, runs object detection, and returns processed images with bounding boxes.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- PyTorch and related packages (see requirements.txt)

### Installation

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   # Or if you don't have the file, install manually:
   pip install fastapi uvicorn torch torchvision pillow python-multipart
   ```

2. Make sure you have the trained model file `home_objects_detection_model.pth` in the root directory.
   If you don't have it, run the training script first:
   ```bash
   python object_detection_model.py
   ```

### Running the API Server

1. Start the server:
   ```bash
   python api_server.py
   ```
   
   Or with uvicorn directly:
   ```bash
   uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
   ```

2. The API will be available at `http://localhost:8000`

## API Endpoints

### `GET /`
- Returns API information

### `POST /detect`
- Upload a single image for object detection
- Form data: `image` (file) and `confidence_threshold` (float)
- Returns detection results and URL to processed image

### `POST /detect-multiple`
- Upload multiple images for object detection
- Form data: `images` (multiple files) and `confidence_threshold` (float)
- Returns detection results for each image

### `GET /health`
- Health check endpoint

### `GET /classes`
- Get the list of detectable classes

## API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation and testing.