# Home Objects Detection System with AI 3D Scene Mapper

This project implements a complete object detection system for home objects using PyTorch, FastAPI, and React. It allows users to upload images and receive processed images with bounding boxes drawn around detected home objects. NEW: Advanced AI 3D Scene Mapping feature that converts 2D images to interactive 3D environments!

## Features

- **Object Detection**: Detects 25+ different home objects including toilets, sinks, mirrors, bathtubs, towels, etc.
- **Web Interface**: Clean, responsive React frontend for easy image upload and result viewing
- **API Backend**: FastAPI server for handling image processing requests
- **Real-time Results**: Instant visualization of detected objects with bounding boxes
- **Configurable**: Adjustable confidence threshold for detection sensitivity
- **AI 3D Scene Mapper**: NEW! Advanced feature that converts 2D images to interactive 3D environments
- **Depth Estimation**: Uses state-of-the-art models (DPT, MiDaS) to estimate depth from 2D images
- **3D Reconstruction**: Places detected objects in 3D space with realistic spatial relationships
- **Interactive Visualization**: Three.js-based 3D viewer for exploring the reconstructed scene

## System Architecture

```
[React Frontend] <---> [FastAPI Backend] <---> [PyTorch Model]
```

## Components

### 1. Object Detection Model (`object_detection_model.py`)
- YOLO-style architecture for object detection
- Custom bounding box drawing functionality
- Synthetic dataset generation for training

### 2. FastAPI Backend (`api_server.py`)
- Image upload and processing endpoints
- Object detection API
- Static file serving for results

### 3. React Frontend (`frontend/`)
- User-friendly interface for image uploads
- Real-time display of detection results
- Confidence threshold adjustment

## Prerequisites

- Python 3.7+
- Node.js 14+
- PyTorch compatible with your system (CPU or CUDA)

## Installation and Setup

### Backend Setup

1. Clone or download this repository
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Train the model (if not already trained):
   ```bash
   python object_detection_model.py
   ```
   This will create the model file `home_objects_detection_model.pth`

4. Start the API server:
   ```bash
   python api_server.py
   ```
   The API will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

4. Open your browser to `http://localhost:3000`

## Usage

1. Ensure the FastAPI backend is running on port 8000
2. Open the React frontend in your browser
3. Upload one or more images using the file selector
4. Adjust the confidence threshold as needed
5. Click "Detect Objects" to process the images
6. View the results with bounding boxes and detection information

## API Endpoints

- `GET /` - API information
- `POST /detect` - Single image detection
- `POST /detect-multiple` - Multiple image detection
- `GET /health` - Health check
- `GET /classes` - List of detectable classes
- `GET /docs` - Interactive API documentation
- `POST /3d-scene-map` - Create 3D reconstruction from 2D image (NEW!)
- `GET /3d-visualization` - Serve 3D visualization interface (NEW!)

## Model Classes

The system can detect the following home objects:
- Toilet
- Sink
- Mirror
- Bathtub
- Showerhead
- Towel
- Toothbrush
- Toothpaste
- Soap Bar
- Shampoo Bottle
- Conditioner Bottle
- Handwash Bottle
- Toilet Paper Roll
- Towel Rack
- Bath Mat
- Hair Dryer
- Razor
- Lotion Bottle
- Trash Bin
- Shower Curtain
- Comb
- Cleaning Brush
- Bucket
- Mug
- Bathroom Shelf

## Development

### Adding New Object Classes

1. Update the `HOME_OBJECTS` list in `object_detection_model.py`
2. Retrain the model with updated class information
3. Update the frontend to handle new classes if needed

### Improving Model Accuracy

1. Add more training data with bounding box annotations
2. Adjust the model architecture in `object_detection_model.py`
3. Tune hyperparameters in the training function

## Project Structure

```
├── object_detection_model.py     # Core detection model and training
├── api_server.py                # FastAPI backend
├── api_server_README.md         # API server documentation
├── requirements.txt             # Python dependencies
├── frontend/                    # React frontend
│   ├── public/
│   ├── src/
│   ├── package.json
│   └── README.md
├── static/                      # Directory for detection results (created automatically)
└── README.md                    # This file
```

## Troubleshooting

- If you get CUDA errors, ensure PyTorch with the appropriate CUDA version is installed
- Make sure the API server is running before starting the frontend
- If detection results don't load, check that the static directory is accessible

## License

This project is available for educational and research purposes.