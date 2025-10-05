# Home Objects Detection Frontend

This is the React frontend for the home objects detection system. It allows users to upload images and see detection results with bounding boxes.

## Getting Started

### Prerequisites

- Node.js (v14 or higher)
- npm or yarn

### Installation

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

### Running the Application

1. Make sure the FastAPI backend is running (see backend README)
2. Start the React development server:
   ```bash
   npm start
   ```

3. Open your browser to `http://localhost:3000`

## Features

- Upload single or multiple images
- Adjust confidence threshold for detections
- View detection results with bounding boxes
- See class names and confidence scores for each detected object

## API Integration

The frontend communicates with the FastAPI backend running on `http://localhost:8000`:
- `/detect` - For single image detection
- `/detect-multiple` - For multiple image detection