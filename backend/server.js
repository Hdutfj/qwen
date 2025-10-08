const express = require('express');
const cors = require('cors');
const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json());
app.use('/uploads', express.static('uploads'));

// JWT Secret
const JWT_SECRET = process.env.JWT_SECRET || 'your_jwt_secret_key';

// In-memory storage (would use a database in production)
let users = [
  {
    id: 1,
    name: 'John Doe',
    email: 'john@example.com',
    password: bcrypt.hashSync('password123', 8)
  }
];
let uploads = [];
let detections = [];

// Multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    if (!fs.existsSync('uploads')) {
      fs.mkdirSync('uploads');
    }
    cb(null, 'uploads/');
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + path.extname(file.originalname));
  }
});
const upload = multer({ storage });

// Authentication middleware
const authenticateToken = (req, res, next) => {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];

  if (!token) {
    return res.status(401).json({ message: 'Access token required' });
  }

  jwt.verify(token, JWT_SECRET, (err, user) => {
    if (err) {
      return res.status(403).json({ message: 'Invalid or expired token' });
    }
    req.user = user;
    next();
  });
};

// Routes

// Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'healthy' });
});

// Registration
app.post('/api/auth/register', async (req, res) => {
  try {
    const { name, email, password } = req.body;

    // Check if user already exists
    const existingUser = users.find(u => u.email === email);
    if (existingUser) {
      return res.status(400).json({ message: 'User already exists' });
    }

    // Hash password
    const hashedPassword = await bcrypt.hash(password, 8);
    
    // Create new user
    const newUser = {
      id: users.length + 1,
      name,
      email,
      password: hashedPassword
    };
    users.push(newUser);

    // Generate JWT token
    const token = jwt.sign({ id: newUser.id, email: newUser.email }, JWT_SECRET, { expiresIn: '24h' });

    res.status(201).json({ 
      message: 'User registered successfully', 
      token,
      user: { id: newUser.id, name: newUser.name, email: newUser.email }
    });
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: error.message });
  }
});

// Login
app.post('/api/auth/login', async (req, res) => {
  try {
    const { email, password } = req.body;

    // Find user
    const user = users.find(u => u.email === email);
    if (!user) {
      return res.status(400).json({ message: 'Invalid credentials' });
    }

    // Check password
    const validPassword = await bcrypt.compare(password, user.password);
    if (!validPassword) {
      return res.status(400).json({ message: 'Invalid credentials' });
    }

    // Generate JWT token
    const token = jwt.sign({ id: user.id, email: user.email }, JWT_SECRET, { expiresIn: '24h' });

    res.json({ 
      message: 'Login successful', 
      token,
      user: { id: user.id, name: user.name, email: user.email }
    });
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: error.message });
  }
});

// Get user profile
app.get('/api/auth/profile', authenticateToken, (req, res) => {
  const user = users.find(u => u.id === req.user.id);
  if (!user) {
    return res.status(404).json({ message: 'User not found' });
  }
  res.json({ user: { id: user.id, name: user.name, email: user.email } });
});

// Object detection endpoint
app.post('/api/detect-batch', authenticateToken, upload.array('images', 10), async (req, res) => {
  try {
    if (!req.files || req.files.length === 0) {
      return res.status(400).json({ message: 'No images provided' });
    }

    // Simulate detection process
    // In a real application, this would call your ML model
    const results = req.files.map(file => {
      const detectionId = Date.now() + Math.random();
      const detection = {
        id: detectionId,
        userId: req.user.id,
        originalImage: file.filename,
        detectedObjects: [
          { label: 'person', confidence: Math.random() * 0.9 + 0.1, bbox: [50, 50, 200, 300] },
          { label: 'car', confidence: Math.random() * 0.9 + 0.1, bbox: [250, 100, 350, 200] }
        ],
        timestamp: new Date().toISOString()
      };
      
      detections.push(detection);
      
      return {
        id: detectionId,
        originalImage: `/uploads/${file.filename}`,
        detectedObjects: detection.detectedObjects,
        processedImage: `/uploads/${file.filename}` // In real app, this would be the processed image
      };
    });

    res.json({ 
      message: 'Detection completed successfully', 
      results 
    });
  } catch (error) {
    res.status(500).json({ message: 'Detection failed', error: error.message });
  }
});

// 3D Scene mapping endpoint
app.post('/api/3d-scene-map', authenticateToken, upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ message: 'No image provided' });
    }

    // Simulate 3D scene generation
    // In a real application, this would call your 3D mapping model
    const sceneId = Date.now() + Math.random();
    const scene = {
      id: sceneId,
      userId: req.user.id,
      originalImage: req.file.filename,
      sceneFile: `scene_${sceneId}.obj`,
      timestamp: new Date().toISOString()
    };

    res.json({ 
      message: '3D scene generated successfully', 
      scene: {
        id: sceneId,
        originalImage: `/uploads/${req.file.filename}`,
        sceneFile: `/uploads/${scene.sceneFile}`, // Placeholder for actual 3D file
        status: 'completed'
      }
    });
  } catch (error) {
    res.status(500).json({ message: '3D scene generation failed', error: error.message });
  }
});

// User dashboard data
app.get('/api/dashboard', authenticateToken, (req, res) => {
  const userDetections = detections.filter(d => d.userId === req.user.id);
  const totalDetections = userDetections.length;
  
  // Count detected objects by label
  const labelCounts = {};
  userDetections.forEach(detection => {
    detection.detectedObjects.forEach(obj => {
      labelCounts[obj.label] = (labelCounts[obj.label] || 0) + 1;
    });
  });

  res.json({
    totalDetections,
    detectionHistory: userDetections.slice(0, 10), // Last 10 detections
    topLabels: Object.entries(labelCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([label, count]) => ({ label, count }))
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});