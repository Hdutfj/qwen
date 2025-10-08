import React, { useState, useRef } from "react";
import {
  Container,
  Row,
  Col,
  Card,
  Button,
  Form,
  ProgressBar,
  Image,
  Spinner,
  Nav,
} from "react-bootstrap";
import { motion } from "framer-motion";
import { CloudUpload, XCircle, EyeFill } from "react-bootstrap-icons";
import axios from "axios";
import { useNotifications } from "../contexts/NotificationContext";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";

const DetectionPage = () => {
  const [images, setImages] = useState([]);
  const [results, setResults] = useState([]);
  const [isDetecting, setIsDetecting] = useState(false);
  const [progress, setProgress] = useState(0);
  const [showConfidence, setShowConfidence] = useState(true);
  const [previewUrls, setPreviewUrls] = useState([]);
  const [detectionMethod, setDetectionMethod] = useState("auto");
  const fileInputRef = useRef(null);
  const { addNotification } = useNotifications();
  const navigate = useNavigate();
  const { currentUser, isAuthenticated, loading, logout } = useAuth();

  // Redirect if not authenticated
  React.useEffect(() => {
    if (!loading && !isAuthenticated) {
      navigate("/login");
    }
  }, [loading, isAuthenticated, navigate]);

  // File selection
  const handleFileChange = (e) => {
    const files = Array.from(e.target.files);
    if (files.length > 0) {
      const previews = files.map((file) => URL.createObjectURL(file));
      setPreviewUrls((prev) => [...prev, ...previews]);
      const formatted = files.map((file) => ({ file, name: file.name }));
      setImages((prev) => [...prev, ...formatted]);
    }
  };

  const removeImage = (index) => {
    setImages((prev) => prev.filter((_, i) => i !== index));
    setPreviewUrls((prev) => prev.filter((_, i) => i !== index));
  };

  const handleDetect = async () => {
    if (images.length === 0) {
      addNotification("error", "Please upload at least one image.");
      return;
    }

    setIsDetecting(true);
    setProgress(0);
    setResults([]);

    const formData = new FormData();
    images.forEach((img) => formData.append("images", img.file));
    formData.append("draw_boxes", "true");
    formData.append("detection_method", detectionMethod);

    try {
      const res = await axios.post("http://localhost:8000/detect-batch", formData, {
        headers: { "Content-Type": "multipart/form-data" },
        onUploadProgress: (e) => {
          const percent = Math.round((e.loaded * 100) / e.total);
          setProgress(percent);
        },
      });

      if (res.data && res.data.results) {
        setResults(res.data.results);
        addNotification("success", "✅ Object detection completed successfully!");
      } else {
        addNotification("error", "❌ No results returned from the backend.");
      }
    } catch (err) {
      console.error("Detection Error:", err);
      addNotification("error", "❌ Error during detection. Check console.");
    } finally {
      setIsDetecting(false);
    }
  };

  // Drag & drop
  const handleDragOver = (e) => e.preventDefault();
  const handleDrop = (e) => {
    e.preventDefault();
    const files = Array.from(e.dataTransfer.files).filter((f) =>
      f.type.startsWith("image/")
    );
    if (files.length > 0) {
      const previews = files.map((file) => URL.createObjectURL(file));
      setPreviewUrls((prev) => [...prev, ...previews]);
      const formatted = files.map((file) => ({ file, name: file.name }));
      setImages((prev) => [...prev, ...formatted]);
    }
  };

  if (loading) return <p>Loading...</p>;

  return (
    <div
      style={{
        display: "flex",
        height: "100vh",
        overflow: "hidden",
        backgroundColor: "#f9fafb",
      }}
    >
      {/* ===== Sidebar (20%) ===== */}
      <div
        style={{
          width: "20%",
          minWidth: "250px",
          backgroundColor: "#1e293b",
          color: "white",
          display: "flex",
          flexDirection: "column",
          padding: "20px",
        }}
      >
        <h3 className="mb-4">Menu</h3>
        <Nav className="flex-column">
          <Nav.Link style={{ color: "white" }} onClick={() => navigate("/dashboard")}>
            Dashboard
          </Nav.Link>
          <Nav.Link style={{ color: "white" }} onClick={logout}>
            Logout
          </Nav.Link>
        </Nav>
      </div>

      {/* ===== Main Content (80%) ===== */}
      <div
        style={{
          flex: 1,
          overflowY: "auto",
          padding: "40px",
        }}
      >
        <Container fluid>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            {/* Header */}
            <Row className="mb-4 align-items-center">
              <Col>
                <h1 className="fw-bold">Object Detection Dashboard</h1>
                <p className="text-muted">
                  Upload multiple images and detect objects using AI.
                </p>
              </Col>
              <Col className="text-end">
                <div
                  style={{
                    backgroundColor: "#2563eb",
                    color: "white",
                    borderRadius: "50%",
                    width: "45px",
                    height: "45px",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: "22px",
                    fontWeight: "bold",
                  }}
                >
                  👤
                </div>
              </Col>
            </Row>

            {/* Upload Section */}
            <Row className="mb-4">
              <Col>
                <Card>
                  <Card.Body>
                    <div
                      className="text-center p-5"
                      onDragOver={handleDragOver}
                      onDrop={handleDrop}
                      style={{
                        border: "2px dashed #6b7280",
                        borderRadius: "10px",
                        background: "#f3f4f6",
                        cursor: "pointer",
                      }}
                    >
                      <CloudUpload size={60} className="text-primary mb-3" />
                      <h5>Drag & Drop your images</h5>
                      <p className="text-muted mb-3">or</p>
                      <Button
                        variant="outline-primary"
                        onClick={() => fileInputRef.current.click()}
                      >
                        Browse Files
                      </Button>
                      <Form.Control
                        type="file"
                        ref={fileInputRef}
                        onChange={handleFileChange}
                        multiple
                        accept="image/*"
                        className="d-none"
                      />
                    </div>
                  </Card.Body>
                </Card>
              </Col>
            </Row>

            {/* Selected Images Preview */}
            {images.length > 0 && (
              <Row className="mb-4">
                <Col>
                  <h5>Selected Images ({images.length})</h5>
                  <div className="d-flex flex-wrap gap-3">
                    {images.map((img, index) => (
                      <div key={index} className="position-relative">
                        <Image
                          src={previewUrls[index]}
                          alt="preview"
                          thumbnail
                          style={{
                            width: "120px",
                            height: "120px",
                            objectFit: "cover",
                          }}
                        />
                        <Button
                          variant="danger"
                          size="sm"
                          className="position-absolute top-0 end-0"
                          onClick={() => removeImage(index)}
                        >
                          <XCircle />
                        </Button>
                        <div className="text-center mt-1">
                          <small className="text-muted">{img.name}</small>
                        </div>
                      </div>
                    ))}
                  </div>
                </Col>
              </Row>
            )}

            {/* Detect Button + Switch */}
            <Row className="mb-4">
              <Col md={6}>
                <Button
                  variant="primary"
                  size="lg"
                  className="w-100"
                  onClick={handleDetect}
                  disabled={isDetecting || images.length === 0}
                >
                  {isDetecting ? (
                    <>
                      <Spinner animation="border" size="sm" className="me-2" />
                      Processing... {progress}%
                    </>
                  ) : (
                    <>
                      <EyeFill className="me-2" />
                      Detect Objects
                    </>
                  )}
                </Button>
              </Col>
              <Col md={6} className="d-flex align-items-center">
                <Form.Check
                  type="switch"
                  id="show-confidence"
                  label="Show Confidence Score"
                  checked={showConfidence}
                  onChange={() => setShowConfidence(!showConfidence)}
                />
              </Col>
            </Row>

            {/* Progress Bar */}
            {isDetecting && (
              <Row className="mb-4">
                <Col>
                  <ProgressBar
                    now={progress}
                    label={`${progress}%`}
                    striped
                    animated
                    variant="primary"
                  />
                </Col>
              </Row>
            )}

            {/* Results */}
            {results.length > 0 && (
              <Row>
                <Col>
                  <h5 className="mb-3">Detection Results</h5>
                  {results.map((res, i) => (
                    <Card key={i} className="mb-3 shadow-sm">
                      <Card.Body>
                        <h6>{res.filename}</h6>
                        {res.result_image_url && (
                          <Image
                            src={`http://localhost:8000${res.result_image_url}`}
                            fluid
                            className="rounded mb-3"
                            style={{ maxHeight: "400px", objectFit: "contain" }}
                          />
                        )}
                        <p className="text-muted mb-1">
                          Source: {res.detection_source}
                        </p>
                        <p className="text-success mb-1">
                          {res.detection_count} objects detected
                        </p>
                        {showConfidence && (
                          <ul className="list-group list-group-flush mt-2">
                            {res.detections.map((det, j) => (
                              <li
                                key={j}
                                className="list-group-item d-flex justify-content-between align-items-center"
                              >
                                <span>{det.class}</span>
                                <span className="badge bg-success">
                                  {(det.confidence * 100).toFixed(1)}%
                                </span>
                              </li>
                            ))}
                          </ul>
                        )}
                      </Card.Body>
                    </Card>
                  ))}
                </Col>
              </Row>
            )}
          </motion.div>
        </Container>
      </div>
    </div>
  );
};

export default DetectionPage;
