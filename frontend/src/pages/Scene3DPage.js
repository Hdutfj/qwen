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
  Nav,
} from "react-bootstrap";
import { motion } from "framer-motion";
import { GearWideConnected, XCircle, Download } from "react-bootstrap-icons";
import { useNotifications } from "../contexts/NotificationContext";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";

const Scene3DPage = () => {
  const [image, setImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [sceneResult, setSceneResult] = useState(null);
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

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith("image/")) {
      setPreviewUrl(URL.createObjectURL(file));
      setImage({ file, name: file.name, size: file.size });
    }
  };

  const removeImage = () => {
    setImage(null);
    setPreviewUrl("");
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const simulate3DGeneration = async () => {
    if (!image) {
      addNotification("error", "Please upload an image first");
      return;
    }

    setIsGenerating(true);
    setProgress(0);
    setSceneResult(null);

    // Simulate progress
    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          return 100;
        }
        return prev + 10;
      });
    }, 300);

    try {
      // Simulate API call delay
      await new Promise((resolve) => setTimeout(resolve, 3000));
      const simulatedResult = {
        id: Date.now(),
        originalImage: previewUrl,
        sceneFile: "scene.obj",
        status: "completed",
      };

      setSceneResult(simulatedResult);
      addNotification("success", "3D scene generated successfully!");
    } catch (error) {
      addNotification("error", "3D scene generation failed. Please try again.");
    } finally {
      clearInterval(interval);
      setIsGenerating(false);
    }
  };

  const handleDragOver = (e) => e.preventDefault();
  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      setPreviewUrl(URL.createObjectURL(file));
      setImage({ file, name: file.name, size: file.size });
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
            <Row className="mb-4">
              <Col>
                <h1 className="fw-bold">3D Scene Mapping</h1>
                <p className="text-muted">Transform your images into interactive 3D scenes</p>
              </Col>
            </Row>

            {/* Upload Section */}
            <Row className="mb-4">
              <Col>
                <Card>
                  <Card.Body>
                    <div
                      className="upload-area text-center p-5"
                      onDragOver={handleDragOver}
                      onDrop={handleDrop}
                      style={{
                        border: "2px dashed #6b7280",
                        borderRadius: "10px",
                        background: "#f3f4f6",
                        cursor: "pointer",
                      }}
                    >
                      <GearWideConnected size={60} className="text-primary mb-3" />
                      <h5>Upload an image to generate a 3D scene</h5>
                      <p className="text-muted mb-3">or</p>
                      <Button
                        variant="outline-primary"
                        onClick={() => fileInputRef.current.click()}
                      >
                        Browse File
                      </Button>
                      <p className="text-muted mt-2">
                        Supports JPG, PNG, JPEG (Max 10MB)
                      </p>
                    </div>
                    <Form.Control
                      type="file"
                      ref={fileInputRef}
                      onChange={handleFileChange}
                      accept="image/*"
                      className="d-none"
                    />
                  </Card.Body>
                </Card>
              </Col>
            </Row>

            {/* Uploaded Image Preview */}
            {image && (
              <Row className="mb-4">
                <Col>
                  <h5>Selected Image</h5>
                  <div className="position-relative d-inline-block">
                    <Image
                      src={previewUrl}
                      alt="Preview"
                      thumbnail
                      style={{ maxWidth: "300px", height: "auto" }}
                    />
                    <Button
                      variant="danger"
                      size="sm"
                      className="position-absolute top-0 end-0"
                      onClick={removeImage}
                    >
                      <XCircle />
                    </Button>
                    <div className="mt-2">
                      <small className="text-muted">
                        {image.name} ({(image.size / 1024 / 1024).toFixed(2)} MB)
                      </small>
                    </div>
                  </div>
                </Col>
              </Row>
            )}

            {/* Generation Controls */}
            <Row className="mb-4">
              <Col>
                <Button
                  variant="primary"
                  size="lg"
                  onClick={simulate3DGeneration}
                  disabled={!image || isGenerating}
                >
                  {isGenerating ? (
                    <span>Generating Scene... {progress}%</span>
                  ) : (
                    <>
                      <GearWideConnected className="me-2" />
                      Generate 3D Scene
                    </>
                  )}
                </Button>
              </Col>
            </Row>

            {/* Progress Bar */}
            {isGenerating && (
              <Row className="mb-4">
                <Col>
                  <ProgressBar
                    now={progress}
                    label={`${progress}%`}
                    animated
                    striped
                    variant="primary"
                  />
                  <p className="text-center mt-2">
                    Processing scene... this may take a few moments
                  </p>
                </Col>
              </Row>
            )}

            {/* Results Section */}
            {sceneResult && (
              <Row>
                <Col>
                  <h5 className="mb-4">Scene Generated Successfully!</h5>
                  <Card>
                    <Card.Body>
                      <div className="d-flex flex-column flex-md-row align-items-center">
                        <div className="flex-grow-1">
                          <Image
                            src={sceneResult.originalImage}
                            alt="Original"
                            thumbnail
                            style={{ maxWidth: "200px", height: "auto" }}
                          />
                        </div>
                        <div className="mt-3 mt-md-0 ms-md-4 text-center text-md-start">
                          <h6>3D Scene Preview</h6>
                          <p className="text-muted">
                            Your image has been transformed into a 3D scene. The scene file is ready for download.
                          </p>
                          <Button
                            variant="success"
                            onClick={() =>
                              addNotification(
                                "info",
                                "Download functionality would be implemented in a real application"
                              )
                            }
                          >
                            <Download className="me-2" />
                            Download 3D Scene
                          </Button>
                        </div>
                      </div>
                    </Card.Body>
                  </Card>
                </Col>
              </Row>
            )}
          </motion.div>
        </Container>
      </div>
    </div>
  );
};

export default Scene3DPage;
