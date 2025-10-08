import React, { useState, useEffect } from "react";
import {
  Container,
  Row,
  Col,
  Card,
  Table,
  ProgressBar,
  Badge,
  Button,
} from "react-bootstrap";
import { motion } from "framer-motion";
import {
  BarChartFill,
  ImageFill,
  Clock,
  GraphUp,
  CheckCircle,
} from "react-bootstrap-icons";
import { useAuth } from "../contexts/AuthContext";
import { useNavigate } from "react-router-dom";

const DashboardPage = () => {
  const { currentUser, isAuthenticated, loading, logout } = useAuth();
  const navigate = useNavigate();

  const [dashboardData, setDashboardData] = useState({
    totalDetections: 0,
    detectionHistory: [],
    topLabels: [],
  });

  const [selectedFile, setSelectedFile] = useState(null);
  const [outputUrl, setOutputUrl] = useState("");
  const [processing, setProcessing] = useState(false);

  // ✅ Redirect if not logged in
  useEffect(() => {
    if (!loading && !isAuthenticated) {
      navigate("/login");
    }
  }, [loading, isAuthenticated, navigate]);

  // ✅ Mock Dashboard Data
  useEffect(() => {
    const fetchDashboardData = async () => {
      await new Promise((resolve) => setTimeout(resolve, 400));
      setDashboardData({
        totalDetections: 42,
        detectionHistory: [
          {
            id: 1,
            image: "room1.jpg",
            objects: 4,
            timestamp: "2025-10-07T12:30:00Z",
            status: "completed",
          },
          {
            id: 2,
            image: "bathroom.jpg",
            objects: 3,
            timestamp: "2025-10-06T10:00:00Z",
            status: "completed",
          },
        ],
        topLabels: [
          { label: "person", count: 28 },
          { label: "car", count: 15 },
          { label: "dog", count: 8 },
          { label: "bicycle", count: 6 },
        ],
      });
    };
    fetchDashboardData();
  }, []);

  // ✅ File Selection
  const handleFileChange = (e) => setSelectedFile(e.target.files[0]);

  // ✅ Detect Objects API
  const handleDetectBatch = async () => {
    if (!selectedFile) return alert("Please select an image first!");
    setProcessing(true);
    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
      const res = await fetch("http://localhost:8000/detect-batch", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setOutputUrl(data.output_url || "");
      alert("✅ Object detection complete!");
    } catch (err) {
      console.error("Detection API Error:", err);
      alert("❌ Error in detection API");
    }
    setProcessing(false);
  };

  // ✅ 3D Scene Map API
  const handle3DSceneMap = async () => {
    if (!selectedFile) return alert("Please select an image first!");
    setProcessing(true);
    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
      const res = await fetch("http://localhost:8000/3d-scene-map", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setOutputUrl(data.output_url || "");
      alert("✅ 3D Scene Map generated!");
    } catch (err) {
      console.error("3D Scene API Error:", err);
      alert("❌ Error in 3D Scene Map API");
    }
    setProcessing(false);
  };

  const getStatusBadge = (status) => {
    switch (status) {
      case "completed":
        return <Badge bg="success">Completed</Badge>;
      case "processing":
        return <Badge bg="warning">Processing</Badge>;
      case "failed":
        return <Badge bg="danger">Failed</Badge>;
      default:
        return <Badge bg="secondary">{status}</Badge>;
    }
  };

  if (loading) return <p>Loading...</p>;

  return (
    <div
      style={{
        display: "flex",
        height: "100vh",
        width: "100%",
        overflow: "hidden",
        backgroundColor: "#f9fafb",
      }}
    >
      {/* ===== Sidebar (20%) ===== */}
      <div
        style={{
          position: "fixed",
          left: 0,
          top: 0,
          width: "20%",
          height: "100%",
          background: "linear-gradient(135deg, #111827, #1f2937)",
          color: "#fff",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          padding: "30px 20px",
          zIndex: 10,
        }}
      >
        <div>
          <h3 className="fw-bold mb-4 text-center">VisionAI Panel</h3>

          <input
            type="file"
            id="fileInput"
            style={{ display: "none" }}
            onChange={handleFileChange}
          />

          <Button
            variant="primary"
            className="w-100 mb-3"
            onClick={() => document.getElementById("fileInput").click()}
          >
            Upload Image
          </Button>

          <Button
            variant="success"
            className="w-100 mb-3"
            onClick={handleDetectBatch}
            disabled={processing}
          >
            {processing ? "Detecting..." : "🔍 Detect Objects"}
          </Button>

          <Button
            variant="info"
            className="w-100 mb-3"
            onClick={handle3DSceneMap}
            disabled={processing}
          >
            {processing ? "Processing..." : "🌍 Generate 3D Scene"}
          </Button>
        </div>

        <Button
          variant="danger"
          className="w-100"
          onClick={() => {
            logout();
            navigate("/login");
          }}
        >
          🚪 Logout
        </Button>
      </div>

      {/* ===== Main Dashboard (80%) ===== */}
      <div
        style={{
          marginLeft: "20%",
          width: "80%",
          overflowY: "auto",
          padding: "40px",
        }}
      >
        <Container fluid>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
          >
            {/* Header */}
            <Row className="mb-4 align-items-center">
              <Col>
                <h1 className="fw-bold">
                  Welcome, {currentUser?.name || "User"}
                </h1>
                <p className="text-muted">
                  Manage object detections and generate 3D scenes.
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

            {/* Stats Section */}
            <Row className="mb-4">
              <Col md={4}>
                <Card className="text-center shadow-sm">
                  <Card.Body>
                    <BarChartFill size={40} className="text-primary mb-3" />
                    <h3>{dashboardData.totalDetections}</h3>
                    <p className="text-muted mb-0">Total Detections</p>
                  </Card.Body>
                </Card>
              </Col>
              <Col md={4}>
                <Card className="text-center shadow-sm">
                  <Card.Body>
                    <ImageFill size={40} className="text-success mb-3" />
                    <h3>12</h3>
                    <p className="text-muted mb-0">3D Scenes Generated</p>
                  </Card.Body>
                </Card>
              </Col>
              <Col md={4}>
                <Card className="text-center shadow-sm">
                  <Card.Body>
                    <GraphUp size={40} className="text-info mb-3" />
                    <h3>98%</h3>
                    <p className="text-muted mb-0">Accuracy Rate</p>
                  </Card.Body>
                </Card>
              </Col>
            </Row>

            {/* Analytics Section */}
            <Row className="mb-4">
              {/* Top Detected Objects */}
              <Col lg={6}>
                <Card className="shadow-sm">
                  <Card.Header>
                    <h5 className="mb-0">
                      <CheckCircle className="me-2" /> Top Detected Objects
                    </h5>
                  </Card.Header>
                  <Card.Body>
                    {dashboardData.topLabels.length ? (
                      dashboardData.topLabels.map((item, i) => (
                        <div key={i} className="mb-3">
                          <div className="d-flex justify-content-between mb-1">
                            <span className="fw-bold">{item.label}</span>
                            <span>{item.count}</span>
                          </div>
                          <ProgressBar
                            now={
                              (item.count /
                                Math.max(
                                  ...dashboardData.topLabels.map((l) => l.count)
                                )) *
                              100
                            }
                            variant="primary"
                          />
                        </div>
                      ))
                    ) : (
                      <p className="text-muted text-center my-4">
                        No data available
                      </p>
                    )}
                  </Card.Body>
                </Card>
              </Col>

              {/* Recent Activity */}
              <Col lg={6}>
                <Card className="shadow-sm">
                  <Card.Header>
                    <h5 className="mb-0">
                      <Clock className="me-2" /> Recent Activity
                    </h5>
                  </Card.Header>
                  <Card.Body>
                    {dashboardData.detectionHistory.length ? (
                      <Table responsive>
                        <thead>
                          <tr>
                            <th>Image</th>
                            <th>Objects</th>
                            <th>Date</th>
                            <th>Status</th>
                          </tr>
                        </thead>
                        <tbody>
                          {dashboardData.detectionHistory.map((item) => (
                            <tr key={item.id}>
                              <td>{item.image}</td>
                              <td>{item.objects}</td>
                              <td>
                                {new Date(item.timestamp).toLocaleDateString()}
                              </td>
                              <td>{getStatusBadge(item.status)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </Table>
                    ) : (
                      <p className="text-muted text-center my-4">
                        No recent activity
                      </p>
                    )}
                  </Card.Body>
                </Card>
              </Col>
            </Row>

            {/* Uploaded File Preview */}
            {selectedFile && (
              <Row className="mb-4">
                <Col>
                  <Card className="shadow-sm text-center p-3">
                    <h5>Selected File:</h5>
                    <p>{selectedFile.name}</p>
                    {outputUrl && (
                      <div>
                        <h6>Output Preview:</h6>
                        <img
                          src={`http://localhost:8000/${outputUrl}`}
                          alt="Output"
                          style={{
                            maxWidth: "70%",
                            borderRadius: "10px",
                            marginTop: "10px",
                          }}
                        />
                      </div>
                    )}
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

export default DashboardPage;
