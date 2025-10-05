import React, { useState } from 'react';
import { Container, Row, Col, Card, Button, Form, Alert, ProgressBar, Image } from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';
import axios from 'axios';

function App() {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.5);

  const handleFileChange = (event) => {
    const files = Array.from(event.target.files);
    setSelectedFiles(files);
    setError('');
  };

  const handleSubmit = async () => {
    if (selectedFiles.length === 0) {
      setError('Please select at least one image file');
      return;
    }

    setLoading(true);
    setError('');
    setResults([]);

    const formData = new FormData();
    selectedFiles.forEach((file) => {
      formData.append('images', file);
    });
    formData.append('confidence_threshold', confidenceThreshold.toString());

    try {
      // Use the correct endpoint based on number of files
      const endpoint = selectedFiles.length === 1 ? '/detect' : '/detect-multiple';
      const response = await axios.post('http://localhost:8000' + endpoint, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (selectedFiles.length === 1) {
        // Single image result
        setResults([response.data]);
      } else {
        // Multiple images result
        setResults(response.data.results);
      }
    } catch (err) {
      console.error('Error:', err);
      setError(err.response?.data?.detail || 'An error occurred while processing the image(s)');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container className="py-4">
      <Row className="mb-4">
        <Col>
          <h1 className="text-center">Home Objects Detection</h1>
          <p className="text-center text-muted">
            Upload images to detect home objects with bounding boxes
          </p>
        </Col>
      </Row>

      <Row>
        <Col md={{ span: 8, offset: 2 }}>
          <Card>
            <Card.Body>
              <Card.Title>Upload Images</Card.Title>
              
              <Form.Group className="mb-3">
                <Form.Label>Select Images</Form.Label>
                <Form.Control
                  type="file"
                  multiple
                  accept="image/*"
                  onChange={handleFileChange}
                  disabled={loading}
                />
                <Form.Text className="text-muted">
                  Select one or more image files (JPG, PNG, etc.)
                </Form.Text>
              </Form.Group>

              <Form.Group className="mb-3">
                <Form.Label>Confidence Threshold: {confidenceThreshold}</Form.Label>
                <Form.Range
                  min="0.1"
                  max="0.9"
                  step="0.05"
                  value={confidenceThreshold}
                  onChange={(e) => setConfidenceThreshold(e.target.value)}
                  disabled={loading}
                />
                <Form.Text className="text-muted">
                  Adjust to set minimum confidence for detections
                </Form.Text>
              </Form.Group>

              <div className="d-grid gap-2">
                <Button
                  variant="primary"
                  onClick={handleSubmit}
                  disabled={loading || selectedFiles.length === 0}
                >
                  {loading ? (
                    <>
                      <span className="spinner-border spinner-border-sm me-2" role="status"></span>
                      Processing...
                    </>
                  ) : (
                    'Detect Objects'
                  )}
                </Button>
              </div>

              {error && <Alert variant="danger" className="mt-3">{error}</Alert>}
            </Card.Body>
          </Card>
        </Col>
      </Row>

      {loading && (
        <Row className="mt-4">
          <Col md={{ span: 6, offset: 3 }}>
            <ProgressBar animated now={100} label="Processing..." />
          </Col>
        </Row>
      )}

      {results.length > 0 && (
        <Row className="mt-4">
          <Col>
            <h3>Detection Results</h3>
            {results.map((result, index) => (
              <div key={index} className="mb-4">
                {result.error ? (
                  <Alert variant="danger">
                    Error processing {result.filename}: {result.error}
                  </Alert>
                ) : (
                  <Card>
                    <Card.Body>
                      <Card.Title>
                        {selectedFiles.length > 1 ? `Result for: ${result.filename}` : 'Detection Result'}
                      </Card.Title>
                      
                      <Row>
                        <Col md={6}>
                          <div className="text-center">
                            <Image
                              src={`http://localhost:8000${result.result_image_url}`}
                              alt="Detection result"
                              fluid
                              style={{ maxHeight: '400px', objectFit: 'contain' }}
                              onError={(e) => {
                                e.target.src = 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300" viewBox="0 0 400 300"><rect width="400" height="300" fill="%23f8f9fa"/><text x="50%" y="50%" text-anchor="middle" dominant-baseline="middle" fill="%236c757d">Image Preview</text></svg>';
                              }}
                            />
                          </div>
                        </Col>
                        <Col md={6}>
                          <h5>Detections:</h5>
                          {result.detections && result.detections.length > 0 ? (
                            <ul className="list-group">
                              {result.detections.map((detection, idx) => (
                                <li key={idx} className="list-group-item">
                                  <strong>{detection.class}</strong> - Confidence: {detection.confidence.toFixed(2)}
                                  <br />
                                  <small>BBox: [{detection.bbox[0]}, {detection.bbox[1]}, {detection.bbox[2]}, {detection.bbox[3]}]</small>
                                </li>
                              ))}
                            </ul>
                          ) : (
                            <p>No objects detected.</p>
                          )}
                          
                          <p className="mt-2">
                            <small className="text-muted">
                              Image size: {result.image_size?.[0]} x {result.image_size?.[1]} pixels
                            </small>
                          </p>
                        </Col>
                      </Row>
                    </Card.Body>
                  </Card>
                )}
              </div>
            ))}
          </Col>
        </Row>
      )}
      
      <Row className="mt-5">
        <Col className="text-center text-muted">
          <small>Home Objects Detection System using PyTorch and FastAPI</small>
        </Col>
      </Row>
    </Container>
  );
}

export default App;