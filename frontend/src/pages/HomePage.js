import React, { useState } from 'react';
import { Container, Row, Col, Card, Button, Carousel } from 'react-bootstrap';
import { motion } from 'framer-motion';
import { ArrowRight, EyeFill, Badge3d, CheckCircle, StarFill, HouseDoor, PersonCircle, Gear, Grid, BoxArrowRight } from 'react-bootstrap-icons';
import { Link } from 'react-router-dom';

const HomePage = () => {
  const [expandedFeatures, setExpandedFeatures] = useState({});

  const features = [
    {
      id: 1,
      title: "AI-Powered Object Detection",
      description: "Detect and identify objects in images with state-of-the-art accuracy using deep learning models.",
      details: "Our object detection system uses advanced neural networks trained on millions of images to accurately identify and locate objects in real-time."
    },
    {
      id: 2,
      title: "3D Scene Reconstruction",
      description: "Transform 2D images into immersive 3D environments with our advanced mapping technology.",
      details: "Generate detailed 3D representations of real-world environments from simple 2D images, enabling applications in AR, VR, and spatial computing."
    },
    {
      id: 3,
      title: "Real-time Processing",
      description: "Experience blazing fast processing speeds with our optimized algorithms.",
      details: "Our system processes images in real-time, enabling interactive applications and immediate feedback."
    },
    {
      id: 4,
      title: "Intelligent Analysis",
      description: "Get detailed insights and analytics about detected objects.",
      details: "Beyond just detection, our system provides contextual information, object relationships, and behavioral analysis."
    }
  ];

  const testimonials = [
    {
      id: 1,
      name: "Alex Johnson",
      role: "Product Manager",
      company: "TechVision",
      content: "This platform transformed our approach to visual content analysis. The accuracy is remarkable!",
      rating: 5
    },
    {
      id: 2,
      name: "Sarah Williams",
      role: "Research Lead",
      company: "InnovateX",
      content: "The 3D mapping capabilities are game-changing for our spatial computing projects.",
      rating: 5
    },
    {
      id: 3,
      name: "Michael Chen",
      role: "CTO",
      company: "RoboTech",
      content: "Implementation was seamless and the API integration couldn't be easier.",
      rating: 4
    }
  ];

  const toggleFeature = (id) => {
    setExpandedFeatures(prev => ({
      ...prev,
      [id]: !prev[id]
    }));
  };

  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
      {/* ===== Sidebar (20%) ===== */}
      <div
        style={{
          position: 'fixed',
          left: 0,
          top: 0,
          width: '20%',
          height: '100%',
          background: 'linear-gradient(135deg, #111827, #1f2937)',
          color: '#fff',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between',
          padding: '30px 20px',
          zIndex: 10,
        }}
      >
        <div>
          <h3 className="fw-bold mb-4 text-center">VisionAI</h3>

          <Link to="/">
            <Button variant="light" className="w-100 mb-3 text-start">
              <HouseDoor className="me-2" /> Home
            </Button>
          </Link>
          <Link to="/dashboard">
            <Button variant="light" className="w-100 mb-3 text-start">
              <Grid className="me-2" /> Dashboard
            </Button>
          </Link>
          <Link to="/auth">
            <Button variant="light" className="w-100 mb-3 text-start">
              <PersonCircle className="me-2" /> Login / Sign Up
            </Button>
          </Link>
          <Link to="/settings">
            <Button variant="light" className="w-100 mb-3 text-start">
              <Gear className="me-2" /> Settings
            </Button>
          </Link>
        </div>

        <Button
          variant="danger"
          className="w-100"
          onClick={() => alert('Logging out...')}
        >
          <BoxArrowRight className="me-2" /> Logout
        </Button>
      </div>

      {/* ===== Main Content (80%) ===== */}
      <div
        style={{
          marginLeft: '20%',
          width: '80%',
          overflowY: 'auto',
        }}
      >
        <main>
          {/* Hero Section */}
          <section className="hero py-5" style={{ backgroundColor: '#f9fafb' }}>
            <Container>
              <motion.div
                className="hero-content"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8 }}
              >
                <h1>Experience the Future of Vision</h1>
                <h2 className="mb-4">AI Object Detection & 3D Mapping System</h2>
                <p className="lead mb-4">
                  Harness the power of artificial intelligence to detect, analyze, and visualize your world in three dimensions.
                </p>
                <Link to="/auth">
                  <Button variant="primary" size="lg" className="btn-animated">
                    Get Started <ArrowRight className="ms-2" />
                  </Button>
                </Link>
              </motion.div>
            </Container>
          </section>

          {/* Features Section */}
          <section className="py-5">
            <Container>
              <Row className="mb-5">
                <Col lg={12}>
                  <motion.h2
                    className="text-center mb-4"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2, duration: 0.6 }}
                  >
                    Powerful Features
                  </motion.h2>
                  <p className="text-center text-muted mb-5">
                    Everything you need to transform visual data into actionable insights
                  </p>
                </Col>
              </Row>

              <Row>
                {features.map((feature, index) => (
                  <Col md={6} lg={3} className="mb-4" key={feature.id}>
                    <motion.div
                      initial={{ opacity: 0, y: 30 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.1 + 0.2, duration: 0.5 }}
                    >
                      <Card className="h-100 feature-card">
                        <Card.Body className="d-flex flex-column">
                          <div className="feature-icon mb-3">
                            {feature.id === 1 && <EyeFill size={40} className="text-primary" />}
                            {feature.id === 2 && <Badge3d size={40} className="text-primary" />}
                            {feature.id === 3 && <CheckCircle size={40} className="text-primary" />}
                            {feature.id === 4 && <StarFill size={40} className="text-primary" />}
                          </div>
                          <Card.Title>{feature.title}</Card.Title>
                          <Card.Text className="flex-grow-1">{feature.description}</Card.Text>
                          <Button
                            variant="link"
                            className="mt-auto p-0 text-decoration-none"
                            onClick={() => toggleFeature(feature.id)}
                          >
                            {expandedFeatures[feature.id] ? 'Show Less' : 'Learn More'}
                          </Button>
                          {expandedFeatures[feature.id] && (
                            <div className="mt-2">
                              <small className="text-muted">{feature.details}</small>
                            </div>
                          )}
                        </Card.Body>
                      </Card>
                    </motion.div>
                  </Col>
                ))}
              </Row>
            </Container>
          </section>

          {/* How It Works Section */}
          <section className="py-5 bg-light">
            <Container>
              <Row className="mb-5">
                <Col lg={12}>
                  <motion.h2
                    className="text-center mb-4"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2, duration: 0.6 }}
                  >
                    How It Works
                  </motion.h2>
                  <p className="text-center text-muted mb-5">
                    Simple steps to transform your visual data
                  </p>
                </Col>
              </Row>

              <Row className="gy-4">
                {[1, 2, 3].map((step, index) => (
                  <Col md={4} key={step}>
                    <motion.div
                      className="text-center"
                      initial={{ opacity: 0, y: 30 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.2 + 0.3, duration: 0.5 }}
                    >
                      <div className="step-number d-inline-flex align-items-center justify-content-center mb-3">
                        <span className="fs-4">{step}</span>
                      </div>
                      <h4>
                        {step === 1 && 'Upload Image'}
                        {step === 2 && 'AI Processing'}
                        {step === 3 && 'View Results'}
                      </h4>
                      <p className="text-muted">
                        {step === 1 && 'Upload your image or select from our sample gallery'}
                        {step === 2 && 'Our AI processes the image and detects objects'}
                        {step === 3 && 'Get detailed results with bounding boxes and labels'}
                      </p>
                    </motion.div>
                  </Col>
                ))}
              </Row>
            </Container>
          </section>

          {/* Testimonials Section */}
          <section className="py-5">
            <Container>
              <Row className="mb-5">
                <Col lg={12}>
                  <motion.h2
                    className="text-center mb-4"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2, duration: 0.6 }}
                  >
                    What Our Users Say
                  </motion.h2>
                </Col>
              </Row>

              <Row>
                <Col lg={8} className="mx-auto">
                  <Carousel indicators interval={5000}>
                    {testimonials.map(testimonial => (
                      <Carousel.Item key={testimonial.id}>
                        <Card className="text-center">
                          <Card.Body>
                            <div className="rating mb-3">
                              {[...Array(5)].map((_, i) => (
                                <StarFill
                                  key={i}
                                  className={i < testimonial.rating ? "text-warning" : "text-secondary"}
                                />
                              ))}
                            </div>
                            <Card.Text className="fst-italic">"{testimonial.content}"</Card.Text>
                            <Card.Title>{testimonial.name}</Card.Title>
                            <Card.Subtitle className="text-muted">{testimonial.role}, {testimonial.company}</Card.Subtitle>
                          </Card.Body>
                        </Card>
                      </Carousel.Item>
                    ))}
                  </Carousel>
                </Col>
              </Row>
            </Container>
          </section>

          {/* CTA Section */}
          <section className="py-5" style={{ backgroundColor: '#f9fafb' }}>
            <Container>
              <Row className="align-items-center">
                <Col lg={8}>
                  <h2 className="mb-3">Ready to Transform Your Visual Data?</h2>
                  <p className="lead mb-0">
                    Join thousands of developers and researchers using our platform for advanced object detection and 3D mapping.
                  </p>
                </Col>
                <Col lg={4} className="text-lg-end mt-4 mt-lg-0">
                  <Link to="/auth">
                    <Button variant="primary" size="lg" className="btn-animated">
                      Start Free Trial
                    </Button>
                  </Link>
                </Col>
              </Row>
            </Container>
          </section>
        </main>
      </div>
    </div>
  );
};

export default HomePage;
