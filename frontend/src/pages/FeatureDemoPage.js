import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Card, Button, Spinner, Badge } from 'react-bootstrap';
import { motion } from 'framer-motion';
import { 
  SunFill, 
  MoonFill, 
  ArrowRepeat, 
  CheckCircle, 
  ExclamationCircle, 
  InfoCircle,
  Download,
  PlayFill,
  PauseFill
} from 'react-bootstrap-icons';
import { useNotifications } from '../contexts/NotificationContext';

const FeatureDemoPage = () => {
  const [loading, setLoading] = useState(false);
  const [darkMode, setDarkMode] = useState(false);
  const [count, setCount] = useState(0);
  const { addNotification } = useNotifications();

  // Load theme from localStorage
  useEffect(() => {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
      setDarkMode(true);
      document.documentElement.setAttribute('data-theme', 'dark');
      document.body.classList.add('dark-mode');
    }
  }, []);

  const triggerNotification = (type) => {
    switch(type) {
      case 'success':
        addNotification('success', 'Operation completed successfully!');
        break;
      case 'error':
        addNotification('error', 'An error occurred during the operation.');
        break;
      case 'info':
        addNotification('info', 'This is an informational message.');
        break;
      default:
        addNotification('info', 'New notification');
    }
  };

  const toggleDarkMode = () => {
    const newMode = !darkMode;
    setDarkMode(newMode);
    document.documentElement.setAttribute('data-theme', newMode ? 'dark' : 'light');
    document.body.classList.toggle('dark-mode', newMode);
    localStorage.setItem('theme', newMode ? 'dark' : 'light');
  };

  const simulateLoading = () => {
    setLoading(true);
    setTimeout(() => setLoading(false), 2000);
  };

  return (
    <div style={{ display: 'flex', minHeight: '100vh', overflow: 'hidden' }}>
      
      {/* ===== Sidebar ===== */}
      <div
        style={{
          width: '220px',
          backgroundColor: '#000',
          color: '#fff',
          display: 'flex',
          flexDirection: 'column',
          padding: '20px',
        }}
      >
        <h3 className="mb-4">MyApp</h3>
        <Button
          variant="outline-light"
          onClick={toggleDarkMode}
          className="mb-3"
        >
          {darkMode ? <SunFill className="me-2" /> : <MoonFill className="me-2" />}
          {darkMode ? 'Light Mode' : 'Dark Mode'}
        </Button>
        <nav style={{ flexGrow: 1 }}>
          <ul style={{ listStyle: 'none', padding: 0 }}>
            <li className="mb-2"><a href="#" style={{ color: '#fff', textDecoration: 'none' }}>Dashboard</a></li>
            <li className="mb-2"><a href="#" style={{ color: '#fff', textDecoration: 'none' }}>Features</a></li>
            <li className="mb-2"><a href="#" style={{ color: '#fff', textDecoration: 'none' }}>Settings</a></li>
          </ul>
        </nav>
        <p className="mt-auto text-muted" style={{ fontSize: '12px' }}>© 2025 MyApp</p>
      </div>

      {/* ===== Main Content ===== */}
      <div
        style={{
          flexGrow: 1,
          backgroundColor: darkMode ? '#121212' : '#f9fafb',
          padding: '40px',
          overflowY: 'auto',
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
                <h1 className="fw-bold">Feature Showcase</h1>
                <p className="text-muted">Explore the various UI components and features of our application</p>
              </Col>
            </Row>

            {/* Theme Toggle */}
            <Row className="mb-4">
              <Col>
                <Card>
                  <Card.Header>
                    <h5 className="mb-0"><SunFill className="me-2" />Theme Toggle</h5>
                  </Card.Header>
                  <Card.Body>
                    <p>Switch between light and dark modes:</p>
                    <Button 
                      variant={darkMode ? "light" : "dark"} 
                      onClick={toggleDarkMode}
                    >
                      {darkMode ? <SunFill className="me-2" /> : <MoonFill className="me-2" />}
                      {darkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
                    </Button>
                    <p className="mt-3">
                      Current theme: <Badge bg={darkMode ? "dark" : "light"} text={darkMode ? "light" : "dark"}>
                        {darkMode ? 'Dark' : 'Light'}
                      </Badge>
                    </p>
                  </Card.Body>
                </Card>
              </Col>
            </Row>

            {/* Button Styles */}
            <Row className="mb-4">
              <Col>
                <Card>
                  <Card.Header>
                    <h5 className="mb-0"><PlayFill className="me-2" />Button Styles</h5>
                  </Card.Header>
                  <Card.Body>
                    <div className="d-flex flex-wrap gap-2">
                      <Button variant="primary">Primary</Button>
                      <Button variant="secondary">Secondary</Button>
                      <Button variant="success">Success</Button>
                      <Button variant="danger">Danger</Button>
                      <Button variant="warning">Warning</Button>
                      <Button variant="info">Info</Button>
                      <Button variant="light">Light</Button>
                      <Button variant="dark">Dark</Button>
                      <Button variant="link">Link</Button>
                    </div>
                    <div className="d-flex flex-wrap gap-2 mt-3">
                      <Button variant="outline-primary">Outline Primary</Button>
                      <Button variant="outline-success">Outline Success</Button>
                      <Button variant="outline-danger">Outline Danger</Button>
                    </div>
                  </Card.Body>
                </Card>
              </Col>
            </Row>

            {/* Loading Demo */}
            <Row className="mb-4">
              <Col>
                <Card>
                  <Card.Header>
                    <h5 className="mb-0"><ArrowRepeat className="me-2" />Loading States</h5>
                  </Card.Header>
                  <Card.Body>
                    <div className="d-flex flex-wrap align-items-center gap-3">
                      <Button 
                        variant="primary" 
                        onClick={() => { setLoading(true); setTimeout(() => setLoading(false), 2000); }}
                        disabled={loading}
                      >
                        {loading ? <Spinner animation="border" size="sm" className="me-2" /> : <ArrowRepeat className="me-2" />}
                        {loading ? 'Loading...' : 'Simulate Loading'}
                      </Button>
                    </div>
                  </Card.Body>
                </Card>
              </Col>
            </Row>

            {/* Notifications */}
            <Row className="mb-4">
              <Col>
                <Card>
                  <Card.Header>
                    <h5 className="mb-0"><InfoCircle className="me-2" />Notifications</h5>
                  </Card.Header>
                  <Card.Body>
                    <div className="d-flex flex-wrap gap-2">
                      <Button variant="success" onClick={() => triggerNotification('success')}>
                        <CheckCircle className="me-2" />Success
                      </Button>
                      <Button variant="danger" onClick={() => triggerNotification('error')}>
                        <ExclamationCircle className="me-2" />Error
                      </Button>
                      <Button variant="info" onClick={() => triggerNotification('info')}>
                        <InfoCircle className="me-2" />Info
                      </Button>
                    </div>
                  </Card.Body>
                </Card>
              </Col>
            </Row>

            {/* Counter */}
            <Row className="mb-4">
              <Col>
                <Card>
                  <Card.Header>
                    <h5 className="mb-0"><CheckCircle className="me-2" />Interactive Component</h5>
                  </Card.Header>
                  <Card.Body>
                    <p>Current count: <strong>{count}</strong></p>
                    <div className="d-flex gap-2 flex-wrap">
                      <Button variant="primary" onClick={() => setCount(count + 1)}>Increment</Button>
                      <Button variant="secondary" onClick={() => setCount(count - 1)}>Decrement</Button>
                      <Button variant="outline-secondary" onClick={() => setCount(0)}>Reset</Button>
                    </div>
                  </Card.Body>
                </Card>
              </Col>
            </Row>

            {/* Responsive Grid */}
            <Row className="mb-4">
              <Col>
                <Card>
                  <Card.Header>
                    <h5 className="mb-0"><Download className="me-2" />Responsive Grid</h5>
                  </Card.Header>
                  <Card.Body>
                    <Row>
                      {[1,2,3,4,5,6].map(i => (
                        <Col key={i} xs={12} sm={6} md={4} lg={2} className="mb-3">
                          <Card className="h-100 text-center">
                            <Card.Body className="d-flex flex-column align-items-center justify-content-center">
                              <div className="bg-primary text-white rounded-circle d-flex align-items-center justify-content-center" style={{ width: '50px', height: '50px' }}>{i}</div>
                              <Card.Title className="mt-2">Card {i}</Card.Title>
                            </Card.Body>
                          </Card>
                        </Col>
                      ))}
                    </Row>
                  </Card.Body>
                </Card>
              </Col>
            </Row>

            {/* Animations */}
            <Row className="mb-4">
              <Col>
                <Card>
                  <Card.Header>
                    <h5 className="mb-0"><PauseFill className="me-2" />Animations</h5>
                  </Card.Header>
                  <Card.Body>
                    <div className="d-flex flex-wrap gap-4">
                      <motion.div whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.9 }}>
                        <Button variant="primary">Hover & Tap</Button>
                      </motion.div>
                      <motion.div animate={{ rotate: [0,10,-10,0] }} transition={{ duration: 0.5, repeat: Infinity, repeatType: 'reverse' }}>
                        <Button variant="success">Rotation</Button>
                      </motion.div>
                      <motion.div animate={{ scale: [1,1.2,1] }} transition={{ duration: 2, repeat: Infinity }}>
                        <Button variant="warning">Pulsing</Button>
                      </motion.div>
                    </div>
                  </Card.Body>
                </Card>
              </Col>
            </Row>

          </motion.div>
        </Container>
      </div>
    </div>
  );
};

export default FeatureDemoPage;
