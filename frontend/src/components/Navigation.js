import React, { useState } from 'react';
import { Navbar, Nav, NavDropdown, Container, Button, Row, Col } from 'react-bootstrap';
import { Link, useLocation } from 'react-router-dom';
import { PersonFill, HouseFill, GearFill, ImageFill, GearWideConnected, MoonFill, SunFill, BarChartFill, ThreeDotsVertical } from 'react-bootstrap-icons';
import { useAuth } from '../contexts/AuthContext';

const Navigation = ({ darkMode, toggleDarkMode }) => {
  const { currentUser, logout } = useAuth();
  const location = useLocation();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const isActive = (path) => location.pathname === path;

  return (
    <>
      {/* Desktop Sidebar Navigation */}
      <div className="sidebar">
        <Container className="d-flex flex-column h-100">
          <div className="d-flex justify-content-between align-items-center mb-4 px-2">
            <Link to="/" className="text-decoration-none">
              <h4 className="text-primary mb-0">AI Vision</h4>
            </Link>
            <Button 
              variant="link" 
              onClick={toggleDarkMode}
              className="p-0 text-decoration-none border-0 bg-transparent"
            >
              {darkMode ? <SunFill size={20} /> : <MoonFill size={20} />}
            </Button>
          </div>
          
          <Nav className="flex-column flex-grow-1">
            <Nav.Link 
              as={Link} 
              to="/" 
              className={`nav-link ${isActive('/') ? 'active' : ''}`}
            >
              <HouseFill className="me-2" /> Home
            </Nav.Link>
            
            {currentUser ? (
              <>
                <Nav.Link 
                  as={Link} 
                  to="/dashboard" 
                  className={`nav-link ${isActive('/dashboard') ? 'active' : ''}`}
                >
                  <BarChartFill className="me-2" /> Dashboard
                </Nav.Link>
                <Nav.Link 
                  as={Link} 
                  to="/detection" 
                  className={`nav-link ${isActive('/detection') ? 'active' : ''}`}
                >
                  <ImageFill className="me-2" /> Object Detection
                </Nav.Link>
                <Nav.Link 
                  as={Link} 
                  to="/3d-scene" 
                  className={`nav-link ${isActive('/3d-scene') ? 'active' : ''}`}
                >
                  <GearWideConnected className="me-2" /> 3D Scene
                </Nav.Link>
                <Nav.Link 
                  as={Link} 
                  to="/features" 
                  className={`nav-link ${isActive('/features') ? 'active' : ''}`}
                >
                  <GearFill className="me-2" /> Features
                </Nav.Link>
                
                <NavDropdown 
                  title={
                    <span>
                      <PersonFill className="me-2" /> {currentUser.name}
                    </span>
                  } 
                  id="user-dropdown"
                  className="mt-auto"
                >
                  <NavDropdown.Item onClick={logout}>Logout</NavDropdown.Item>
                </NavDropdown>
              </>
            ) : (
              <Nav.Link 
                as={Link} 
                to="/auth" 
                className={`nav-link ${isActive('/auth') ? 'active' : ''}`}
              >
                <PersonFill className="me-2" /> Login
              </Nav.Link>
            )}
          </Nav>
        </Container>
      </div>

      {/* Mobile Bottom Navigation */}
      <div className="mobile-nav d-md-none">
        <Row className="g-0">
          <Col>
            <Nav.Link 
              as={Link} 
              to="/" 
              className={`nav-link p-2 text-center ${isActive('/') ? 'active' : ''}`}
            >
              <HouseFill size={20} className="d-block mx-auto mb-1" />
              <small>Home</small>
            </Nav.Link>
          </Col>
          {currentUser && (
            <>
              <Col>
                <Nav.Link 
                  as={Link} 
                  to="/detection" 
                  className={`nav-link p-2 text-center ${isActive('/detection') ? 'active' : ''}`}
                >
                  <ImageFill size={20} className="d-block mx-auto mb-1" />
                  <small>Detect</small>
                </Nav.Link>
              </Col>
              <Col>
                <Nav.Link 
                  as={Link} 
                  to="/3d-scene" 
                  className={`nav-link p-2 text-center ${isActive('/3d-scene') ? 'active' : ''}`}
                >
                  <GearWideConnected size={20} className="d-block mx-auto mb-1" />
                  <small>3D</small>
                </Nav.Link>
              </Col>
              <Col>
                <Nav.Link 
                  as={Link} 
                  to="/dashboard" 
                  className={`nav-link p-2 text-center ${isActive('/dashboard') ? 'active' : ''}`}
                >
                  <BarChartFill size={20} className="d-block mx-auto mb-1" />
                  <small>Stats</small>
                </Nav.Link>
              </Col>
            </>
          )}
          <Col>
            {currentUser ? (
              <Nav.Link 
                onClick={logout}
                className="nav-link p-2 text-center"
              >
                <PersonFill size={20} className="d-block mx-auto mb-1" />
                <small>Logout</small>
              </Nav.Link>
            ) : (
              <Nav.Link 
                as={Link} 
                to="/auth" 
                className={`nav-link p-2 text-center ${isActive('/auth') ? 'active' : ''}`}
              >
                <PersonFill size={20} className="d-block mx-auto mb-1" />
                <small>Account</small>
              </Nav.Link>
            )}
          </Col>
        </Row>
      </div>
    </>
  );
};

export default Navigation;