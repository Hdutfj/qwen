// src/pages/AuthPage.js
import React, { useState, useEffect } from "react";
import { Container, Card, Form, Button, Alert } from "react-bootstrap";
import { motion } from "framer-motion";
import {
  PersonFill,
  EnvelopeFill,
  LockFill,
  ArrowRight,
} from "react-bootstrap-icons";
import { useAuth } from "../contexts/AuthContext";
import { useNavigate } from "react-router-dom";

const AuthPage = () => {
  const { login, register } = useAuth();
  const [isLogin, setIsLogin] = useState(true);
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  /* ==========================
     🔁 Auto Redirect if Logged In
     ========================== */
  useEffect(() => {
    const token = localStorage.getItem("token");
    if (token) navigate("/dashboard", { replace: true });
  }, [navigate]);

  /* ==========================
     🔐 Handle Form Submit
     ========================== */
  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    if (!isLogin && password !== confirmPassword) {
      setError("Passwords do not match");
      return;
    }

    setLoading(true);
    const action = isLogin ? login : register;
    const args = isLogin ? [email, password] : [name, email, password];
    const result = await action(...args);
    setLoading(false);

    if (!result.success) {
      setError(result.message);
    } else {
      navigate("/dashboard", { replace: true });
    }
  };

  /* ==========================
     🧩 Render Form
     ========================== */
  return (
    <Container
      className="d-flex align-items-center justify-content-center"
      style={{ minHeight: "90vh" }}
    >
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="w-100"
        style={{ maxWidth: "400px" }}
      >
        <Card className="shadow-lg border-0">
          <Card.Body className="p-4">
            <div className="text-center mb-4">
              <h2 className="fw-bold">
                {isLogin ? "Welcome Back 👋" : "Create Account 🚀"}
              </h2>
              <p className="text-muted">
                {isLogin
                  ? "Sign in to access your AI tools"
                  : "Join and explore our AI detection system"}
              </p>
            </div>

            {error && <Alert variant="danger">{error}</Alert>}

            <Form onSubmit={handleSubmit}>
              {!isLogin && (
                <Form.Group className="mb-3" controlId="name">
                  <Form.Label>Full Name</Form.Label>
                  <div className="input-group">
                    <span className="input-group-text">
                      <PersonFill />
                    </span>
                    <Form.Control
                      type="text"
                      placeholder="Enter your name"
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      required={!isLogin}
                    />
                  </div>
                </Form.Group>
              )}

              <Form.Group className="mb-3" controlId="email">
                <Form.Label>Email Address</Form.Label>
                <div className="input-group">
                  <span className="input-group-text">
                    <EnvelopeFill />
                  </span>
                  <Form.Control
                    type="email"
                    placeholder="Enter your email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required
                  />
                </div>
              </Form.Group>

              <Form.Group className="mb-3" controlId="password">
                <Form.Label>Password</Form.Label>
                <div className="input-group">
                  <span className="input-group-text">
                    <LockFill />
                  </span>
                  <Form.Control
                    type="password"
                    placeholder="Enter your password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
                  />
                </div>
              </Form.Group>

              {!isLogin && (
                <Form.Group className="mb-3" controlId="confirmPassword">
                  <Form.Label>Confirm Password</Form.Label>
                  <div className="input-group">
                    <span className="input-group-text">
                      <LockFill />
                    </span>
                    <Form.Control
                      type="password"
                      placeholder="Confirm password"
                      value={confirmPassword}
                      onChange={(e) => setConfirmPassword(e.target.value)}
                      required
                    />
                  </div>
                </Form.Group>
              )}

              <div className="d-grid">
                <Button
                  variant="primary"
                  type="submit"
                  disabled={loading}
                  className="btn-animated"
                >
                  {loading
                    ? "Processing..."
                    : isLogin
                    ? "Sign In"
                    : "Sign Up"}
                  <ArrowRight className="ms-2" />
                </Button>
              </div>
            </Form>

            <div className="text-center mt-4">
              <Button
                variant="link"
                onClick={() => setIsLogin(!isLogin)}
                className="text-decoration-none"
              >
                {isLogin
                  ? "Don't have an account? Sign Up"
                  : "Already have an account? Sign In"}
              </Button>
            </div>
          </Card.Body>
        </Card>
      </motion.div>
    </Container>
  );
};

export default AuthPage;
