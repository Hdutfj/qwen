import React, { useState, useEffect } from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
  useLocation,
} from "react-router-dom";
import { Container } from "react-bootstrap";
import { AuthProvider, useAuth } from "./contexts/AuthContext";
import { NotificationProvider } from "./contexts/NotificationContext";
import Navigation from "./components/Navigation";
import HomePage from "./pages/HomePage";
import DetectionPage from "./pages/DetectionPage";
import Scene3DPage from "./pages/Scene3DPage";
import AuthPage from "./pages/AuthPage";
import DashboardPage from "./pages/DashboardPage";
import FeatureDemoPage from "./pages/FeatureDemoPage";
import NotificationSystem from "./components/NotificationSystem";
import "./App.css";

/* ============================
   ✅ Protected Route Component
   ============================ */
const ProtectedRoute = ({ children }) => {
  const { currentUser } = useAuth();
  const location = useLocation();

  if (!currentUser) {
    return <Navigate to="/auth" state={{ from: location }} replace />;
  }

  return children;
};

/* ============================
   ✅ AuthWrapper to fix Router issue
   ============================ */
const AuthWrapper = ({ children }) => {
  const { setCurrentUser } = useAuth();

  useEffect(() => {
    const token = localStorage.getItem("token");
    const user = localStorage.getItem("user");

    if (token && user && setCurrentUser) {
      try {
        const parsedUser = JSON.parse(user);
        setCurrentUser({ ...parsedUser, token });
      } catch (err) {
        console.error("Failed to parse user data:", err);
        localStorage.removeItem("token");
        localStorage.removeItem("user");
      }
    }
  }, [setCurrentUser]);

  return children;
};

/* ============================
   ✅ Main App Component
   ============================ */
function App() {
  const [darkMode, setDarkMode] = useState(false);

  // 🌙 Load saved theme
  useEffect(() => {
    const savedTheme = localStorage.getItem("theme");
    if (savedTheme === "dark") {
      setDarkMode(true);
      document.body.classList.add("dark-mode");
    }
  }, []);

  // 🌗 Toggle theme and persist it
  useEffect(() => {
    if (darkMode) {
      document.documentElement.setAttribute("data-theme", "dark");
      document.body.classList.add("dark-mode");
    } else {
      document.documentElement.setAttribute("data-theme", "light");
      document.body.classList.remove("dark-mode");
    }
    localStorage.setItem("theme", darkMode ? "dark" : "light");
  }, [darkMode]);

  return (
    <NotificationProvider>
      {/* ✅ Router must wrap AuthProvider */}
      <Router>
        <AuthProvider>
          <AuthWrapper>
            <div className={`app ${darkMode ? "dark-mode" : ""}`}>
              <NotificationSystem />
              <Navigation
                darkMode={darkMode}
                toggleDarkMode={() => setDarkMode((prev) => !prev)}
              />

              <Container className="main-content">
                <Routes>
                  {/* 🌐 Public Routes */}
                  <Route path="/" element={<HomePage />} />
                  <Route path="/auth" element={<AuthPage />} />

                  {/* 🔐 Protected Routes */}
                  <Route
                    path="/dashboard"
                    element={
                      <ProtectedRoute>
                        <DashboardPage />
                      </ProtectedRoute>
                    }
                  />
                  <Route
                    path="/detection"
                    element={
                      <ProtectedRoute>
                        <DetectionPage />
                      </ProtectedRoute>
                    }
                  />
                  <Route
                    path="/3d-scene"
                    element={
                      <ProtectedRoute>
                        <Scene3DPage />
                      </ProtectedRoute>
                    }
                  />
                  <Route
                    path="/features"
                    element={
                      <ProtectedRoute>
                        <FeatureDemoPage />
                      </ProtectedRoute>
                    }
                  />

                  {/* 🚫 Catch-all */}
                  <Route path="*" element={<Navigate to="/" replace />} />
                </Routes>
              </Container>
            </div>
          </AuthWrapper>
        </AuthProvider>
      </Router>
    </NotificationProvider>
  );
}

export default App;
