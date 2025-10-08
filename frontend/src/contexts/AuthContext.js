// src/contexts/AuthContext.js
import React, { createContext, useContext, useEffect, useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";

// ✅ Base API URL — change port if your FastAPI backend uses a different one
axios.defaults.baseURL = process.env.REACT_APP_API_URL || "http://127.0.0.1:8000";

const AuthContext = createContext();
export const useAuth = () => useContext(AuthContext);

export const AuthProvider = ({ children }) => {
  const navigate = useNavigate();
  const [currentUser, setCurrentUser] = useState(null);
  const [token, setToken] = useState(localStorage.getItem("token"));
  const [loading, setLoading] = useState(true);

  /* ==========================
     🧩 VERIFY TOKEN ON START
     ========================== */
  useEffect(() => {
    const verifyToken = async () => {
      if (token) {
        axios.defaults.headers.common["Authorization"] = `Bearer ${token}`;
        try {
          const res = await axios.get("/api/auth/profile");
          setCurrentUser(res.data.user);
        } catch (err) {
          console.warn("Token invalid, logging out...");
          logout();
        }
      }
      setLoading(false);
    };
    verifyToken();
  }, [token]);

  /* ==========================
     🔐 LOGIN FUNCTION
     ========================== */
  const login = async (email, password) => {
    try {
      const res = await axios.post("/api/auth/login", { email, password });
      const { token, user } = res.data;

      localStorage.setItem("token", token);
      localStorage.setItem("user", JSON.stringify(user));

      axios.defaults.headers.common["Authorization"] = `Bearer ${token}`;
      setToken(token);
      setCurrentUser(user);

      navigate("/dashboard", { replace: true });
      return { success: true, user };
    } catch (err) {
      console.error("Login failed:", err);
      return {
        success: false,
        message: err.response?.data?.message || "Invalid credentials",
      };
    }
  };

  /* ==========================
     🧾 REGISTER FUNCTION
     ========================== */
  const register = async (name, email, password) => {
    try {
      const res = await axios.post("/api/auth/register", {
        name,
        email,
        password,
      });
      const { token, user } = res.data;

      localStorage.setItem("token", token);
      localStorage.setItem("user", JSON.stringify(user));

      axios.defaults.headers.common["Authorization"] = `Bearer ${token}`;
      setToken(token);
      setCurrentUser(user);

      navigate("/dashboard", { replace: true });
      return { success: true, user };
    } catch (err) {
      console.error("Registration failed:", err);
      return {
        success: false,
        message: err.response?.data?.message || "Registration failed",
      };
    }
  };

  /* ==========================
     🚪 LOGOUT FUNCTION
     ========================== */
  const logout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    delete axios.defaults.headers.common["Authorization"];
    setCurrentUser(null);
    setToken(null);
    navigate("/auth", { replace: true });
  };

  const value = {
    currentUser,
    token,
    isAuthenticated: !!currentUser,
    login,
    register,
    logout,
  };

  return (
    <AuthContext.Provider value={value}>
      {!loading && children}
    </AuthContext.Provider>
  );
};
