import axios from "axios";
export const axiosInstance = axios.create({
  baseURL: "http://localhost:5000",
  headers: {
    "Content-Type": "application/json",
  },
  withCredentials: true,
});

export const AiInstance = axios.create({
  baseURL: "https://neuroscan-ai-module.onrender.com",
  headers: {
    "Content-Type": "multipart/form-data",
  },
});
