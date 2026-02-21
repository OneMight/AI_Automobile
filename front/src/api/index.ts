import axios from "axios";
export const axiosInstance = axios.create({
  baseURL: `${import.meta.env.VITE_API_URL}`, //https://neuroscan-backend-ogps.onrender.com
  headers: {
    "Content-Type": "application/json",
    Authorization: `Bearer ${localStorage.getItem("token")}`,
  },
  withCredentials: true,
});

export const AiInstance = axios.create({
  baseURL: "https://maykess-neuroscan-aimodule.hf.space", //http://localhost:8000
  headers: {
    "Content-Type": "multipart/form-data",
  },
});
axiosInstance.interceptors.request.use((config) => {
  config.headers.Authorization = `Bearer ${localStorage.getItem("token")}`;
  return config;
});
