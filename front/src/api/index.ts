import axios from "axios";
export const axiosInstance = axios.create({
  baseURL: "https://neuroscan-backend-ogps.onrender.com",
  headers: {
    "Content-Type": "application/json",
  },
  withCredentials: true,
});

export const AiInstance = axios.create({
  baseURL: "https://maykess-neuroscan-aimodule.hf.space", //http://localhost:8000
  headers: {
    "Content-Type": "multipart/form-data",
  },
});
