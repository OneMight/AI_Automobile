import axios from "axios";
export const axiosInstance = axios.create({
  baseURL: `${import.meta.env.VITE_API_URL}`,
  headers: {
    "Content-Type": "application/json",
    Authorization: `Bearer ${localStorage.getItem("refreshToken")}`,
  },
  withCredentials: true,
});

export const AiInstance = axios.create({
  baseURL: "https://maykess-neuroscan-aimodule.hf.space",
  headers: {
    "Content-Type": "multipart/form-data",
  },
});
axiosInstance.interceptors.request.use((config) => {
  config.headers.Authorization = `Bearer ${localStorage.getItem("refreshToken")}`;
  return config;
});
