import { axiosInstance } from "@/api";
import type { User } from "@/shared/types/types";

export const getUserRole = async (): Promise<User | null> => {
  const refreshToken = localStorage.getItem("refreshToken");

  const response = await axiosInstance.post("api/user/getAuth", {
    refreshToken,
  });
  return response.data;
};
