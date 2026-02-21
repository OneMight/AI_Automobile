import type { RegisterUser, User, UserLogin } from "@/shared/types/types";
import { axiosInstance } from "./index";
import { useQuery } from "@tanstack/react-query";
import axios from "axios";
type BackendError = {
  message: string;
};

export const LoginUser = async ({
  email,
  password,
}: UserLogin): Promise<string | null> => {
  try {
    const response = await axiosInstance.post("/api/user/login", {
      email,
      password,
    });
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      const serverMessage = (error.response?.data as BackendError)?.message;

      return serverMessage || error.message;
    }
    return "Неизвестная ошибка";
  }
};
export const useGetDataToken = (refreshToken: string | null) => {
  const fetchToken = async (): Promise<User | null> => {
    const response = await axiosInstance.post("api/user/getAuth", {
      refreshToken: refreshToken,
    });
    return response.data;
  };

  const {
    data: user,
    isError,
    isLoading,
  } = useQuery({
    queryKey: ["userToken", refreshToken],
    queryFn: fetchToken,
    retry: 0,
    enabled: !!refreshToken,
    staleTime: 0,
    gcTime: 0,
  });

  return {
    user,
    isError,
    isLoading,
  };
};

export const Register = async ({
  email,
  password,
  age,
}: RegisterUser): Promise<string | null> => {
  try {
    const response = await axiosInstance.post("/api/user/register", {
      email,
      password,
      age,
    });
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      const serverMessage = (error.response?.data as BackendError)?.message;

      return serverMessage || error.message;
    }
    return "Неизвестная ошибка";
  }
};
export const Logout = () => {
  localStorage.removeItem("refreshToken");
  axiosInstance.post("/api/user/logout");
};
