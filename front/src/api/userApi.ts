import type { User, UserLogin } from "@/shared/types/types";
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
export const useGetDataToken = () => {
  const fetchToken = async () => {
    const response = await axiosInstance.post("api/employee/token");
    return response.data;
  };

  const {
    data: user,
    error,
    isLoading,
  } = useQuery({
    queryKey: ["userToken"],
    queryFn: fetchToken,
    staleTime: 0,
  });

  return {
    user,
    error,
    isLoading,
  };
};

export const Register = async ({
  email,
  password,
  age,
}: User): Promise<string | null> => {
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
