import type { UserLogin } from "@/shared/types/types";
import { axiosInstance } from "./index";
import { useQuery } from "@tanstack/react-query";

export const LoginUser = async ({
  email,
  password,
}: UserLogin): Promise<{ message: string }> => {
  try {
    const response = await axiosInstance.post("/api/user/login", {
      email,
      password,
    });
    return response.data;
  } catch (error) {
    console.log(error);
    return { message: "email and password is incorrect" };
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
