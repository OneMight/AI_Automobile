import type { DeterminedModel, OwnerResponse } from "@/shared/types/types";
import { axiosInstance } from ".";
import { useQuery } from "@tanstack/react-query";

export const useGetModels = (id: number | undefined, limit?: number) => {
  const fetchModelsById = async (): Promise<DeterminedModel[]> => {
    const response = await axiosInstance.post(`/api/determinedModel/${id}`, {
      limit,
    });
    return response.data;
  };
  const {
    data: models,
    isError,
    isLoading,
  } = useQuery({
    queryKey: ["determinedModels", id],
    queryFn: fetchModelsById,
    retry: 0,
    enabled: !!id,
    staleTime: 0,
    gcTime: 0,
  });
  return {
    models,
    isError,
    isLoading,
  };
};
export const postModel = async (
  model: FormData,
  id: number | undefined,
): Promise<number> => {
  const response = await axiosInstance.post(
    `/api/determinedModel/post/${id}`,
    model,
    {
      headers: {
        "Content-Type": undefined,
        Authorization: `Bearer ${localStorage.getItem("token")}`,
      },
    },
  );
  return await response.data;
};

export const useGetAllModels = () => {
  const getAllModels = async (): Promise<OwnerResponse> => {
    const response = await axiosInstance.get(`/api/determinedModel/`);
    return response.data;
  };

  const { data, isLoading, isError } = useQuery({
    queryKey: ["getAllModels"],
    queryFn: getAllModels,
    staleTime: 0,
    gcTime: 0,
  });
  return {
    data,
    isLoading,
    isError,
  };
};
