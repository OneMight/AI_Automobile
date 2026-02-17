import type { DeterminedModel } from "@/shared/types/types";
import { axiosInstance } from ".";
import { useQuery } from "@tanstack/react-query";

export const useGetModels = (id: number | undefined) => {
  const fetchModelsById = async (): Promise<DeterminedModel[]> => {
    const response = await axiosInstance.post(`/api/determinedModel/${id}`);
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
      },
    },
  );
  return await response.data;
};
