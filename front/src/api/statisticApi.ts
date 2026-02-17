import type { Statistic } from "@/shared/types/types";
import { axiosInstance } from ".";
import { useQuery } from "@tanstack/react-query";
export const useGetStatisticsById = (id: number | undefined) => {
  const fetchById = async (): Promise<Statistic> => {
    const response = await axiosInstance.get(`/api/statistics/${id}`);
    return response.data;
  };
  const {
    data: statistic,
    isError,
    isLoading: isStatisticLoading,
  } = useQuery({
    queryKey: ["statistics", id],
    queryFn: fetchById,
    retry: 0,
    enabled: !!id,
    staleTime: 0,
    gcTime: 0,
  });
  return {
    statistic,
    isError,
    isStatisticLoading,
  };
};

export const updateStatistic = async (id: number, idModel: number) => {
  await axiosInstance.put(`api/statistics/${id}`, {
    idModel,
  });
};
