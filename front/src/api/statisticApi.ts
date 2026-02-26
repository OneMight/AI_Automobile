import type { Statistic } from "@/shared/types/types";
import { axiosInstance } from ".";
import { useQuery } from "@tanstack/react-query";
import type { AgeData, RecognitionData } from "@/shared/types/interfaces";
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
export const useGetAgesStatistic = () => {
  const fetchAgesStat = async (): Promise<AgeData[]> => {
    const response = await axiosInstance.get("/api/statistics/ages");
    return response.data;
  };
  const {
    data: ages,
    isError,
    isLoading: isAgesLoading,
  } = useQuery({
    queryKey: ["fetchAgesStat"],
    queryFn: fetchAgesStat,

    staleTime: 0,
    gcTime: 0,
  });
  return {
    ages,
    isError,
    isAgesLoading,
  };
};
export const useGetRecognitionStatistic = () => {
  const fetchRecognitionsStat = async (): Promise<RecognitionData[]> => {
    const response = await axiosInstance.get("/api/statistics/recognitions");
    return response.data;
  };
  const {
    data: recogniontion,
    isError,
    isLoading: isRecognitionLoading,
  } = useQuery({
    queryKey: ["fetchRecognitionsStat"],
    queryFn: fetchRecognitionsStat,

    staleTime: 0,
    gcTime: 0,
  });
  return {
    recogniontion,
    isError,
    isRecognitionLoading,
  };
};
