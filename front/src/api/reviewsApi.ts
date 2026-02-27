import { useInfiniteQuery, useQuery } from "@tanstack/react-query";
import { axiosInstance } from ".";
import type { ReviewsResponse } from "@/shared/types/types";

export const useGetReviews = () => {
  return useInfiniteQuery<ReviewsResponse>({
    queryKey: ["reviews"],
    queryFn: async ({ pageParam = 0 }) => {
      const response = await axiosInstance.get(
        `api/reviews?limit=10&offset=${pageParam}`,
      );
      return response.data;
    },
    initialPageParam: 0,
    getNextPageParam: (lastPage, allPages) => {
      const loadedItems = allPages.length * 10;
      return loadedItems < lastPage.count ? loadedItems : undefined;
    },
  });
};
export const pushReview = async ({
  id,
  description,
  rating,
}: {
  id: number;
  description: string;
  rating: number;
}) => {
  await axiosInstance.post(`api/reviews/${id}`, {
    description: description,
    rating: rating,
  });
};
export const useGetLatestReviews = (limit: number) => {
  const getReviews = async (): Promise<ReviewsResponse> => {
    const response = await axiosInstance.get(`api/reviews?limit=${limit}`);
    return response.data;
  };
  const { data, isLoading, isError } = useQuery({
    queryKey: ["CarouselReview"],
    queryFn: getReviews,
    staleTime: 0,
    gcTime: 0,
  });
  return {
    data,
    isLoading,
    isError,
  };
};
export const answerReview = async ({
  id,
  answer,
}: {
  id: number;
  answer: string;
}) => {
  await axiosInstance.post(`api/reviews/answer/${id}`, {
    answer: answer,
  });
};
