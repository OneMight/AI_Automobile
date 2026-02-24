import { useInfiniteQuery } from "@tanstack/react-query";
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
