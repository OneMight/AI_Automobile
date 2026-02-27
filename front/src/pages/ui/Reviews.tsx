import { Review, Spinner } from "@/components";
import { useTranslation } from "react-i18next";
import { useInView } from "react-intersection-observer";

import { ReviewModal } from "@/layouts";
import { useGetReviews } from "@/api/reviewsApi";
import { useEffect } from "react";
import type { ReviewProps } from "@/shared/types/interfaces";
export const Reviews = ({ admin }: ReviewProps) => {
  const { t } = useTranslation("Reviews");
  const { data, fetchNextPage, hasNextPage, isFetchingNextPage, status } =
    useGetReviews();
  const { ref, inView } = useInView();
  useEffect(() => {
    if (inView && hasNextPage && !isFetchingNextPage) {
      fetchNextPage();
    }
  }, [inView, hasNextPage, isFetchingNextPage, fetchNextPage]);
  const reviews = data?.pages.flatMap((page) => page.rows) ?? [];
  return (
    <div className="flex items-start justify-center w-full flex-col gap-10 mobile:px-6 px-3">
      <div className="flex flex-col mobile:flex-row w-full items-start mobile:items-center gap-3 justify-between">
        <div className="flex flex-col items-start gap-3">
          <h1 className="text-3xl">{t("title")}</h1>
          <p className="text-secondary-text">{t("description")}</p>
        </div>
        {!admin && <ReviewModal />}
      </div>
      <div className="flex flex-col gap-5 items-center justify-center w-full mb-10">
        {status === "pending" ? (
          <Spinner className="size-10" />
        ) : (
          <div className="w-full grid desktop:grid-cols-3 tablet:grid-cols-2  gap-4">
            {reviews.map((review) => (
              <Review key={review.id} review={review} admin={admin} />
            ))}
          </div>
        )}
        <div ref={ref} className="h-10 w-full flex justify-center items-center">
          {isFetchingNextPage ? (
            <Spinner />
          ) : hasNextPage ? null : (
            <p className="text-tag-text">{t("noMore")}</p>
          )}
        </div>
      </div>
    </div>
  );
};
