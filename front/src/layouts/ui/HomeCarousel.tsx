import * as React from "react";
import { Carousel, Review, Spinner } from "@/components";
import Autoplay from "embla-carousel-autoplay";
import { useGetLatestReviews } from "@/api/reviewsApi";
import { useTranslation } from "react-i18next";

export function HomeCarousel() {
  const plugin = React.useRef(
    Autoplay({ delay: 5000, stopOnInteraction: false }),
  );
  const { data, isLoading } = useGetLatestReviews(10);
  const { t } = useTranslation("HomePage");
  if (isLoading) {
    return <Spinner className="size-10" />;
  }
  return data?.rows.length == 0 ? (
    <div>
      <p className="text-lg">{t("noReviews")}</p>
    </div>
  ) : (
    <Carousel.Carousel
      plugins={[plugin.current]}
      className="w-full"
      opts={{
        watchDrag: false,
        loop: true,
      }}
    >
      <Carousel.CarouselContent>
        {data?.rows.map((review) => (
          <Carousel.CarouselItem
            key={review.id}
            className="basis-full  flex justify-center desktop:basis-1/3 tablet:basis-1/2"
          >
            <Review review={review} className="h-full" />
          </Carousel.CarouselItem>
        ))}
      </Carousel.CarouselContent>
    </Carousel.Carousel>
  );
}
