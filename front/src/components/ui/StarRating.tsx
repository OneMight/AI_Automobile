import { cn } from "@/lib/utils";
import type { StarRatingProps } from "@/shared/types/interfaces";
import { Star } from "lucide-react";
import { useState } from "react";

export const StarRating = ({
  rating,
  interactive = false,
  onRatingChange,
  size = 20,
  className,
}: StarRatingProps) => {
  const [hoverRating, setHoverRating] = useState<number | null>(null);
  const displayRating = hoverRating !== null ? hoverRating : rating;
  return (
    <div
      className={cn(
        "flex gap-1 bg-main-app p-3 w-35 rounded-xl border border-button-stroke",
        className,
      )}
    >
      {[1, 2, 3, 4, 5].map((star) => (
        <button
          key={star}
          type="button"
          disabled={!interactive}
          onClick={() => interactive && onRatingChange?.(star)}
          onMouseEnter={() => interactive && setHoverRating(star)}
          onMouseLeave={() => interactive && setHoverRating(null)}
          className={`${interactive ? "cursor-pointer hover:scale-110 transition-transform" : "cursor-default"} focus:outline-none`}
        >
          <Star
            size={size}
            className={`${star <= displayRating ? "fill-main text-main" : "fill-transparent text-main"} transition-colors duration-200`}
          />
        </button>
      ))}
    </div>
  );
};
