import { Avatar } from "radix-ui";
import { StarRating } from "./StarRating";
import type { ReviewsBlockProps } from "@/shared/types/interfaces";
import { User } from "lucide-react";
import { convertYear } from "@/lib/converDate";
import { cn } from "@/lib/utils";

export const Review = ({ review, className }: ReviewsBlockProps) => {
  return (
    <div
      className={cn(
        "min-w-75 w-full flex flex-col items-start gap-3 p-5 bg-secondary-bg rounded-2xl",
        className,
      )}
    >
      <div className="flex flex-row gap-3 items-center">
        <Avatar.Avatar className="bg-main-app p-3 rounded-full">
          <Avatar.AvatarFallback>
            <User />
          </Avatar.AvatarFallback>
        </Avatar.Avatar>
        <div>
          <h2 className="text-lg">{review.email}</h2>
          <p className="text-sm text-secondary-text">
            {convertYear(review.createdAt)}
          </p>
        </div>
      </div>
      <StarRating
        rating={review.rating}
        className="bg-transparent border-transparent pl-0"
      />
      <p className="break-all">{review.description}</p>
    </div>
  );
};
