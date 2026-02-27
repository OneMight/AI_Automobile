import { Avatar } from "radix-ui";
import { StarRating } from "./StarRating";
import type { ReviewsBlockProps } from "@/shared/types/interfaces";
import { User } from "lucide-react";
import { convertYear } from "@/lib/converDate";
import { cn } from "@/lib/utils";
import { Button } from "./Button";
import { Textarea } from "./TextArea";
import { useTranslation } from "react-i18next";
import { useState, type ChangeEvent } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Spinner } from "./Spinner";
import { answerReview } from "@/api/reviewsApi";

export const Review = ({ review, className, admin }: ReviewsBlockProps) => {
  const { t } = useTranslation("Reviews");
  const [answer, setAnswer] = useState("");
  const handleSetAnswer = (e: ChangeEvent<HTMLTextAreaElement>) => {
    setAnswer(e.target.value);
  };
  console.log(answer);
  const queryClient = useQueryClient();
  const { mutate, isPending } = useMutation({
    mutationFn: answerReview,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["reviews"] });
      setAnswer("");
    },
    onError: (error) => {
      console.error("Ошибка при публикации:", error);
    },
  });
  const handlePushComment = () => {
    if (!answer) return;
    mutate({ id: review.id, answer: answer });
  };
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
      {review.answer && (
        <p className="pl-2 break-all">
          {t("answer")}: {review.answer}
        </p>
      )}
      {admin && !review.answer && (
        <div className="flex flex-col items-end justify-center w-full gap-3">
          <Textarea
            className="min-h-20"
            onChange={handleSetAnswer}
            maxLength={100}
            required
          />
          <Button onClick={handlePushComment} disabled={isPending || !answer}>
            {isPending ? <Spinner className="size-4" /> : t("pushAnswer")}
          </Button>
        </div>
      )}
    </div>
  );
};
