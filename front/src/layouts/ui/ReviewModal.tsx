import { Button, Label, Spinner, StarRating, Textarea } from "@/components";
import { Dialog } from "@/components/index";
import { useState } from "react";
import { useTranslation } from "react-i18next";
import { pushReview } from "@/api/reviewsApi";
import { cn } from "@/lib/utils";
import { useUser } from "@/lib/useUser";
import { useMutation, useQueryClient } from "@tanstack/react-query";
export const ReviewModal = () => {
  const { t } = useTranslation("Reviews");
  const [rating, setRating] = useState(0);
  const [comment, setComment] = useState("");
  const [open, setOpen] = useState(false);
  const user = useUser();
  const handlePushCommet = () => {
    if (!comment || rating === 0) return;
    mutate({ id: user.id, description: comment, rating });
  };
  const queryClient = useQueryClient();
  const { mutate, isPending } = useMutation({
    mutationFn: pushReview,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["reviews"] });
      setOpen(false);
      setRating(0);
      setComment("");
    },
    onError: (error) => {
      console.error("Ошибка при публикации:", error);
    },
  });
  return (
    <Dialog.Dialog open={open} onOpenChange={setOpen}>
      <Dialog.DialogTrigger
        className={cn(
          "hover:h-10 transition-all hover:cursor-pointer bg-main hover:bg-hover-button-bg text-primary-foreground p-2 rounded-xl",
          user.id ? "" : "hover:cursor-not-allowed hover:bg-main",
        )}
        disabled={user.id ? false : true}
      >
        {t("writeReview")}
      </Dialog.DialogTrigger>
      <Dialog.DialogContent className="bg-secondary-bg border-0">
        <Dialog.DialogHeader>
          <Dialog.DialogTitle className="flex gap-3 flex-col">
            <span className="text-3xl">{t("publish")}</span>
            <span className="text-secondary-text">
              {t("decriptionPublish")}
            </span>
          </Dialog.DialogTitle>
        </Dialog.DialogHeader>
        <div className="flex flex-col gap-5 text-secondary-text">
          <div className="flex flex-col gap-2">
            <Label className="text-lg">{t("Rate")}</Label>
            <StarRating
              rating={rating}
              onRatingChange={setRating}
              interactive
            />
          </div>
          <div className="flex flex-col gap-2">
            <Label className="text-lg">{t("comment")}</Label>
            <Textarea
              value={comment}
              className="text-white"
              onChange={(e) => setComment(e.target.value)}
              maxLength={200}
              required
            />
          </div>
        </div>

        <Dialog.DialogFooter>
          <Dialog.DialogClose asChild>
            <Button
              onClick={handlePushCommet}
              disabled={isPending || !comment || rating === 0}
            >
              {isPending ? <Spinner className="size-4" /> : t("publish")}
            </Button>
          </Dialog.DialogClose>
        </Dialog.DialogFooter>
      </Dialog.DialogContent>
    </Dialog.Dialog>
  );
};
