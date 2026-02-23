import { Button, Label, StarRating, Textarea } from "@/components";
import { Dialog } from "@/components/index";
import { useState } from "react";
import { useTranslation } from "react-i18next";

export const ReviewModal = () => {
  const { t } = useTranslation("Reviews");
  const [rating, setRating] = useState(0);
  return (
    <Dialog.Dialog>
      <Dialog.DialogTrigger>
        <Button className="hover:h-10 transition-all">
          {t("writeReview")}
        </Button>
      </Dialog.DialogTrigger>
      <Dialog.DialogContent className="bg-secondary-bg border-0">
        <Dialog.DialogHeader>
          <Dialog.DialogTitle className="flex gap-3 flex-col">
            <h1 className="text-3xl">{t("publish")}</h1>
            <p className="text-secondary-text">{t("decriptionPublish")}</p>
          </Dialog.DialogTitle>
        </Dialog.DialogHeader>
        <Dialog.DialogDescription className="flex flex-col gap-5">
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
            <Textarea></Textarea>
          </div>
        </Dialog.DialogDescription>

        <Dialog.DialogFooter>
          <Dialog.DialogClose asChild>
            <Button>{t("publish")}</Button>
          </Dialog.DialogClose>
        </Dialog.DialogFooter>
      </Dialog.DialogContent>
    </Dialog.Dialog>
  );
};
