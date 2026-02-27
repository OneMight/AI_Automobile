import { AlertDialog, Input, Label, Spinner } from "@/components";
import { useState } from "react";
import { useTranslation } from "react-i18next";
import { postFeedback } from "@/api/feedbackApi";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useUser } from "@/lib/useUser";
import { cn } from "@/lib/utils";
export interface SubmitionErrorFormProps {
  setIsOpenError: (value: boolean) => void;
  setError: (value: null) => void;
  setIsSuccessFeedback: (value: boolean) => void;
}
export const SubmitionErrorForm = ({
  setIsOpenError,
  setError,
  setIsSuccessFeedback,
}: SubmitionErrorFormProps) => {
  const [mark, setMark] = useState("");
  const [model, setModel] = useState("");
  const [manufactureYear, setManufactureYear] = useState("");
  const handleClose = () => {
    setError(null);
    setIsOpenError(false);
  };
  const { user } = useUser();
  const queryClient = useQueryClient();
  const { mutate, isPending, isSuccess } = useMutation({
    mutationFn: postFeedback,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["reviews"] });
    },
    onError: (error) => {
      console.error("Ошибка при публикации:", error);
    },
  });
  const handleSubmit = () => {
    if (!model || !mark || !manufactureYear) return;
    mutate({
      id: user?.id,
      mark: mark,
      model: model,
      manufactureYear: manufactureYear,
    });
  };
  if (isSuccess) {
    setIsSuccessFeedback(isSuccess);
    setError(null);
    setIsOpenError(false);
  }
  const { t } = useTranslation("UploadPage");
  return (
    <AlertDialog.AlertDialog>
      <AlertDialog.AlertDialogTrigger className="bg-main p-1 rounded-lg">
        {t("openSubmit")}
      </AlertDialog.AlertDialogTrigger>
      <AlertDialog.AlertDialogContent>
        <AlertDialog.AlertDialogHeader>
          <AlertDialog.AlertDialogTitle>
            {t("enhanceModel")}
          </AlertDialog.AlertDialogTitle>
          <div className="w-full py-3 flex flex-col gap-3">
            <div className="w-full flex flex-col items-start gap-2">
              <Label>{t("mark")}</Label>
              <Input
                className="pl-3"
                value={mark}
                onChange={(e) => setMark(e.target.value)}
              />
            </div>
            <div className="w-full flex flex-col items-start gap-2">
              <Label>{t("model")}</Label>
              <Input
                className="pl-3"
                value={model}
                onChange={(e) => setModel(e.target.value)}
              />
            </div>
            <div className="w-full flex flex-col items-start gap-2">
              <Label>{t("manufactureYear")}</Label>
              <Input
                className="pl-3"
                value={manufactureYear}
                onChange={(e) => setManufactureYear(e.target.value)}
              />
            </div>
          </div>
        </AlertDialog.AlertDialogHeader>
        <AlertDialog.AlertDialogFooter>
          <AlertDialog.AlertDialogAction
            onClick={handleClose}
            className="bg-main/50"
            disabled={isPending}
          >
            {t("close")}
          </AlertDialog.AlertDialogAction>
          <AlertDialog.AlertDialogAction
            onClick={handleSubmit}
            className={cn("bg-main-app", isPending && "bg-main-app/50")}
          >
            {isPending ? <Spinner className="size-4" /> : t("submit")}
          </AlertDialog.AlertDialogAction>
        </AlertDialog.AlertDialogFooter>
      </AlertDialog.AlertDialogContent>
    </AlertDialog.AlertDialog>
  );
};
