import { AlertDialog } from "@/components";
import type { RecognitionErrorProps } from "@/shared/types/interfaces";
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

export const RecognitionErrorAlert = ({
  title,
  desctiption,
  setError,
}: RecognitionErrorProps) => {
  const [isOpen, setIsOpen] = useState(false);
  const { t } = useTranslation("UploadPage");
  useEffect(() => {
    setIsOpen(true);
  }, []);
  const handleClose = () => {
    setError(null);
    setIsOpen(false);
  };
  return (
    <AlertDialog.AlertDialog
      defaultOpen={isOpen}
      onOpenChange={handleClose}
      open={isOpen}
    >
      <AlertDialog.AlertDialogContent>
        <AlertDialog.AlertDialogHeader>
          <AlertDialog.AlertDialogTitle>{title}</AlertDialog.AlertDialogTitle>
          <AlertDialog.AlertDialogDescription>
            {desctiption}
          </AlertDialog.AlertDialogDescription>
        </AlertDialog.AlertDialogHeader>
        <AlertDialog.AlertDialogFooter>
          <AlertDialog.AlertDialogAction
            onClick={handleClose}
            className="bg-main-app/50"
          >
            {t("ok")}
          </AlertDialog.AlertDialogAction>
        </AlertDialog.AlertDialogFooter>
      </AlertDialog.AlertDialogContent>
    </AlertDialog.AlertDialog>
  );
};
