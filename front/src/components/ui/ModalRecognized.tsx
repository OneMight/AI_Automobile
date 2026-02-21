import { useTranslation } from "react-i18next";
import { Button, Dialog } from "..";
import type { ModalProps } from "@/shared/types/types";
import { Link } from "@tanstack/react-router";
import { ROUTES } from "@/shared/routes/routesPath";
import { useEffect, useState } from "react";

export const ModalRecognized = ({
  recognizedModel,
  imageURL,
  setResult,
}: ModalProps) => {
  const { t } = useTranslation("UploadPage");
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    setIsOpen(true);
  }, []);
  const handleClose = () => {
    setResult(null);
    setIsOpen(false);
  };
  return (
    <Dialog.Dialog
      open={isOpen}
      onOpenChange={handleClose}
      defaultOpen={isOpen}
    >
      <Dialog.DialogContent className="bg-secondary-bg border-0">
        <Dialog.DialogHeader className="text-xl">
          <Dialog.DialogTitle>
            {" "}
            <span className="text-main">{t("congratulations")}</span>{" "}
          </Dialog.DialogTitle>
          {recognizedModel.mark} {recognizedModel.model}{" "}
          {recognizedModel.manufactureYear} {t("withConfidence")}{" "}
          {recognizedModel.confidence * 100}%
        </Dialog.DialogHeader>
        <Dialog.DialogDescription>
          {" "}
          <img
            src={imageURL}
            alt={`${recognizedModel.mark}_${recognizedModel.model}_${recognizedModel.manufactureYear}`}
          />
        </Dialog.DialogDescription>

        <Dialog.DialogFooter>
          <Dialog.DialogClose>
            <Link to={ROUTES.HISTORY}>{t("historyLink")}</Link>
          </Dialog.DialogClose>
          <Dialog.DialogClose asChild>
            <Button onClick={handleClose}>{t("close")}</Button>
          </Dialog.DialogClose>
        </Dialog.DialogFooter>
      </Dialog.DialogContent>
    </Dialog.Dialog>
  );
};
