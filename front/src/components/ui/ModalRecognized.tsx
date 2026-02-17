import { useTranslation } from "react-i18next";
import { Dialog } from "..";
import type { ModalProps } from "@/shared/types/types";
import { Link } from "@tanstack/react-router";
import { ROUTES } from "@/shared/routes/routesPath";

export const ModalRecognized = ({ recognizedModel, imageURL }: ModalProps) => {
  const { t } = useTranslation("UploadPage");
  return (
    <Dialog.Dialog>
      <Dialog.DialogTrigger>open</Dialog.DialogTrigger>
      <Dialog.DialogContent className="bg-secondary-bg border-0">
        <Dialog.DialogHeader className="text-xl">
          <span className="text-main">{t("congratulations")}</span>{" "}
          {recognizedModel.mark} {recognizedModel.model}{" "}
          {recognizedModel.manufactureYear} {t("withConfidence")}{" "}
          {recognizedModel.confidence * 100}%
        </Dialog.DialogHeader>

        <img
          src={imageURL}
          alt={`${recognizedModel.mark}_${recognizedModel.model}_${recognizedModel.manufactureYear}`}
        />
        <Dialog.DialogFooter>
          <Dialog.DialogClose>
            <Link to={ROUTES.HISTORY}>{t("historyLink")}</Link>
          </Dialog.DialogClose>
          <Dialog.DialogClose>{t("close")}</Dialog.DialogClose>
        </Dialog.DialogFooter>
      </Dialog.DialogContent>
    </Dialog.Dialog>
  );
};
