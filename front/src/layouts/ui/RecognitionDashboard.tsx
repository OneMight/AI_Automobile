import { useGetModels } from "@/api/modelsApi";
import { Button, ModelBlock } from "@/components";
import { ArrowRight } from "@/shared/images";
import { ROUTES } from "@/shared/routes/routesPath";
import { Link, useNavigate } from "@tanstack/react-router";
import { useTranslation } from "react-i18next";

export const RecognitionDashboard = ({
  idUser,
}: {
  idUser: number | undefined;
}) => {
  const { t } = useTranslation("Dashboard");
  const navigate = useNavigate();
  const { models, isLoading } = useGetModels(idUser, 10);
  if (isLoading) {
    return <p>Loading</p>;
  }
  const handleDirectToHistory = () => {
    navigate({ to: ROUTES.HISTORY });
  };
  return typeof models !== "undefined" ? (
    <div className="w-full flex flex-col bg-secondary-bg p-4 gap-6">
      <div className="flex flex-row gap-3 w-full items-center justify-between">
        <h1 className="font-bold">{t("tableTitle")}</h1>
        <Link
          to={ROUTES.HISTORY}
          className="flex flex-row items-center gap-1 group text-main hover:text-main/50 transition-colors"
        >
          {t("historyLink")}{" "}
          <ArrowRight className="text-main group-hover:text-main/50" />
        </Link>
      </div>
      <div className="flex flex-col gap-4 items-start justify-center w-full">
        {models.map((model) => (
          <ModelBlock key={model.id} model={model} />
        ))}
      </div>
      {models.length >= 10 && (
        <div className="w-full flex flex-col gap-2">
          <p className="text-center">...</p>
          <Button className="w-full" onClick={handleDirectToHistory}>
            {t("goToHistory")}
          </Button>
        </div>
      )}
    </div>
  ) : (
    <p className="font-bold text-2xl text-center">
      {t("noRecognitions")}{" "}
      <Link
        to={ROUTES.UPLOAD}
        className="text-main hover:text-main/50 transition-colors"
      >
        {t("goToUpload")}
      </Link>
    </p>
  );
};
