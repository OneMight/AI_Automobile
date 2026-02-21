import { useNavigate } from "@tanstack/react-router";
import { Button } from "./Button";
import { ArrowRight } from "@/shared/images";
import { ROUTES } from "@/shared/routes/routesPath";
import { useTranslation } from "react-i18next";
export const HomeMain = () => {
  const navigate = useNavigate();
  const { t } = useTranslation("HomePage");
  const handleDirect = (link: string) => {
    navigate({ to: link });
  };
  return (
    <div className="w-full flex items-center justify-center flex-col  gap-15">
      <h1 className="mobile:text-7xl text-3xl font-bold">
        Neuro <span className="text-main">Scan</span>
      </h1>
      <p className="text-center text-xl mobile:text-3xl text-secondary-text max-w-200 px-2">
        {t("mainText")}
      </p>
      <div className="flex flex-col mobile:flex-row gap-5 items-center">
        <Button
          onClick={() => handleDirect(ROUTES.REGISTER)}
          className="p-6 flex items-center w-55"
        >
          {t("startFree")} <ArrowRight />
        </Button>
        <Button
          onClick={() => handleDirect(ROUTES.LOGIN)}
          className="p-6 min-w-55"
          variant="secondary"
        >
          {t("login")}
        </Button>
      </div>
    </div>
  );
};
