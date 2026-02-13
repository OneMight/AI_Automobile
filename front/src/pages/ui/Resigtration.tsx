import { RegistrationForm } from "@/layouts";
import { SingleLogo } from "@/shared/images";
import { ROUTES } from "@/shared/routes/routesPath";
import { Link } from "@tanstack/react-router";
import { useTranslation } from "react-i18next";

export const Registration = () => {
  const { t } = useTranslation("Registration");
  return (
    <div className="flex flex-col gap-10 p-5 bg-secondary-bg rounded-2xl max-w-110 w-full mx-3">
      <div className="flex flex-col items-center justify-center gap-5">
        <SingleLogo />
        <h1 className="font-bold text-2xl">{t("title")}</h1>
        <p className="text-sm text-secondary-text">{t("description")}</p>
      </div>
      <RegistrationForm />
      <p>
        {t("haveAccount")}{" "}
        <Link className="text-main hover:text-main/50" to={ROUTES.LOGIN}>
          {t("login")}
        </Link>
      </p>
    </div>
  );
};
