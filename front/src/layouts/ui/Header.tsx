import { Button } from "@/components";
import { ExitIcon, Logo } from "@/shared/images";
import { ROUTES } from "@/shared/routes/routesPath";
import { Link, useNavigate } from "@tanstack/react-router";
import { useTranslation } from "react-i18next";

export const Header = () => {
  const navigate = useNavigate();
  const { t } = useTranslation("Header");
  const handleDirect = (link: string) => {
    navigate({ to: link });
  };
  return (
    <header className=" flex flex-row items-center justify-between p-3 px-6 w-full max-w-480">
      <Link to={ROUTES.HOME}>
        <Logo />
      </Link>
      <div className="flex flex-row items-center gap-10">
        <Link
          to={ROUTES.LOGIN}
          className="text-sm hover:text-slate-200 text-secondary-text"
        >
          {t("reviews")}
        </Link>

        <Button
          className="text-button-text"
          onClick={() => handleDirect(ROUTES.LOGIN)}
        >
          <ExitIcon />
          {t("entire")}
        </Button>
      </div>
    </header>
  );
};
