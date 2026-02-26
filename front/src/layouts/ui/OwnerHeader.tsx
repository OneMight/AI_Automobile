import { Logout } from "@/api/userApi";
import { Button } from "@/components";
import { ExitIcon, Logo } from "@/shared/images";
import { ROUTES } from "@/shared/routes/routesPath";
import { useQueryClient } from "@tanstack/react-query";
import { Link, useNavigate } from "@tanstack/react-router";
import { useTranslation } from "react-i18next";

export const OwnerHeader = () => {
  const { t } = useTranslation("Header");
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const handleDirect = () => {
    Logout();
    queryClient.setQueryData(["userToken"], null);
    queryClient.removeQueries({ queryKey: ["userToken"] });
    navigate({ to: ROUTES.HOME, replace: true });
    window.location.href = ROUTES.HOME;
  };
  return (
    <header className="flex flex-row items-center justify-between p-3 px-3 mobile:px-6 w-full max-w-480">
      <Link to={ROUTES.OWNER}>
        <Logo />
      </Link>

      <Button className="text-button-text" onClick={handleDirect}>
        <ExitIcon />
        {t("exit")}
      </Button>
    </header>
  );
};
