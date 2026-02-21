import { Logout } from "@/api/userApi";
import { Button, CustomLink } from "@/components";
import { cn } from "@/lib/utils";
import { useQueryClient } from "@tanstack/react-query";
import {
  DashBoardIcon,
  ExitIcon,
  HistoryIcon,
  Logo,
  UploadIcon,
} from "@/shared/images";
import { ROUTES } from "@/shared/routes/routesPath";
import { Link, useNavigate } from "@tanstack/react-router";
import { useTranslation } from "react-i18next";

export const LoginedHeader = () => {
  const queryClient = useQueryClient();
  const { t } = useTranslation("Header");
  const navigate = useNavigate();
  const handleLogout = () => {
    Logout();
    queryClient.setQueryData(["userToken"], null);
    queryClient.removeQueries({ queryKey: ["userToken"] });
    window.location.reload();
    navigate({ to: ROUTES.HOME, replace: true });
  };
  return (
    <header className=" flex flex-row items-center justify-between p-3 px-6 w-full max-w-480">
      <Link to={ROUTES.DASHBOARD}>
        <Logo />
      </Link>
      <nav className=" flex flex-row gap-1 items-center">
        <Link to={ROUTES.DASHBOARD} className="group">
          {({ isActive }) => (
            <CustomLink
              route={ROUTES.DASHBOARD}
              isActive={isActive}
              icon={
                <DashBoardIcon
                  className={cn(
                    "group-hover:text-main duration-200",
                    isActive ? "group-hover:text-main/50" : "",
                  )}
                />
              }
            >
              {t("dashboard")}
            </CustomLink>
          )}
        </Link>

        <Link to={ROUTES.UPLOAD} className="group">
          {({ isActive }) => (
            <CustomLink
              route={ROUTES.UPLOAD}
              isActive={isActive}
              icon={
                <UploadIcon
                  className={cn(
                    "group-hover:text-main duration-200",
                    isActive ? "group-hover:text-main/50" : "",
                  )}
                />
              }
            >
              {t("upload")}
            </CustomLink>
          )}
        </Link>
        <Link to={ROUTES.HISTORY} className="group">
          {({ isActive }) => (
            <CustomLink
              route={ROUTES.HISTORY}
              isActive={isActive}
              icon={
                <HistoryIcon
                  className={cn(
                    "group-hover:text-main duration-200",
                    isActive ? "group-hover:text-main/50" : "",
                  )}
                />
              }
            >
              {t("history")}
            </CustomLink>
          )}
        </Link>
      </nav>
      <div className="flex flex-row gap-5 items-center">
        <Button
          className="bg-transparent hover:bg-transparent p-0 group"
          onClick={handleLogout}
        >
          <ExitIcon className="text-gray-500 size-5 group-hover:text-red-500 duration-200" />
        </Button>
      </div>
    </header>
  );
};
