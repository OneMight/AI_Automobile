import { Logout } from "@/api/userApi";
import { Burger, Button, CustomLink } from "@/components";
import { useUser } from "@/lib/useUser";
import { cn, isAdmin } from "@/lib/utils";
import { DashBoardIcon, ExitIcon, Logo, UploadIcon } from "@/shared/images";
import { ROUTES } from "@/shared/routes/routesPath";
import { useQueryClient } from "@tanstack/react-query";
import { Link, useNavigate } from "@tanstack/react-router";
import { useTranslation } from "react-i18next";

export const AdminHeader = () => {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const { user } = useUser();
  const userIsAdmin = isAdmin(user?.role);
  const { t } = useTranslation("Header");
  const handleLogout = () => {
    Logout();
    queryClient.setQueryData(["userToken"], null);
    queryClient.removeQueries({ queryKey: ["userToken"] });
    navigate({ to: ROUTES.HOME, replace: true });
    window.location.href = ROUTES.HOME;
  };
  return (
    <header className=" flex flex-row items-center justify-between p-3 px-3 mobile:px-6 w-full max-w-480">
      <Link to={ROUTES.ADMIN}>
        <Logo />
      </Link>
      <nav className="hidden header:flex flex-row gap-1 items-center">
        <Link to={ROUTES.ADMIN} className="group">
          {({ isActive }) => (
            <CustomLink
              route={ROUTES.ADMIN}
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
              {t("reviews")}
            </CustomLink>
          )}
        </Link>
        <Link to={ROUTES.SWAGGER} className="group">
          {({ isActive }) => (
            <CustomLink
              route={ROUTES.SWAGGER}
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
              {t("db")}
            </CustomLink>
          )}
        </Link>
      </nav>
      <div className="block header:hidden">
        <Burger admin={userIsAdmin} />
      </div>
      <div className="hidden header:flex flex-row gap-5 items-center">
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
