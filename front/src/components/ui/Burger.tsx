import { Link, useNavigate } from "@tanstack/react-router";
import { Button, CustomLink, DropDownMenu } from "..";
import { HistoryIcon, MenuIcon, UploadIcon } from "lucide-react";
import { ROUTES } from "@/shared/routes/routesPath";
import { DashBoardIcon, ExitIcon } from "@/shared/images";
import { cn } from "@/lib/utils";
import { useTranslation } from "react-i18next";
import { Logout } from "@/api/userApi";
import { useQueryClient } from "@tanstack/react-query";
export const Burger = () => {
  const { t } = useTranslation("Header");
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const handleLogout = () => {
    Logout();
    queryClient.setQueryData(["userToken"], null);
    queryClient.removeQueries({ queryKey: ["userToken"] });
    navigate({ to: ROUTES.HOME, replace: true });
    window.location.href = ROUTES.HOME;
  };
  return (
    <DropDownMenu.DropdownMenu>
      <DropDownMenu.DropdownMenuTrigger asChild>
        <Button>
          <MenuIcon />
          Menu
        </Button>
      </DropDownMenu.DropdownMenuTrigger>
      <DropDownMenu.DropdownMenuContent className="bg-secondary-bg right-2">
        <DropDownMenu.DropdownMenuGroup>
          <DropDownMenu.DropdownMenuLabel>
            <Link to={ROUTES.DASHBOARD}></Link>
          </DropDownMenu.DropdownMenuLabel>
          <DropDownMenu.DropdownMenuItem>
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
          </DropDownMenu.DropdownMenuItem>
          <DropDownMenu.DropdownMenuItem>
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
          </DropDownMenu.DropdownMenuItem>
          <DropDownMenu.DropdownMenuItem>
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
          </DropDownMenu.DropdownMenuItem>
          <DropDownMenu.DropdownMenuItem>
            <Link to={ROUTES.REVIEWS} className="group">
              {({ isActive }) => (
                <CustomLink route={ROUTES.REVIEWS} isActive={isActive}>
                  {t("reviews")}
                </CustomLink>
              )}
            </Link>
          </DropDownMenu.DropdownMenuItem>
        </DropDownMenu.DropdownMenuGroup>
        <DropDownMenu.DropdownMenuGroup>
          <DropDownMenu.DropdownMenuItem>
            <Button
              className="bg-transparent hover:bg-transparent group has-[>svg]:px-1"
              onClick={handleLogout}
            >
              <ExitIcon className="text-gray-500 size-5 group-hover:text-red-500 duration-200" />
              {t("exit")}
            </Button>
          </DropDownMenu.DropdownMenuItem>
        </DropDownMenu.DropdownMenuGroup>
      </DropDownMenu.DropdownMenuContent>
    </DropDownMenu.DropdownMenu>
  );
};
