import { useTranslation } from "react-i18next";
import { CustomLink, DropDownMenu } from "..";
import { Link } from "@tanstack/react-router";
import { ROUTES } from "@/shared/routes/routesPath";
import { cn } from "@/lib/utils";
import { DashBoardIcon } from "@/shared/images";
import { HistoryIcon, UploadIcon } from "lucide-react";

export const BurgerUser = () => {
  const { t } = useTranslation("Header");
  return (
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
  );
};
