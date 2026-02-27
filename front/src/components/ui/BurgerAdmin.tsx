import { Link } from "@tanstack/react-router";
import { CustomLink, DropDownMenu } from "..";
import { ROUTES } from "@/shared/routes/routesPath";
import { DashBoardIcon } from "@/shared/images";
import { cn } from "@/lib/utils";
import { UploadIcon } from "lucide-react";
import { useTranslation } from "react-i18next";

export const BurgerAdmin = () => {
  const { t } = useTranslation("Header");
  return (
    <DropDownMenu.DropdownMenuGroup className="flex-col">
      <DropDownMenu.DropdownMenuItem>
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
      </DropDownMenu.DropdownMenuItem>
      <DropDownMenu.DropdownMenuItem>
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
      </DropDownMenu.DropdownMenuItem>
    </DropDownMenu.DropdownMenuGroup>
  );
};
