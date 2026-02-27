import { useNavigate } from "@tanstack/react-router";
import { BurgerAdmin, BurgerUser, Button, DropDownMenu } from "..";
import { MenuIcon } from "lucide-react";
import { ROUTES } from "@/shared/routes/routesPath";
import { ExitIcon } from "@/shared/images";
import { useTranslation } from "react-i18next";
import { Logout } from "@/api/userApi";
import { useQueryClient } from "@tanstack/react-query";
import type { BurgerProps } from "@/shared/types/interfaces";
export const Burger = ({ admin }: BurgerProps) => {
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
        {admin ? <BurgerAdmin /> : <BurgerUser />}

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
