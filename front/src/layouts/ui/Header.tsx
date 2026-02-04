import { Button } from "@/components";
import { ExitIcon, Logo } from "@/shared/images";
import { ROUTES } from "@/shared/routes/routesPath";
import { Link, useNavigate } from "@tanstack/react-router";

export const Header = () => {
  const navigate = useNavigate();
  const handleDirect = (link: string) => {
    navigate({ to: link });
  };
  return (
    <header className="bg-header-bg flex flex-row items-center justify-between p-3 px-6 w-full max-w-480">
      <Link to={ROUTES.HOME}>
        <Logo />
      </Link>
      <div className="flex flex-row items-center gap-10">
        <Link to={ROUTES.LOGIN} className="text-sm text-secondary-text">
          Отзывы
        </Link>

        <Button
          className=" text-button-text"
          onClick={() => handleDirect(ROUTES.LOGIN)}
        >
          <ExitIcon />
          Войти
        </Button>
      </div>
    </header>
  );
};
