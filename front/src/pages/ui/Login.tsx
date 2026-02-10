import { LoginForm } from "@/layouts/ui/LoginForm";
import { SingleLogo } from "@/shared/images";
import { useTranslation } from "react-i18next";

export default function Login() {
  const { t } = useTranslation("Login");
  return (
    <div className="flex flex-col gap-10 p-10 bg-secondary-bg rounded-2xl max-w-110 w-full mx-3">
      <div className="flex flex-col items-center justify-center gap-5">
        <SingleLogo />
        <h1 className="font-bold text-2xl">{t("title")}</h1>
        <p className="text-sm text-secondary-text">{t("description")}</p>
      </div>
      <LoginForm />
    </div>
  );
}
