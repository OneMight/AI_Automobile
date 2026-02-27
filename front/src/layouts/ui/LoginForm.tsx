import { Button, CustomAlert, Fields, Input, Spinner } from "@/components";
import { useTranslation } from "react-i18next";
import { Lock, Mail, ArrowRight } from "lucide-react";
import { useForm } from "@tanstack/react-form";
import * as z from "zod";
import { useState } from "react";
import { LoginUser } from "@/api/userApi";
import { useNavigate } from "@tanstack/react-router";
import { ROUTES } from "@/shared/routes/routesPath";
import { useQueryClient } from "@tanstack/react-query";
import { cn } from "@/lib/utils";
export const LoginForm = () => {
  const navigate = useNavigate();
  const { t } = useTranslation("Login");
  const queryClient = useQueryClient();
  const [isLoading, setIsLoading] = useState(false);
  const [isError, setIsError] = useState<string>("");
  const [focusedField, setFocusedField] = useState<string | null>(null);
  const formSchema = z.object({
    email: z
      .string()
      .email(t("invalidEmail"))
      .refine(
        (val) => {
          const domain = val.split("@")[1];
          const allowedDomains = [
            "gmail.com",
            "mail.ru",
            "yandex.ru",
            "outlook.com",
            "icloud.com",
          ];
          return allowedDomains.includes(domain?.toLowerCase());
        },
        { message: t("invalidEmail") },
      ),
    password: z.string().min(8, t("smallPassword")),
  });
  const form = useForm({
    defaultValues: {
      email: "",
      password: "",
    },
    validators: {
      onChange: formSchema,
    },
    onSubmit: async ({ value }) => {
      setIsLoading(true);
      const errorMessage = await LoginUser(value);
      if (errorMessage?.message?.includes("email")) {
        setIsLoading(false);
        setIsError(t("invalidEmail"));
      } else if (errorMessage!.message?.includes!("password")) {
        setIsLoading(false);
        setIsError(t("invalidPassword"));
      } else if (typeof errorMessage?.refreshToken === "string") {
        await queryClient.invalidateQueries({ queryKey: ["userToken"] });
        localStorage.setItem("refreshToken", errorMessage.refreshToken);
        const roleOfUser =
          errorMessage.role === "owner"
            ? ROUTES.OWNER
            : errorMessage.role === "admin"
              ? ROUTES.ADMIN
              : ROUTES.DASHBOARD;
        setIsLoading(false);
        navigate({ to: roleOfUser });
        window.location.href = roleOfUser;
      }
    },
  });

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        e.stopPropagation();
        form.handleSubmit();
      }}
      className="w-full flex flex-col items-center gap-5"
    >
      {isError.length !== 0 && (
        <CustomAlert error={isError} setIsError={setIsError} />
      )}
      <Fields.FieldGroup className="gap-5">
        <form.Field
          name="email"
          children={(field) => {
            const hasErrors = field.state.meta.errors.length > 0;
            const hasSubmitions =
              field.form.state.submissionAttempts > 0 ||
              field.state.meta.isTouched;
            const showError =
              hasErrors && hasSubmitions && focusedField !== field.name;
            return (
              <Fields.Field>
                <Fields.FieldLabel
                  htmlFor={field.name}
                  className={showError ? "text-red-500" : "text-secondary-text"}
                >
                  {showError ? t("invalidEmail") : t("emailLabel")}
                </Fields.FieldLabel>
                <div className="relative group">
                  <Input
                    id={field.name}
                    name={field.name}
                    value={field.state.value}
                    onBlur={() => {
                      setFocusedField(null);
                      field.handleBlur();
                    }}
                    onFocus={() => setFocusedField(field.name)}
                    onChange={(e) => {
                      field.handleChange(e.target.value);
                      field.setMeta((prev) => ({ ...prev, isTouched: false }));
                    }}
                    placeholder={t("emailPlaceholder")}
                    className={cn(
                      showError ? "border-red-500 focus:border-red-500" : "",
                    )}
                  >
                    <Mail
                      className={`absolute left-2 top-2 transition-colors pointer-events-none ${
                        showError
                          ? "text-red-500"
                          : "group-focus-within:text-main"
                      }`}
                    />
                  </Input>
                </div>
              </Fields.Field>
            );
          }}
        />

        <form.Field
          name="password"
          children={(field) => {
            const hasErrors = field.state.meta.errors.length > 0;
            const hasSubmitions =
              field.form.state.submissionAttempts > 0 ||
              field.state.meta.isTouched;
            const showError =
              hasErrors && hasSubmitions && focusedField !== field.name;

            return (
              <Fields.Field>
                <Fields.FieldLabel
                  htmlFor={field.name}
                  className={showError ? "text-red-500" : "text-secondary-text"}
                >
                  {showError
                    ? field.state.meta.errors
                        .map((err) =>
                          typeof err === "object" ? err.message : err,
                        )
                        .join(", ")
                    : t("passwordLabel")}
                </Fields.FieldLabel>
                <div className="relative group">
                  <Input
                    id={field.name}
                    name={field.name}
                    type="password"
                    onFocus={() => setFocusedField(field.name)}
                    value={field.state.value}
                    onBlur={() => {
                      setFocusedField(null);
                      field.handleBlur();
                    }}
                    onChange={(e) => {
                      field.handleChange(e.target.value);
                      field.setMeta((prev) => ({ ...prev, isTouched: false }));
                    }}
                    placeholder={t("passwordPlaceholder")}
                    className={
                      showError ? "border-red-500 focus:border-red-500" : ""
                    }
                  >
                    <Lock
                      className={`absolute left-2 top-2 transition-colors pointer-events-none ${
                        showError
                          ? "text-red-500"
                          : "group-focus-within:text-main"
                      }`}
                    />
                  </Input>
                </div>
              </Fields.Field>
            );
          }}
        />
      </Fields.FieldGroup>

      <form.Subscribe
        selector={(state) => [state.canSubmit, state.isSubmitting]}
        children={() => (
          <Button
            type="submit"
            disabled={isLoading}
            className={cn("w-full text-black", isLoading && "bg-main/50")}
          >
            {isLoading ? <Spinner className="size-4" /> : t("submit")}{" "}
            <ArrowRight />
          </Button>
        )}
      />
    </form>
  );
};
