import { Button, CustomAlert, Fields, Input } from "@/components";
import { useTranslation } from "react-i18next";
import { Lock, Mail, ArrowRight } from "lucide-react";
import { useForm } from "@tanstack/react-form";
import * as z from "zod";
import { useState } from "react";
import { useNavigate } from "@tanstack/react-router";
import { ROUTES } from "@/shared/routes/routesPath";
import { Register } from "@/api/userApi";
import { useQueryClient } from "@tanstack/react-query";
export const RegistrationForm = () => {
  const navigate = useNavigate();
  const { t } = useTranslation("Registration");
  const [isError, setIsError] = useState<string>("");
  const queryClient = useQueryClient();
  const [focusedField, setFocusedField] = useState<string | null>(null);
  const formSchema = z
    .object({
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
      repeatPassword: z.string(),
      age: z.int().gte(10).lte(120),
    })
    .superRefine(({ password, repeatPassword }, ctx) => {
      if (password !== repeatPassword) {
        ctx.addIssue({
          code: "custom",
          message: t("passwordMatch"),
          path: ["repeatPassword"],
        });
      }
    });
  const form = useForm({
    defaultValues: {
      email: "",
      password: "",
      repeatPassword: "",
      age: 10,
    },
    validators: {
      onChange: formSchema,
    },
    onSubmit: async ({ value }) => {
      const errorMessage = await Register(value);
      if (errorMessage!.includes("User")) {
        setIsError(t("errorMessage"));
      } else {
        await queryClient.invalidateQueries({ queryKey: ["userToken"] });
        navigate({ to: ROUTES.DASHBOARD });
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
      {isError && <CustomAlert error={isError} setIsError={setIsError} />}
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
                    className={
                      showError ? "border-red-500 focus:border-red-500" : ""
                    }
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
        <form.Field
          name="repeatPassword"
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
                    : t("passwordRepeatLabel")}
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
                    placeholder={t("passwordRepeatPlaceholder")}
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
          <Button type="submit" className="w-full text-black">
            {t("submit")} <ArrowRight />
          </Button>
        )}
      />
    </form>
  );
};
