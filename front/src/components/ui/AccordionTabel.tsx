import type { AccordionProps } from "@/shared/types/interfaces";
import { Accordion } from "..";
import type { ModelTable } from "@/shared/types/types";
import { useTranslation } from "react-i18next";
import { cn } from "@/lib/utils";

export const AccordionTable = ({
  children,
  content,
}: AccordionProps<ModelTable>) => {
  console.log(content);
  const { t } = useTranslation("History");
  return (
    <Accordion.Accordion type="single" collapsible className="w-full">
      <Accordion.AccordionItem value="item-1">
        <Accordion.AccordionTrigger className="hover:cursor-pointer p-0">
          {children}
        </Accordion.AccordionTrigger>
        <Accordion.AccordionContent className="flex flex-row items-start gap-10 justify-start p-4">
          <img
            className="max-w-100"
            src={content.modelImage}
            alt={`${content.mark}_${content.model}`}
          />
          <div className="flex flex-col text-xl justify-between">
            <p>
              {t("mark")}: {content.mark}
            </p>
            <p>
              {t("model")}: {content.model}
            </p>
            <p>
              {t("manufactureYear")}: {content.manufactureYear}
            </p>
          </div>
          <div className="flex flex-col gap-3">
            <h2 className="font-bold">{t("analysisMetrics")}</h2>
            <div className="flex flex-row gap-3">
              <div className="bg-secondary-bg p-5 min-w-40">
                <p className="text-secondary-text">{t("confidence")}</p>
                <p className="text-center text-xl"> {content.confidence}%</p>
              </div>
              <div className="bg-secondary-bg p-5">
                <p className="text-secondary-text">{t("processingTime")}</p>
                <p
                  className={cn(
                    "text-center text-xl",
                    content.determinedTime > 5
                      ? "text-bad"
                      : content.determinedTime < 5 && content.determinedTime > 3
                        ? "text-good"
                        : content.determinedTime < 3
                          ? "text-great"
                          : "",
                  )}
                >
                  {content.determinedTime}s
                </p>
              </div>
            </div>
          </div>
        </Accordion.AccordionContent>
      </Accordion.AccordionItem>
    </Accordion.Accordion>
  );
};
