import { Button } from "@/components";
import { useTranslation } from "react-i18next";
import { FilterIcon, Calendar, StarIcon } from "lucide-react";
import { ReviewModal } from "@/layouts";
export const Reviews = () => {
  const { t } = useTranslation("Reviews");
  return (
    <div className="flex items-start justify-center w-full flex-col gap-10 mobile:px-6 px-3">
      <div className="flex flex-row w-full items-center justify-between">
        <div className="flex flex-col items-start  gap-3">
          <h1 className="text-3xl">{t("title")}</h1>
          <p className="text-secondary-text">{t("description")}</p>
        </div>
        <ReviewModal />
      </div>
      <div className="flex flex-col gap-5 items-start justify-center w-full ">
        <div className="bg-secondary-text/10 flex flex-row justify-between items-center w-full p-5 rounded-2xl">
          <div className="flex flex-row gap-3 items-center ">
            <FilterIcon />
            {t("filters")}
          </div>
          <div className="flex flex-row items-center gap-5">
            <Button variant={"secondary"} className="group hover:bg-white/10">
              <Calendar /> {t("byDate")}
            </Button>
            <Button variant={"secondary"} className="group hover:bg-white/10">
              <StarIcon /> {t("byRate")}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};
