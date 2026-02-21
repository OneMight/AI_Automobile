import { useGetStatisticsById } from "@/api/statisticApi";
import { useUser } from "@/lib/useUser";
import { useTranslation } from "react-i18next";
import { columns } from "@/components";
import { DataTable } from "@/layouts";
import { useGetModels } from "@/api/modelsApi";
export const History = () => {
  const user = useUser();
  const { models } = useGetModels(user.id);
  const dataModel = models?.map((model) => ({
    id: model.id,
    confidence: model.confidence * 100,
    createdAt: model.createdAt,
    mark: model.Car.mark,
    model: model.Car.model,
    manufactureYear: model.Car.manufactureYear,
    determinedTime: model.determinedTime,
    modelImage: model.modelImage,
  }));
  const { statistic, isStatisticLoading } = useGetStatisticsById(user.id);
  const { t } = useTranslation("History");
  if (isStatisticLoading) {
    <p>Loading</p>;
  }
  return (
    <div className="w-full flex flex-col items-center justify-center max-w-7xl px-6 gap-10">
      <div className="flex flex-col mobile:flex-row w-full gap-4 justify-between items-start mobile:items-center">
        <div className="flex flex-col items-start justify-center gap-2">
          <h1 className="font-bold text-xl mobile:text-3xl">{t("title")}</h1>
          <p className="text-secondary-text text-sm mobile:text-xl">
            {t("description")}
          </p>
        </div>
        <div className="flex flex-row w-60 items-center justify-between bg-main/50 rounded-2xl p-2">
          <p>{t("totalRecords")}</p>
          <p>{statistic?.recognitions}</p>
        </div>
      </div>
      {typeof dataModel !== "undefined" && (
        <DataTable columns={columns} data={dataModel} />
      )}
    </div>
  );
};
