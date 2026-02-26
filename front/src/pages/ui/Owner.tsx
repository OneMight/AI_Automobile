import { AgesCharts, OwnerTable, RecognitionsCharts } from "@/layouts";
import { useTranslation } from "react-i18next";
import { ownerColumns, Spinner } from "@/components";
import { useGetAllModels } from "@/api/modelsApi";
import {
  useGetAgesStatistic,
  useGetRecognitionStatistic,
} from "@/api/statisticApi";
export const Owner = () => {
  const { data, isLoading } = useGetAllModels();
  const { t } = useTranslation("Owner");
  const { ages, isAgesLoading } = useGetAgesStatistic();
  const { recogniontion, isRecognitionLoading } = useGetRecognitionStatistic();
  const ownerTable = data?.rows?.map((model) => ({
    id: model.id,
    email: model.email,
    confidence: model.confidence * 100,
    createdAt: model.createdAt,
    mark: model.mark,
    model: model.model,
    manufactureYear: model.manufactureYear,
    determinedTime: model.determinedTime,
    modelImage: model.modelImage,
  }));
  return (
    <div className="flex flex-col gap-10 w-full mobile:p-5 p-2">
      <div className="flex flex-col gap-2 items-start">
        <h1 className="tablet:text-3xl font-bold text-xl">{t("title")}</h1>
        <p className="tablet:text-xl text-lg">{t("description")}</p>
      </div>
      <div className="flex flex-col gap-10 tablet:flex-row tablet:justify-around items-center w-full">
        {isAgesLoading ? (
          <Spinner className="size-10" />
        ) : (
          typeof ages !== "undefined" && <AgesCharts data={ages} />
        )}
        {isRecognitionLoading ? (
          <Spinner className="size-10" />
        ) : (
          typeof recogniontion !== "undefined" && (
            <RecognitionsCharts data={recogniontion} />
          )
        )}
      </div>

      {isLoading ? (
        <Spinner className="size-10" />
      ) : (
        typeof ownerTable !== "undefined" && (
          <OwnerTable columns={ownerColumns} data={ownerTable} />
        )
      )}
    </div>
  );
};
