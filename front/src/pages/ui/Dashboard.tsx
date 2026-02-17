import { useGetModels } from "@/api/modelsApi";
import { useGetStatisticsById } from "@/api/statisticApi";
import { StatisticBlock } from "@/components";
import { RecognitionDashboard } from "@/layouts";
import { useUser } from "@/lib/useUser";
import { RecognitionIcon } from "@/shared/images";
import { useTranslation } from "react-i18next";

export default function Dashboard() {
  const { user, isLoading } = useUser();
  const { t } = useTranslation("Dashboard");
  const { statistic, isStatisticLoading } = useGetStatisticsById(user?.id);
  const { models } = useGetModels(user?.id);
  if (isLoading) {
    return <p>Loading</p>;
  }
  return (
    <div className="flex flex-col gap-5 w-full p-10">
      <div className="flex gap-2 flex-col">
        <h1 className="text-2xl font-bold">{t("title")}</h1>
        <p className="text-secondary-text">{t("description")}</p>
      </div>

      {isStatisticLoading ? (
        <></>
      ) : (
        statistic !== undefined && (
          <div className="flex flex-col gap-10 w-full">
            <div className="flex flex-row gap-10 w-full">
              <StatisticBlock
                title={t("recognitions")}
                value={statistic.recognitions}
                icon={<RecognitionIcon />}
              />
              <StatisticBlock
                title={t("confidence")}
                value={`${statistic.avg_percent}%`}
                icon={<RecognitionIcon />}
              />
              <StatisticBlock
                title={t("time")}
                value={statistic.processingTime}
                icon={<RecognitionIcon />}
              />
            </div>
            {statistic?.recognitions !== 0 && (
              <RecognitionDashboard idUser={user?.id} />
            )}
          </div>
        )
      )}
    </div>
  );
}
