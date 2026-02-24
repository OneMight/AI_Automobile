import { Benefit } from "@/components/ui/Benefit";
import { HomeMain } from "@/components/ui/HomeMain";
import { HomeCarousel } from "@/layouts";
import { SpeedIcon } from "@/shared/images";
import { useTranslation } from "react-i18next";
export default function Home() {
  const { t } = useTranslation("HomePage");

  return (
    <div className="flex flex-col w-full items-center justify-center gap-20">
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full h-full overflow-hidden pointer-events-none ">
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-blue-600/10 rounded-full blur-3xl translate-y-1/2" />
        <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20" />
      </div>
      <HomeMain />
      <div className="flex flex-col xl:flex-row gap-10 mb-4">
        <Benefit title="0.15s" description={t("speed")}>
          <SpeedIcon />
        </Benefit>
        <Benefit title="99%" description={t("accuracy")}>
          <SpeedIcon />
        </Benefit>
        <Benefit title="40+" description={t("models")}>
          <SpeedIcon />
        </Benefit>
      </div>
      <div className="max-w-400 w-full bg-secondary-bg p-2 mobile:p-8 flex flex-col items-start gap-10 justify-center">
        <div className="flex flex-col items-start gap-3">
          <h1 className="text-xl tablet:text-3xl font-bold">
            {t("usersReviews")}
          </h1>
          <p className="text-sm tablet:text-xl text-secondary-text">
            {t("letKnow")}
          </p>
        </div>
        <HomeCarousel />
      </div>
    </div>
  );
}
