import { Benefit } from "@/components/ui/Benefit";
import { HomeMain } from "@/components/ui/HomeMain";
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
      <div className="flex flex-row gap-10">
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
    </div>
  );
}
