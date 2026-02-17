import type { StatisticBlockProps } from "@/shared/types/interfaces";

export const StatisticBlock = ({ title, value, icon }: StatisticBlockProps) => {
  return (
    <div className="flex flex-col min-h-40 min-w-55 gap-5 p-6 items-start rounded-xl bg-secondary-bg w-full">
      <div className="flex flex-row w-full justify-between items-center">
        <p className="text-secondary-text">{title}</p>
        {icon}
      </div>
      <p className="font-bold text-2xl">{value}</p>
    </div>
  );
};
