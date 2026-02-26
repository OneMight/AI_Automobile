import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import type { PropsAge } from "@/shared/types/interfaces";
import { useTranslation } from "react-i18next";

export const AgesCharts = ({ data }: PropsAge) => {
  const { t } = useTranslation("Owner");
  return (
    <div className={"max-w-100 min-w-75 h-75 w-full"}>
      <h3 className="text-center font-bold">{t("userAge")}</h3>
      <ResponsiveContainer className={"max-w-100"}>
        <BarChart
          data={data}
          margin={{ top: 20, right: 30, left: 0, bottom: 0 }}
        >
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="category" axisLine={false} tickLine={false} />
          <YAxis allowDecimals={false} axisLine={false} tickLine={false} />
          <Tooltip
            cursor={{ fill: "transparent" }}
            contentStyle={{
              borderRadius: "10px",
              border: "none",
              boxShadow: "0px 4px 10px rgba(0,0,0,0.1)",
              color: "#000000",
            }}
          />
          <Bar
            dataKey="count"
            fill="#06b6d4"
            radius={[4, 4, 0, 0]}
            barSize={40}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};
