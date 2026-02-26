import type { PropsLine } from "@/shared/types/interfaces";
import { useTranslation } from "react-i18next";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

export const RecognitionsCharts = ({ data }: PropsLine) => {
  const { t } = useTranslation("Owner");
  return (
    <div className="max-w-100 min-w-75 w-full h-75">
      <h3 className="text-center font-bold">{t("activity")}</h3>
      <ResponsiveContainer className={"w-full h-full"}>
        <LineChart
          data={data}
          margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
        >
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#eee" />

          <XAxis
            dataKey="day"
            tick={{ fontSize: 12 }}
            axisLine={false}
            tickLine={false}
          />

          <YAxis
            allowDecimals={false}
            tick={{ fontSize: 12 }}
            axisLine={false}
            tickLine={false}
          />

          <Tooltip
            contentStyle={{
              borderRadius: "8px",
              border: "none",
              boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
            }}
          />

          <Line
            type="monotone"
            dataKey="count"
            stroke="#8884d8"
            strokeWidth={3}
            dot={{ r: 4, fill: "#8884d8", strokeWidth: 2, stroke: "#fff" }}
            activeDot={{ r: 6, strokeWidth: 0 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};
