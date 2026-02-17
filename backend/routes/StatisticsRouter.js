import { Router } from "express";
import { StatisticController } from "../controllers/StatisticController.js";
const statistic = new StatisticController();
const StatisticsRouter = new Router();
StatisticsRouter.get("/:id", statistic.getStatisticsById);
StatisticsRouter.put("/:id", statistic.updateStatistic);
export { StatisticsRouter };
