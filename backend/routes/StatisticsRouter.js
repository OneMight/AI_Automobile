import { Router } from "express";
import { StatisticController } from "../controllers/StatisticController.js";
const statistic = new StatisticController();
const StatisticsRouter = new Router();

export { StatisticsRouter };
