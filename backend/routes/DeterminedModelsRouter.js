import { Router } from "express";
import { DeterminedModelsController } from "../controllers/DeterminedModelsController.js";
const determined = new DeterminedModelsController();
const DeterminedModelsRouter = new Router();

export { DeterminedModelsRouter };
