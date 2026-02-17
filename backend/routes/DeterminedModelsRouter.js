import { Router } from "express";
import { DeterminedModelsController } from "../controllers/DeterminedModelsController.js";
import { CarsController } from "../controllers/CarsController.js";
const carController = new CarsController();
const determined = new DeterminedModelsController();
const DeterminedModelsRouter = new Router();
DeterminedModelsRouter.post("/:id", determined.getDeterminedModelsByUserId);
DeterminedModelsRouter.post(
  "/post/:id",
  carController.CarMiddleware,
  determined.postDeterminedModel,
);
export { DeterminedModelsRouter };
