import { Router } from "express";
import {
  DeterminedModelsController,
  upload,
} from "../controllers/DeterminedModelsController.js";
import { CarsController } from "../controllers/CarsController.js";
const carController = new CarsController();
const determined = new DeterminedModelsController();
const DeterminedModelsRouter = new Router();
DeterminedModelsRouter.post("/:id", determined.getDeterminedModelsByUserId);
DeterminedModelsRouter.post(
  "/post/:id",
  upload.single("image"),
  carController.CarMiddleware,
  determined.postDeterminedModel,
);
export { DeterminedModelsRouter };
