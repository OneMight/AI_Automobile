import { Router } from "express";
const CarsRouter = new Router();
import { CarsController } from "../controllers/CarsController.js";
const carsController = new CarsController();

CarsRouter.get("/", carsController.get);
CarsRouter.post("/create", carsController.create);
export { CarsRouter };
