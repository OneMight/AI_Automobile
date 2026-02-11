import { Router } from "express";
import { ReviewsController } from "../controllers/ReviewsController.js";
const reviews = new ReviewsController();
const ReviewsRouter = new Router();

export { ReviewsRouter };
