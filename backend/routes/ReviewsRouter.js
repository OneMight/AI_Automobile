import { Router } from "express";
import { ReviewsController } from "../controllers/ReviewsController.js";
const reviews = new ReviewsController();
const ReviewsRouter = new Router();
ReviewsRouter.get("/", reviews.getReviews);
ReviewsRouter.post("/:id", reviews.postReview);
ReviewsRouter.post("/answer/:id", reviews.answerReview);
export { ReviewsRouter };
