import { Router } from "express";
import { FeedBackController } from "../controllers/FeedBackController.js";
const feedbackController = new FeedBackController();
const FeedBackRouter = new Router();
FeedBackRouter.get("/", feedbackController.getFeedbacks);
FeedBackRouter.post("/:id", feedbackController.postFeedback);
export { FeedBackRouter };
