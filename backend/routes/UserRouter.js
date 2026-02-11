import { Router } from "express";
import { UserController } from "../controllers/UserController.js";
const user = new UserController();
const UserRouter = new Router();

export { UserRouter };
