import { Router } from "express";
import { UserController } from "../controllers/UserController.js";
const user = new UserController();
const UserRouter = new Router();
UserRouter.post("/register", user.register);
UserRouter.post("/login", user.loginUser);
UserRouter.post("/logout", user.logout);
export { UserRouter };
