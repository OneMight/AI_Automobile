import { User } from "../models/models.js";
import { UserService } from "../services/userService.js";
import cookieParser from "cookie-parser";
const userService = new UserService();
export class UserController {
  async register(req, res) {
    const { email, age, password } = req.body;
    try {
      const user = await User.findOne({
        where: {
          email,
        },
      });
      if (user) {
        return res.status(400).json("User already exists");
      }
      const registeredUser = await User.create({
        email,
        age,
        password,
      });
      return res.statis(200).json(registeredUser);
    } catch (error) {
      return res.status(500).json({ message: error });
    }
  }
  async loginUser(req, res) {
    try {
      const { email, password } = req.body;
      const userData = await userService.login(email, password);
      res.cookie("refreshToken", userData.refreshToken, {
        maxAge: 2 * 60 * 60 * 1000,
        httpOnly: true,
        signed: true,
      });
      return res.status(200).json();
    } catch (error) {
      return res.status(500).json("Login or password is incorrect");
    }
  }
  async logout(req, res) {
    try {
      const cookies = req.cookies;
      const refreshToken = cookieParser.signedCookie(
        cookies.refreshToken,
        process.env.SECRET_KEY,
      );
      await userService.logout(refreshToken);
      res.clearCookie("refreshToken");
      return res.status(200).json({ message: "Successfully logged out" });
    } catch (error) {
      return res
        .status(500)
        .json({ message: `Internal server error ${error}` });
    }
  }
}
