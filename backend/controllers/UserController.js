import { Statistics, User } from "../models/models.js";
import { UserService } from "../services/userService.js";
import { TokenService } from "../services/tokenService.js";
import bcrypt from "bcrypt";
const tokenService = new TokenService();
import cookieParser from "cookie-parser";
const userService = new UserService();
export class UserController {
  async register(req, res) {
    const { email, age, password, role = "user" } = req.body;
    try {
      const user = await User.findOne({
        where: {
          email,
        },
      });
      if (user) {
        return res.status(400).json({ message: "User already exists" });
      }
      const hashedPassword = await bcrypt.hash(password, 3);
      const createdUser = await User.create({
        role,
        email,
        age,
        password: hashedPassword,
      });
      await Statistics.create({
        idUser: createdUser.id,
        avg_percent: 0,
        recognitions: 0,
      });
      const userData = await userService.login(email, password);
      res.cookie("refreshToken", userData.refreshToken, {
        maxAge: 2 * 60 * 60 * 1000,
        httpOnly: true,
        signed: true,
        secure: true,
        sameSite: "none",
      });
      return res.status(200).json();
    } catch (error) {
      return res.status(500).json({ message: error.message });
    }
  }
  async loginUser(req, res) {
    try {
      const { email, password } = req.body;
      const userData = await userService.login(email, password);
      if (userData?.message) {
        return res.status(404).json({ message: userData.message });
      }
      res.cookie("refreshToken", userData.refreshToken, {
        maxAge: 2 * 60 * 60 * 1000,
        httpOnly: true,
        signed: true,
        secure: true,
        sameSite: "none",
      });
      return res.status(200).json();
    } catch (error) {
      return res.status(500).json(error);
    }
  }
  async logout(req, res) {
    try {
      const rawCookie =
        req.signedCookies?.refreshToken || req.cookies?.refreshToken;
      const refreshToken = cookieParser.signedCookie(
        rawCookie,
        process.env.SECRET_KEY,
      );
      await userService.logout(refreshToken);
      res.clearCookie("refreshToken", {
        httpOnly: true,
        secure: true,
        sameSite: "none",
      });
      return res.status(200).json({ message: "Successfully logged out" });
    } catch (error) {
      return res
        .status(500)
        .json({ message: `Internal server error ${error}` });
    }
  }
  async getUserByToken(req, res) {
    try {
      const rawCookie =
        req.signedCookies?.refreshToken || req.cookies?.refreshToken;
      const refreshToken = cookieParser.signedCookie(
        rawCookie,
        process.env.SECRET_KEY,
      );
      const user = await tokenService.getDataByToken(refreshToken);
      return res.status(200).json(user);
    } catch (error) {
      return res.status(500).json({ message: `${error}` });
    }
  }
}
