import { UserDto } from "../dto/userDto.js";
import { User } from "../models/models.js";
import { TokenService } from "./tokenService.js";
import bcrypt from "bcrypt";
const tokenService = new TokenService();
export class UserService {
  async login(email, password) {
    const user = await User.findOne({
      where: {
        email,
      },
    });
    if (!user) {
      return { message: "email is incorrect" };
    }
    const isPassword = await bcrypt.compare(password, user.password);
    if (!isPassword) {
      return { message: "Incorrect password" };
    }
    const userDto = new UserDto(user);
    const tokens = tokenService.generateToken({ user });
    await tokenService.saveToken(userDto.id, tokens.refreshToken);
    return {
      ...tokens,
      user: userDto,
    };
  }
  async logout(refreshToken) {
    const token = await tokenService.removeToken(refreshToken);
    return token;
  }
}
