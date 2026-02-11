import { UserDto } from "../dto/userDto.js";
import { TokenService } from "./tokenService.js";
const tokenService = new TokenService();
export class UserService {
  async login(email, password) {
    const user = await Employee.findOne({
      where: {
        email,
      },
    });

    if (!user) {
      throw new Error({ message: "Name is incorrect" });
    }
    const isPassword = await bcrypt.compare(password, user.password);
    if (!isPassword) {
      throw new Error({ message: "Incorrect password" });
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
