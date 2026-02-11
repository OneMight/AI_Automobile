import jwt from "jsonwebtoken";
import { Tokens } from "../models/models.js";

export class TokenService {
  generateToken(payload) {
    const accesToken = jwt.sign(payload, process.env.SECRET_KEY, {
      expiresIn: "2h",
    });
    const refreshToken = jwt.sign(payload, process.env.SECRET_REFRESH_KEY, {
      expiresIn: "2h",
    });
    return {
      accesToken,
      refreshToken,
    };
  }
  async saveToken(id, refreshToken) {
    const datatoken = await Tokens.findOne({
      where: {
        idUser: id,
      },
    });
    if (datatoken) {
      datatoken.refreshToken = refreshToken;
      return datatoken.save();
    }
    const token = await Tokens.create({ idUser: id, refreshToken });
    return token;
  }
  async removeToken(refreshToken) {
    const tokenData = await Tokens.destroy({
      where: {
        refreshToken,
      },
    });
    return tokenData;
  }
  async getDataByToken(refreshToken) {
    const tokenData = await Tokens.findOne({
      where: {
        refreshToken,
      },
    });
    if (!tokenData) {
      return new Error({ message: `User not authorized` });
    }
    const decoded = jwt.verify(refreshToken, process.env.SECRET_REFRESH_KEY);
    return decoded;
  }
}
