import dotenv from "dotenv";
import express from "express";
import cors from "cors";
import { router } from "./routes/index.js";
import { sequelize } from "./db.js";
dotenv.config();

const PORT = process.env.PORT || 5000;
const app = express();
app.use(
  cors({
    origin: process.env.ORIGIN,
    credentials: true,
  }),
);
app.use(express.json());
app.use("/api", router);
const start = async () => {
  try {
    await sequelize.authenticate();
    console.log("Соединение с базой данных успешно!");
    await sequelize.sync();

    app.listen(PORT, () => {
      console.log(`Server running at http://localhost:${PORT}`);
    });
  } catch (e) {
    console.error("Ошибка при подключении к базе данных:", e);
  }
};
start();
