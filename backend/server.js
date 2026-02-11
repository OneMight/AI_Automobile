import dotenv from "dotenv";
import express from "express";
import cors from "cors";
import { router } from "./routes/index.js";
import { sequelize } from "./db.js";
import swaggerJsDoc from "swagger-jsdoc";
import swaggerUi from "swagger-ui-express";
import { swaggerOptions } from "./swagger.config.js";
dotenv.config();
const swaggerDocs = swaggerJsDoc(swaggerOptions);
const PORT = process.env.PORT || 5000;
const app = express();
app.use(
  cors({
    origin: process.env.ORIGIN,
    credentials: true,
  }),
);
// console.log(JSON.stringify(swaggerDocs, null, 2));
app.use(express.json());
app.use("/api", router);
app.use("/api-docs", swaggerUi.serve, swaggerUi.setup(swaggerDocs));
const start = async () => {
  try {
    await sequelize.authenticate();
    await sequelize.sync();

    app.listen(PORT, () => {
      console.log(`Server running at http://localhost:${PORT}`);
    });
  } catch (e) {
    console.error("Ошибка при подключении к базе данных:", e);
  }
};
start();
