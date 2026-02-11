import dotenv from "dotenv";
import { carPath, userPaths } from "./swaggerPaths/index.js";
dotenv.config();
export const swaggerOptions = {
  definition: {
    openapi: "3.0.0",
    info: {
      title: "AI Automobile API",
      version: "1.0.0",
    },
    paths: { ...userPaths, ...carPath },
    components: {
      schemas: {
        Car: {
          type: "object",
          properties: {
            id: { type: "integer", example: 1 },
            mark: { type: "string", example: "Toyota" },
            model: { type: "string", example: "Camry" },
            manufactureYear: { type: "string", example: "2022" },
          },
        },
      },
    },
  },
  apis: [],
};
