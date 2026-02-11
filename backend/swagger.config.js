import dotenv from "dotenv";
dotenv.config();
export const swaggerOptions = {
  definition: {
    openapi: "3.0.0",
    info: {
      title: "AI Automobile API",
      version: "1.0.0",
    },
    paths: {
      "/api/car": {
        get: {
          summary: "Список всех автомобилей",
          tags: ["Cars"],
          responses: {
            200: {
              description: "Успешный ответ",
              content: {
                "application/json": {
                  schema: {
                    type: "array",
                    items: { $ref: "#/components/schemas/Car" },
                  },
                },
              },
            },
          },
        },
      },
      "/api/car/create": {
        post: {
          summary: "Добавить новый автомобиль",
          tags: ["Cars"],
          requestBody: {
            required: true,
            content: {
              "application/json": {
                schema: {
                  type: "object",
                  properties: {
                    mark: { type: "string", example: "Tesla" },
                    model: { type: "string", example: "Model 3" },
                    manufactureYear: { type: "string", example: "2023" },
                  },
                  required: ["mark", "model", "manufactureYear"],
                },
              },
            },
          },
          responses: {
            201: {
              description: "Машина успешно создана",
              content: {
                "application/json": {
                  schema: { $ref: "#/components/schemas/Car" },
                },
              },
            },
            500: {
              description: "Ошибка сервера",
            },
          },
        },
      },
    },
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
