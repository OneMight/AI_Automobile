export const carPath = {
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
};
