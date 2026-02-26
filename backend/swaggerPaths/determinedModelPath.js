export const determinedModelsPaths = {
  "/api/determinedModel/{id}": {
    post: {
      summary: "Получить историю распознаваний пользователя",
      tags: ["DeterminedModels"],
      parameters: [
        { name: "id", in: "path", required: true, schema: { type: "integer" } },
      ],
      responses: {
        200: { description: "Список распознанных моделей" },
      },
    },
  },
  "/api/determinedModel/post/{id}": {
    post: {
      summary: "Загрузить фото и распознать автомобиль",
      tags: ["DeterminedModels"],
      parameters: [
        { name: "id", in: "path", required: true, schema: { type: "integer" } },
      ],
      requestBody: {
        content: {
          "multipart/form-data": {
            schema: {
              type: "object",
              properties: {
                image: {
                  type: "string",
                  format: "binary",
                  description: "Файл изображения",
                },
                mark: { type: "string" },
                model: { type: "string" },
                manufactureYear: { type: "string" },
                confidence: { type: "number" },
                determinedTime: { type: "string" },
              },
            },
          },
        },
      },
      responses: {
        200: { description: "ID созданной записи" },
      },
    },
  },
  "/api/determinedModel": {
    get: {
      summary: "Получить список всех распознанных моделей",
      description:
        "Возвращает список записей из DeterminedModels с информацией о пользователе и автомобиле.",
      tags: ["DeterminedModels"],
      responses: {
        200: {
          description: "Успешный возврат списка моделей",
          content: {
            "application/json": {
              schema: {
                type: "object",
                properties: {
                  count: {
                    type: "integer",
                    example: 1,
                  },
                  rows: {
                    type: "array",
                    items: {
                      type: "object",
                      properties: {
                        id: { type: "integer" },
                        userId: { type: "integer" },
                        carId: { type: "integer" },
                        email: { type: "string", example: "user@example.com" },
                        determinedTime: { type: "string", format: "date-time" },
                        confidence: { type: "number", example: 0.98 },
                        modelImage: {
                          type: "string",
                          description: "Путь к изображению",
                        },
                        User: {
                          type: "object",
                          properties: {
                            id: { type: "integer" },
                            email: { type: "string" },
                          },
                        },
                        Car: {
                          type: "object",
                          properties: {
                            id: { type: "integer" },
                            mark: { type: "string" },
                            model: { type: "string" },
                          },
                        },
                      },
                    },
                  },
                },
              },
            },
          },
        },
        500: {
          description: "Ошибка сервера",
          content: {
            "application/json": {
              schema: {
                type: "object",
                properties: {
                  message: { type: "string" },
                },
              },
            },
          },
        },
      },
    },
  },
};
