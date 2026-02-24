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
  "/api/determined/post/{id}": {
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
};
