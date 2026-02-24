export const statisticsPaths = {
  "/api/statistics/{id}": {
    get: {
      summary: "Получить статистику пользователя по ID",
      tags: ["Statistics"],
      parameters: [
        { name: "id", in: "path", required: true, schema: { type: "integer" } },
      ],
      responses: {
        200: { description: "Данные статистики" },
      },
    },
    put: {
      summary: "Обновить статистику пользователя",
      tags: ["Statistics"],
      parameters: [
        { name: "id", in: "path", required: true, schema: { type: "integer" } },
      ],
      requestBody: {
        content: {
          "application/json": {
            schema: {
              type: "object",
              properties: {
                avg_percent: { type: "number" },
                recognitions: { type: "integer" },
              },
            },
          },
        },
      },
      responses: {
        200: { description: "Статистика обновлена" },
      },
    },
  },
};
