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
  "/api/statistics/recognitions": {
    get: {
      summary: "Статистика распознаваний за 7 дней",
      tags: ["Statistics"],
      responses: {
        200: {
          description: "Массив объектов с датой и количеством",
          example: [{ day: "2023-10-01", count: 15 }],
        },
      },
    },
  },
  "/api/statistics/ages": {
    get: {
      summary: "Распределение пользователей по возрасту",
      tags: ["Statistics"],
      responses: {
        200: {
          description: "Количество пользователей в группах 10-18, 18-30, 30+",
          example: [
            { category: "10-18", count: 50 },
            { category: "18-30", count: 120 },
          ],
        },
      },
    },
  },
};
