export const reviewsPaths = {
  "/api/reviews": {
    get: {
      summary: "Получить список отзывов",
      tags: ["Reviews"],
      parameters: [
        {
          name: "limit",
          in: "query",
          schema: { type: "integer", default: 10 },
        },
        {
          name: "offset",
          in: "query",
          schema: { type: "integer", default: 0 },
        },
      ],
      responses: {
        200: {
          description: "Список отзывов",
          content: {
            "application/json": {
              schema: {
                type: "object",
                properties: {
                  count: { type: "integer", example: 2 },
                  rows: { type: "array", items: { type: "object" } },
                },
              },
            },
          },
        },
      },
    },
  },
  "/api/reviews/{id}": {
    post: {
      summary: "Оставить отзыв",
      tags: ["Reviews"],
      parameters: [
        {
          name: "id",
          in: "path",
          required: true,
          schema: { type: "integer" },
          description: "ID пользователя",
        },
      ],
      requestBody: {
        required: true,
        content: {
          "application/json": {
            schema: {
              type: "object",
              properties: {
                description: {
                  type: "string",
                  example: "Отличное приложение!",
                },
                rating: { type: "integer", example: 5 },
              },
            },
          },
        },
      },
      responses: {
        200: { description: "Отзыв успешно добавлен" },
      },
    },
  },
};
