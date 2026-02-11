export const userPaths = {
  "/api/user/register": {
    post: {
      summary: "Регистрация пользователя",
      tags: ["Auth"],
      requestBody: {
        required: true,
        content: {
          "application/json": {
            schema: {
              type: "object",
              properties: {
                email: { type: "string", example: "test@mail.ru" },
                age: { type: "integer", example: 25 },
                password: { type: "string", example: "123456" },
              },
              required: ["email", "password"],
            },
          },
        },
      },
      responses: {
        200: { description: "Пользователь успешно создан" },
        400: { description: "Пользователь уже существует" },
      },
    },
  },
  "/api/user/login": {
    post: {
      summary: "Авторизация пользователя",
      tags: ["Auth"],
      requestBody: {
        required: true,
        content: {
          "application/json": {
            schema: {
              type: "object",
              properties: {
                email: { type: "string", example: "test@mail.ru" },
                password: { type: "string", example: "123456" },
              },
            },
          },
        },
      },
      responses: {
        200: {
          description: "Успешный вход, refreshToken установлен в Cookie",
          headers: {
            "Set-Cookie": {
              description: "Содержит refreshToken",
              schema: {
                type: "string",
                example: "refreshToken=abcde...; HttpOnly",
              },
            },
          },
        },
        500: { description: "Логин или пароль неверны" },
      },
    },
  },
  "/api/user/logout": {
    post: {
      summary: "Выход из системы",
      tags: ["Auth"],
      responses: {
        200: { description: "Успешный выход" },
        500: { description: "Ошибка сервера" },
      },
    },
  },
};
