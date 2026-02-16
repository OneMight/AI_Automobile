export type User = {
  email: string;
  password: string;
  age?: number;
  role: "user" | "admin";
};
export type UserLogin = Omit<User, "age" | "role">;
