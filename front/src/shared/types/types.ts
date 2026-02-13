export type User = {
  email: string;
  password: string;
  age?: number;
};
export type UserLogin = Omit<User, "age">;
