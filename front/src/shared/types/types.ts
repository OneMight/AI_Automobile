export type User = {
  id: number;
  email: string;
  password: string;
  age?: number;
  role: "user" | "admin";
};
export type RegisterUser = Omit<User, "role" | "id">;
export type UserLogin = Omit<User, "age" | "role" | "id">;
export type Statistic = {
  avg_percent: number;
  recognitions: number;
  processingTime: string;
};
export type DeterminedModel = {
  id: number;
  model_name: string;
  confidence: number;
  recognitionTime: number;
  model_image: string;
  file_name: string;
};
