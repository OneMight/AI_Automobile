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
  model: string;
  mark: string;
  manufactureYear: string;
  confidence: number;
  determinedTime: number;
  modelImage: string;
};

export type SimilarModel = {
  confidence: number;
  mark: string;
  model: string;
};
export type RecognitionResponse = {
  confidence: number;
  mark: string;
  model: string;
  manufactureYear: string;
  determinedTime: number;
  similarModels: SimilarModel[];
};

export type ModalProps = {
  recognizedModel: Omit<
    RecognitionResponse,
    "similarModels" | "recognizedTime"
  >;

  imageURL: string;
};
export type PostDeterminedModel = Omit<
  DeterminedModel,
  "id" | "similarModels"
> & {
  id: number;
};
