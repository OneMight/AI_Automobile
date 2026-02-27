export type User = {
  id: number;
  email: string;
  password: string;
  age?: number;
  role: Role;
};
export type Role = "user" | "admin" | "owner";
export type RegisterUser = Omit<User, "role" | "id">;
export type UserLogin = Omit<User, "age" | "role" | "id">;
export type Statistic = {
  avg_percent: number;
  recognitions: number;
  processingTime: string;
};
export type DeterminedModel = {
  id: number;
  Car: Car;
  confidence: number;
  determinedTime: number;
  modelImage: string;
  createdAt: Date | string;
};
export type ModelTable = Omit<DeterminedModel, "Car"> & Omit<Car, "idCar">;
export type OwnerTableTypes = Omit<DeterminedModel, "Car"> &
  Omit<Car, "idCar"> & {
    email: string;
  };

export type OwnerResponse = {
  count: number;
  rows: OwnerTableTypes[];
};
export type Car = Omit<SimilarModel, "confidence"> & {
  manufactureYear: string;
};
export type SimilarModel = {
  idCar: number;
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
  setResult: (value: null) => void;
  imageURL: string;
};
export type PostDeterminedModel = Omit<
  DeterminedModel,
  "id" | "similarModels"
> & {
  id: number;
};
export type PostModel = {
  id: number;
};
export type Error = {
  detail: string;
};
export type ReviewsResponse = {
  count: number;
  rows: Reviews[];
};
export type Reviews = {
  id: number;
  rating: number;
  description: string;
  email: string;
  role: Role;
  answer: string;
  createdAt: Date;
};
