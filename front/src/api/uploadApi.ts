import type { RecognitionResponse } from "@/shared/types/types";
import { AiInstance } from ".";

export const uploadImage = async (file: FormData) => {
  const response = await AiInstance.post<RecognitionResponse>("/predict", file);
  return response.data;
};
