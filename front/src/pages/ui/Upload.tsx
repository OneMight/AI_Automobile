import { uploadImage } from "@/api/uploadApi";
import { ImageUpload } from "@/layouts";
import { useUser } from "@/lib/useUser";
import type { RecognitionResponse } from "@/shared/types/types";
import { useState } from "react";

export const Upload = () => {
  const { user } = useUser();
  const [isAnalyzing, setIsAnalyzing] = useState(false); // Состояние загрузки нейронки
  const [result, setResult] = useState<RecognitionResponse | null>(null);

  const handleFileChange = async (file: File) => {
    if (!user?.id) return;

    setIsAnalyzing(true); // Включаем лоадер

    const formData = new FormData();
    formData.append("file", file);
    formData.append("userId", String(user.id));

    try {
      const data = await uploadImage(formData);
      setResult(data);
    } catch (error) {
      console.error("Ошибка ИИ:", error);
    } finally {
      setIsAnalyzing(false);
    }
  };
  console.log(result);
  return (
    <div className="flex flex-col gap-5 w-full p-10">
      <h1 className="text-2xl font-bold">Распознавание авто</h1>

      {isAnalyzing ? (
        <div className="w-full h-[230px] flex flex-col items-center justify-center border-2 border-main rounded-lg bg-secondary-bg/50">
          <div className="w-64 h-2 bg-gray-700 rounded-full overflow-hidden">
            <div className="h-full bg-main animate-progress-bar"></div>
          </div>
          <p className="mt-4 text-main animate-pulse">
            Нейросеть анализирует фото...
          </p>
        </div>
      ) : (
        <ImageUpload onUpload={handleFileChange} />
      )}

      {result && (
        <div className="mt-5 p-4 bg-green-500/10 border border-green-500 rounded">
          Результат: {result.mark} {result.model} ({result.yearManufacture})
        </div>
      )}
    </div>
  );
};
