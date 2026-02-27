import { postModel } from "@/api/modelsApi";
import { updateStatistic } from "@/api/statisticApi";
import { uploadImage } from "@/api/uploadApi";
import { ModalRecognized } from "@/components";
import { ImageUpload, RecognitionErrorAlert } from "@/layouts";
import { useUser } from "@/lib/useUser";
import type { RecognitionResponse } from "@/shared/types/types";
import { useState } from "react";
import { useTranslation } from "react-i18next";

export const Upload = () => {
  const { user } = useUser();
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const { t } = useTranslation("UploadPage");
  const [result, setResult] = useState<RecognitionResponse | null>(null);
  const [imageURL, setImageURL] = useState("");
  const [error, setError] = useState<string | null>("");
  const handleFileChange = async (file: File) => {
    if (!user?.id) return;
    const imageObject = URL.createObjectURL(file);
    setImageURL(imageObject);
    setIsAnalyzing(true);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("userId", String(user.id));

    const data = await uploadImage(formData);
    if ("mark" in data) {
      setResult({ ...data, confidence: Number(data.confidence.toFixed(2)) });
      const newData = new FormData();
      newData.append("image", file);
      newData.append("mark", data.mark);
      newData.append("model", data.model);
      newData.append("manufactureYear", String(data.manufactureYear));
      newData.append("confidence", data.confidence.toFixed(2));
      newData.append("determinedTime", String(data.determinedTime));
      try {
        const idModel = await postModel(newData, user.id);
        updateStatistic(user.id, idModel);
      } catch (error) {
        console.error("Ошибка ИИ:", error);
      } finally {
        setIsAnalyzing(false);
      }
    } else {
      setError(t("uploadingError"));
      setIsAnalyzing(false);
    }
  };
  return (
    <div className="flex flex-col gap-5 w-full p-10">
      {error && (
        <RecognitionErrorAlert
          desctiption={error}
          setError={setError}
          title={t("errorTitle")}
        />
      )}

      <div>
        <h1 className="text-2xl font-bold text-center">{t("imageAnalizer")}</h1>
        <p className="text-center text-secondary-text">{t("description")}</p>
      </div>

      {isAnalyzing ? (
        <div className="w-full h-57.5 flex flex-col items-center justify-center border-2 border-main rounded-lg bg-secondary-bg/50">
          <div className="w-64 h-2 bg-gray-700 rounded-full overflow-hidden">
            <div className="h-full bg-main animate-progress-bar"></div>
          </div>
          <p className="mt-4 text-main animate-pulse">{t("loading")}</p>
        </div>
      ) : (
        <ImageUpload onUpload={handleFileChange} />
      )}

      {result !== null && (
        <ModalRecognized
          recognizedModel={result}
          imageURL={imageURL}
          setResult={setResult}
        />
      )}
    </div>
  );
};
