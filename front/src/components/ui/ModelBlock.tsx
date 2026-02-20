import type { ModelBlockProps } from "@/shared/types/interfaces";

export const ModelBlock = ({ model }: ModelBlockProps) => {
  return (
    <div className="flex flex-row w-full justify-between items-center p-3 rounded-2xl bg-secondary-text/10">
      <div className="flex flex-row gap-3 items-center justify-start">
        <img
          src={`${import.meta.env.VITE_API_URL}/${model.modelImage}`}
          className="w-20"
          alt=""
        />
        <p className="text-white">
          {model.Car?.mark} {model.Car?.model} {model.Car?.manufactureYear}
        </p>
      </div>
      <div>
        <p className="text-2xl text-main">{model.confidence * 100}%</p>
        <p className="text-lg">{model.determinedTime}s</p>
      </div>
    </div>
  );
};
