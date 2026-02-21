import type { ModelBlockProps } from "@/shared/types/interfaces";

export const ModelBlock = ({ model }: ModelBlockProps) => {
  return (
    <div className="flex flex-col mobile:flex-row w-full justify-between items-center mobile:p-3 pb-3 rounded-2xl bg-secondary-text/10">
      <div className="flex flex-col mobile:flex-row gap-3 w-full items-center justify-start">
        <img
          src={model.modelImage}
          className="w-full mobile:w-20 rounded-t-2xl mobile:rounded-none"
          alt=""
        />
        <p className="text-white">
          {model.Car?.mark} {model.Car?.model} {model.Car?.manufactureYear}
        </p>
      </div>
      <div className="flex flex-row mobile:flex-col w-full mt-2 mobile:mb-0 mobile:w-auto items-center justify-around mobile:p-0 px-3">
        <p className="mobile:text-2xl text-lg text-main">
          {model.confidence * 100}%
        </p>
        <p className="text-lg">{model.determinedTime}s</p>
      </div>
    </div>
  );
};
