import type { BenefitProps } from "@/shared/types/interfaces";

export const Benefit = ({ children, title, description }: BenefitProps) => {
  return (
    <div className="flex flex-col items-center justify-center gap-4 bg-secondary-bg-button/50 p-5 rounded-2xl min-w-67.5">
      {children}
      <h2>{title}</h2>
      <p>{description}</p>
    </div>
  );
};
