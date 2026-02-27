import { axiosInstance } from ".";

export const postFeedback = async ({
  id,
  mark,
  model,
  manufactureYear,
}: {
  id: number | undefined;
  mark: string;
  model: string;
  manufactureYear: string;
}) => {
  await axiosInstance.post(`api/feedback/${id}`, {
    mark: mark,
    model: model,
    manufactureYear: manufactureYear,
  });
};
