import { Op } from "sequelize";
import path from "path";
import { Cars, DeterminedModels, Statistics } from "../models/models.js";
import multer from "multer";
import { v2 as cloudinary } from "cloudinary";
import { CloudinaryStorage } from "multer-storage-cloudinary";

cloudinary.config({
  cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
  api_key: process.env.CLOUDINARY_API_KEY,
  api_secret: process.env.CLOUDINARY_API_SECRET,
});

const storage = new CloudinaryStorage({
  cloudinary: cloudinary,
  params: {
    folder: "car_recognitions",
    allowed_formats: ["jpg", "png", "jpeg"],
  },
});
export const upload = multer({ storage: storage });
// const upload = multer({ storage: storage });
// const storage = multer.diskStorage({
//   destination: "./uploads/",
//   filename: (_, file, cb) => {
//     cb(null, Date.now() + path.extname(file.originalname));
//   },
// });
// export const upload = multer({ storage });
export class DeterminedModelsController {
  async getDeterminedModelsByUserId(req, res) {
    try {
      const id = req.params.id;
      const { limit } = req.body;
      const determinedModels = await DeterminedModels.findAll({
        where: {
          idUser: id,
        },
        include: [
          {
            model: Cars,
            required: true,
          },
        ],
        order: [["createdAt", "DESC"]],
        limit: limit,
      });
      if (determinedModels.length == 0) {
        return res.status(404).json({ message: "Determined models not found" });
      }
      return res.status(200).json(determinedModels);
    } catch (error) {
      return res.status(500).json({ message: error.message });
    }
  }
  async postDeterminedModel(req, res) {
    try {
      const { id } = req.params;
      const { mark, model, manufactureYear, confidence, determinedTime } =
        req.body;
      if (!req.file) {
        return res
          .status(400)
          .json({ message: "Файл не получен. Проверьте имя поля в FormData" });
      }
      // const image = `/uploads/${req.file.filename}`;
      const image = req.file.path;
      const car = await Cars.findOne({
        where: {
          mark: mark,
          model: model,
          manufactureYear: manufactureYear,
        },
      });
      const statistic = await Statistics.findOne({
        where: {
          idUser: id,
        },
      });
      if (!statistic) {
        await Statistics.create({
          idUser: id,
          avg_percent: 0,
          processingTime: 0.0,
          recongitions: 0,
        });
      }
      const determinedModel = await DeterminedModels.create({
        idUser: id,
        idCar: car.id,
        confidence,
        determinedTime,
        modelImage: image,
      });
      return res.status(200).json(determinedModel.id);
    } catch (error) {
      return res.status(500).json({ message: error.message });
    }
  }
}
