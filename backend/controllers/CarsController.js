import { Cars } from "../models/models.js";

export class CarsController {
  async get(_, res) {
    try {
      const cars = await Cars.findAll();

      if (!cars || cars.length === 0) {
        return res.status(404).json({ message: "Машины не найдены" });
      }

      return res.json(cars);
    } catch (e) {
      console.error(e);
      return res
        .status(500)
        .json({ message: "Ошибка сервера при получении машин" });
    }
  }
  async create(req, res) {
    const { mark, model, manufactureYear } = req.body;
    try {
      const car = Cars.create({
        model,
        mark,
        manufactureYear,
      });
      return res.status(200).json(car);
    } catch (error) {
      return res.status(500).json({ message: error });
    }
  }
  CarMiddleware = async (req, res, next) => {
    try {
      const { mark, model, manufactureYear } = req.body;
      const car = await Cars.findOne({
        where: {
          mark: mark,
          model: model,
          manufactureYear: manufactureYear,
        },
      });
      if (!car) {
        await Cars.create({
          mark,
          model,
          manufactureYear,
        });
      }
      next();
    } catch (error) {
      return res.status(500).json({ message: error.message });
    }
  };
}
