import { Op } from "sequelize";
import { Statistics } from "../models/models.js";
export class StatisticController {
  async getStatisticsById(req, res) {
    try {
      const id = req.params.id;
      const statistic = await Statistics.findOne({
        where: {
          idUser: id,
        },
      });
      if (!statistic) {
        return res.status(404).json({ message: "Statistic not found" });
      }
      return res.status(200).json(statistic);
    } catch (error) {
      return res.status(500).json({ message: error.message });
    }
  }
}
