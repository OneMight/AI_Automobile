import { Op } from "sequelize";
import { DeterminedModels, Statistics } from "../models/models.js";
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
  async updateStatistic(req, res) {
    try {
      const { id } = req.params;
      const { idModel } = req.body;
      const statistic = await Statistics.findOne({
        where: {
          idUser: id,
        },
      });
      if (!statistic) {
        return res.status(404).json({ message: "notFound" });
      }
      const modelDetermined = await DeterminedModels.findByPk(idModel);
      const recognitions = statistic.recognitions + 1;
      const avg_percent =
        (parseFloat(statistic.avg_percent) +
          parseFloat(modelDetermined.confidence)) /
        (recognitions == 0 ? 1 : recognitions);
      const processingTime =
        (statistic.processingTime == null
          ? 0 + modelDetermined.determinedTime
          : parseFloat(statistic.processingTime) +
            parseFloat(modelDetermined.determinedTime)) / recognitions;
      await statistic.update({ avg_percent, recognitions, processingTime });
      return res.status(200).json();
    } catch (error) {
      console.error(error);
      return res.status(500).json({ message: error.message });
    }
  }
}
