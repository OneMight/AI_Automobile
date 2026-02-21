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

      const statistic = await Statistics.findOne({
        where: { idUser: id },
      });

      if (!statistic) {
        return res.status(404).json({ message: "notFound" });
      }

      const modelDetermined = await DeterminedModels.findAll({
        where: { idUser: id },
      });

      const count = modelDetermined.length;

      if (count === 0) {
        await statistic.update({
          avg_percent: 0,
          recognitions: 0,
          processingTime: 0,
        });
        return res.status(200).json();
      }

      const totalConfidence = modelDetermined.reduce((sum, item) => {
        return sum + item.confidence;
      }, 0);

      const totalTime = modelDetermined.reduce((sum, item) => {
        return sum + parseFloat(item.determinedTime);
      }, 0);

      const avg_percent = totalConfidence / count;
      const processingTime = totalTime / count;

      await statistic.update({
        avg_percent,
        recognitions: count,
        processingTime,
      });

      return res.status(200).json();
    } catch (error) {
      console.error(error);
      return res.status(500).json({ message: error.message });
    }
  }
}
