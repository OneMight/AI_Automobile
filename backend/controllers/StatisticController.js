import { Op } from "sequelize";
import { DeterminedModels, Statistics, User } from "../models/models.js";
import { sequelize } from "../db.js";
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
  async getUserAgeStats(_, res) {
    try {
      const ages = await User.findAll({
        attributes: [
          [
            sequelize.literal(
              `SUM(CASE WHEN age >= 10 AND age < 18 THEN 1 ELSE 0 END)`,
            ),
            "young",
          ],
          [
            sequelize.literal(
              `SUM(CASE WHEN age >= 18 AND age < 30 THEN 1 ELSE 0 END)`,
            ),
            "adult",
          ],
          [
            sequelize.literal(`SUM(CASE WHEN age >= 30 THEN 1 ELSE 0 END)`),
            "senior",
          ],
        ],
      });

      const result = [
        { category: "10-18", count: parseInt(ages[0].dataValues.young) || 0 },
        { category: "18-30", count: parseInt(ages[0].dataValues.adult) || 0 },
        { category: "30+", count: parseInt(ages[0].dataValues.senior) || 0 },
      ];

      return res.status(200).json(result);
    } catch (error) {
      return res.status(500).json({ message: error.message });
    }
  }
  async getRecognitionStats(_, res) {
    try {
      const sevenDaysAgo = new Date();
      sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7);

      const stats = await DeterminedModels.findAll({
        attributes: [
          [sequelize.fn("DATE", sequelize.col("createdAt")), "day"],
          [sequelize.fn("COUNT", sequelize.col("id")), "count"],
        ],
        where: {
          createdAt: {
            [Op.gte]: sevenDaysAgo,
          },
        },
        group: [sequelize.fn("DATE", sequelize.col("createdAt"))],
        order: [[sequelize.fn("DATE", sequelize.col("createdAt")), "ASC"]],
      });

      return res.status(200).json(stats);
    } catch (error) {
      return res.status(500).json({ message: error.message });
    }
  }
}
