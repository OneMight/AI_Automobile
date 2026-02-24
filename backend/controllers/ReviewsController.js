import { sequelize } from "../db.js";
import { Reviews, User } from "../models/models.js";
export class ReviewsController {
  async postReview(req, res) {
    try {
      const { id } = req.params;
      const { rating, description } = req.body;
      const review = await Reviews.create({
        rating,
        description,
        idUser: id,
      });
      return res.status(200).json(review);
    } catch (error) {
      return res.status(500).json({ message: error.message });
    }
  }
  async getReviews(req, res) {
    const limit = parseInt(req.query.limit) || 10;
    const offset = parseInt(req.query.offset) || 0;
    try {
      const reviews = await Reviews.findAndCountAll({
        limit: limit,
        offset: offset,
        include: [
          {
            model: User,
            attributes: [],
          },
        ],
        attributes: {
          include: [
            [sequelize.col("User.email"), "email"],
            [sequelize.col("User.role"), "role"],
          ],
        },
        order: [["createdAt", "DESC"]],
        raw: true,
      });
      return res.status(200).json(reviews);
    } catch (error) {
      return res.status(500).json({ message: error.message });
    }
  }
}
