import { FeedBacks } from "../models/models.js";

export class FeedBackController {
  async postFeedback(req, res) {
    const id = req.params.id;
    const { mark, model, manufactureYear } = req.body;
    try {
      const feedback = await FeedBacks.create({
        idUser: id,
        mark,
        model,
        manufactureYear,
      });
      return res.status(200).json(feedback);
    } catch (error) {
      return res.status(500).json({ message: error.message });
    }
  }
  async getFeedbacks(_, res) {
    try {
      const feedbacks = await FeedBacks.findAll();
      return res.status(200).json(feedbacks);
    } catch (error) {
      return res.status(500).json({ message: error.message });
    }
  }
}
