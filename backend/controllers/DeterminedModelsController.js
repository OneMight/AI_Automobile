import { Op } from "sequelize";
import { DeterminedModels } from "../models/models.js";
export class DeterminedModelsController {
  async getDeterminedModelsByUserId(req, res) {
    try {
      const id = req.params.id;
      const determinedModels = await DeterminedModels.findAll({
        where: {
          idUser: id,
        },
      });
      if (!determinedModels) {
        return res.status(404).json({ message: "Determined models not found" });
      }
      return res.status(200).json(determinedModels);
    } catch (error) {
      return res.status(500).json({ message: error.message });
    }
  }
}
