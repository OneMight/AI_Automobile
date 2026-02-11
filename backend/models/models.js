import { DataTypes } from "sequelize";
import { sequelize } from "../db.js";

const User = sequelize.define(
  "Users",
  {
    id: { type: DataTypes.BIGINT, primaryKey: true, autoIncrement: true },
    email: { type: DataTypes.STRING, allowNull: false },
    password: { type: DataTypes.STRING, allowNull: false },
    age: { type: DataTypes.INTEGER, allowNull: true },
  },
  {
    timestamps: false,
  },
);
const Tokens = sequelize.define(
  "Tokens",
  {
    id: { type: DataTypes.BIGINT, primaryKey: true, autoIncrement: true },
    idUser: { type: DataTypes.BIGINT, allowNull: false },
    refreshToken: { type: DataTypes.STRING(1024), allowNull: false },
  },
  {
    timestamps: false,
  },
);
const Statistics = sequelize.define(
  "Statistics",
  {
    id: { type: DataTypes.BIGINT, primaryKey: true, autoIncrement: true },
    idUser: { type: DataTypes.BIGINT, allowNull: false },
    avg_percent: { type: DataTypes.DECIMAL, allowNull: false },
    recognitions: { type: DataTypes.INTEGER, allowNull: false },
  },
  {
    timestamps: false,
  },
);
const DeterminedModels = sequelize.define(
  "DeterminedModels",
  {
    id: { type: DataTypes.BIGINT, primaryKey: true, autoIncrement: true },
    idUser: { type: DataTypes.BIGINT, allowNull: false },
    idCar: { type: DataTypes.BIGINT, allowNull: false },
    determinedTime: { type: DataTypes.TIME, allowNull: false },
  },
  {
    timestamps: false,
  },
);
const Cars = sequelize.define(
  "Cars",
  {
    id: { type: DataTypes.BIGINT, primaryKey: true, autoIncrement: true },
    mark: { type: DataTypes.STRING(50), allowNull: false },
    model: { type: DataTypes.STRING(100), allowNull: false },
    manufactureYear: { type: DataTypes.STRING, allowNull: false },
  },
  {
    timestamps: false,
  },
);
const Reviews = sequelize.define(
  "Reviews",
  {
    id: { type: DataTypes.BIGINT, primaryKey: true, autoIncrement: true },
    idUser: { type: DataTypes.BIGINT, allowNull: false },
    reting: { type: DataTypes.INTEGER, allowNull: false },
    description: { type: DataTypes.STRING, allowNull: true },
    created: { type: DataTypes.TIME, allowNull: false },
  },
  {
    timestamps: false,
  },
);
User.hasMany(Reviews, { foreignKey: "idUser" });
Reviews.belongsTo(User, { foreignKey: "idUser" });

User.hasOne(Tokens, { foreignKey: "idUser" });
Tokens.belongsTo(User, { foreignKey: "idUser" });

User.hasOne(Statistics, { foreignKey: "idUser" });
Statistics.belongsTo(User, { foreignKey: "idUser" });

User.hasMany(DeterminedModels, { foreignKey: "idUser" });
DeterminedModels.belongsTo(User, { foreignKey: "idUser" });

Cars.hasMany(DeterminedModels, { foreignKey: "idCar" });
DeterminedModels.belongsTo(Cars, { foreignKey: "idCar" });

export { User, Tokens, Cars, DeterminedModels, Statistics, Reviews };
