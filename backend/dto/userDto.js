export class UserDto {
  FIO_employee;
  id;
  isAdmin;
  category;
  speciality;
  constructor(model) {
    this.FIO_employee = model.FIO_Employee;
    this.category = model.Category;
    this.id = model.idEmployee;
    this.speciality = model.Speciality;
    this.isAdmin = model.isAdmin;
  }
}
