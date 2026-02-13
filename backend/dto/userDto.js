export class UserDto {
  email;
  id;
  age;
  role;
  constructor(model) {
    this.email = model.email;
    this.id = model.id;
    this.age = model.age;
    this.role = model.role;
  }
}
