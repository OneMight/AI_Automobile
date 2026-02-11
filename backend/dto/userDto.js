export class UserDto {
  email;
  id;
  age;
  constructor(model) {
    this.email = model.email;
    this.id = model.id;
    this.age = model.age;
  }
}
