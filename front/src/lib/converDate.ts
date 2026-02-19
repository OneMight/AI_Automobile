export const convertDate = (date: Date) => {
  const dateObj = new Date(date);
  return (
    `${dateObj.getDate() < 10 ? `0${dateObj.getDate()}` : `${dateObj.getDate()}`}.` +
    `${dateObj.getMonth() + 1 < 10 ? `0${dateObj.getMonth() + 1}` : `${dateObj.getMonth() + 1}`}.` +
    `${dateObj.getFullYear()} ` +
    `${dateObj.getHours() < 10 ? `0${dateObj.getHours()}` : dateObj.getHours()}:` +
    `${dateObj.getMinutes() < 10 ? `0${dateObj.getMinutes()}` : dateObj.getMinutes()}:` +
    `${dateObj.getSeconds() < 10 ? `0${dateObj.getSeconds()}` : `${dateObj.getSeconds()}`}`
  );
};
