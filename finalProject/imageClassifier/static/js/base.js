const image_input = document.querySelector("#imgInput");
var uploaded_image;

image_input.addEventListener('change', function() {
  const reader = new FileReader();
  reader.addEventListener('load', () => {
    uploaded_image = reader.result;
    document.querySelector("#imgContainer").style.backgroundImage = `url(${uploaded_image})`;
    document.querySelector("#conText").style.display = "none";
  });
  reader.readAsDataURL(this.files[0]);
});