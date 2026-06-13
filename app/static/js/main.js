document.addEventListener('DOMContentLoaded', function () {
  console.log('RetinaHeartPredict UI loaded successfully.');

  const uploadArea = document.querySelector('.upload-area');
  if (uploadArea) {
    uploadArea.addEventListener('dragover', function (event) {
      event.preventDefault();
      uploadArea.classList.add('border-primary');
    });

    uploadArea.addEventListener('dragleave', function () {
      uploadArea.classList.remove('border-primary');
    });
  }
});
