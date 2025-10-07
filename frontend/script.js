document.addEventListener('DOMContentLoaded', function() {
  const uploadArea = document.getElementById('uploadArea');
  const fileInput = document.getElementById('fileInput');
  const previewContainer = document.getElementById('previewContainer');
  const previewImage = document.getElementById('previewImage');
  const removeImage = document.getElementById('removeImage');
  const objectDetectionBtn = document.getElementById('objectDetectionBtn');
  const threeDMapBtn = document.getElementById('threeDMapBtn');
  const resultsSection = document.getElementById('resultsSection');
  const resultContainer = document.getElementById('resultContainer');
  const loadingIndicator = document.getElementById('loadingIndicator');
  const browseBtn = document.querySelector('.browse-btn');

  // Modal elements
  const fullscreenModal = document.getElementById('fullscreenModal');
  const fullscreenImage = document.getElementById('fullscreenImage');
  const closeModal = document.getElementById('closeModal');
  const downloadImageBtn = document.getElementById('downloadImage');

  let selectedFile = null;

  browseBtn.addEventListener('click', () => fileInput.click());
  uploadArea.addEventListener('dragover', handleDragOver);
  uploadArea.addEventListener('drop', handleDrop);
  fileInput.addEventListener('change', handleFileSelect);
  removeImage.addEventListener('click', removeSelectedImage);
  objectDetectionBtn.addEventListener('click', () => processImage('object-detection'));
  threeDMapBtn.addEventListener('click', () => processImage('3d-map'));
  closeModal.addEventListener('click', () => fullscreenModal.style.display = 'none');

  function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
  }

  function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length) handleFiles(files);
  }

  function handleFileSelect(e) {
    handleFiles(e.target.files);
  }

  function handleFiles(files) {
    const file = files[0];
    if (file && file.type.startsWith('image/')) {
      selectedFile = file;
      showImagePreview(file);
    } else {
      alert('Please select an image file (JPG, PNG, etc.)');
    }
  }

  function showImagePreview(file) {
    const reader = new FileReader();
    reader.onload = e => {
      previewImage.src = e.target.result;
      previewContainer.style.display = 'block';
      uploadArea.style.display = 'none';
    };
    reader.readAsDataURL(file);
  }

  function removeSelectedImage() {
    selectedFile = null;
    previewContainer.style.display = 'none';
    uploadArea.style.display = 'block';
    resultsSection.style.display = 'none';
    resultContainer.innerHTML = '';
  }

  async function processImage(type) {
    if (!selectedFile) {
      alert('Please select an image first');
      return;
    }

    loadingIndicator.style.display = 'block';
    resultsSection.style.display = 'block';
    resultContainer.innerHTML = '';

    try {
      const formData = new FormData();
      formData.append('images', selectedFile);

      let apiUrl = '';
      let resultTitle = '';

      if (type === 'object-detection') {
        apiUrl = 'http://localhost:8000/detect-batch';
        resultTitle = 'Object Detection Results';
      } else {
        apiUrl = 'http://localhost:8000/3d-scene-map';
        resultTitle = '3D Scene Map Results';
      }

      const response = await fetch(apiUrl, { method: 'POST', body: formData });
      if (!response.ok) throw new Error(`API request failed: ${response.status}`);

      const data = await response.json();
      displayResults(data, resultTitle);
    } catch (err) {
      console.error('Error:', err);
      displayError(err.message);
    } finally {
      loadingIndicator.style.display = 'none';
    }
  }

  function displayResults(data, title) {
    resultContainer.innerHTML = '';

    const resultItem = document.createElement('div');
    resultItem.className = 'result-item';
    const titleElement = document.createElement('h3');
    titleElement.textContent = title;
    resultItem.appendChild(titleElement);

    if (data.results) {
      data.results.forEach(item => {
        const div = document.createElement('div');
        div.className = 'result-info';

        const img = document.createElement('img');
        img.className = 'result-image';
        img.src = `http://localhost:8000${item.result_image_url}`;
        img.addEventListener('click', () => openFullscreen(img.src)); // ✅ click to enlarge

        const details = document.createElement('div');
        details.className = 'result-details';
        details.innerHTML = `
          <h4>${item.filename}</h4>
          <p><strong>Detections:</strong> ${item.detection_count}</p>
          <p><strong>Detected:</strong> ${item.detected_objects.join(', ')}</p>
          <p><strong>Inference:</strong> ${item.inference_time_ms} ms</p>
        `;

        div.appendChild(img);
        div.appendChild(details);
        resultItem.appendChild(div);
      });
    } else {
      resultItem.innerHTML += `<p>No results found</p>`;
    }

    resultContainer.appendChild(resultItem);
    resultsSection.scrollIntoView({ behavior: 'smooth' });
  }

  function openFullscreen(src) {
    fullscreenImage.src = src;
    fullscreenModal.style.display = 'flex';
    downloadImageBtn.onclick = () => {
      const a = document.createElement('a');
      a.href = src;
      a.download = 'processed_image.png';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    };
  }

  function displayError(msg) {
    resultContainer.innerHTML = `
      <div class="result-item">
        <h3 style="color:red;">Error</h3>
        <p>${msg}</p>
        <p>Please make sure backend API is running on <strong>http://localhost:8000</strong></p>
      </div>
    `;
  }
});
