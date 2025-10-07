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

  const fullscreenModal = document.getElementById('fullscreenModal');
  const fullscreenImage = document.getElementById('fullscreenImage');
  const closeModal = document.getElementById('closeModal');
  const downloadImageBtn = document.getElementById('downloadImage');

  const chatbotToggle = document.getElementById('chatbotToggle');
  const chatbotWindow = document.getElementById('chatbotWindow');
  const chatbotClose = document.getElementById('chatbotClose');
  const chatbotMessages = document.getElementById('chatbotMessages');
  const chatbotInput = document.getElementById('chatbotInput');
  const chatbotSend = document.getElementById('chatbotSend');

  let selectedFile = null;

  browseBtn.addEventListener('click', () => fileInput.click());
  uploadArea.addEventListener('dragover', e => { e.preventDefault(); uploadArea.classList.add('dragover'); });
  uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
  uploadArea.addEventListener('drop', e => { e.preventDefault(); uploadArea.classList.remove('dragover'); handleFiles(e.dataTransfer.files); });
  fileInput.addEventListener('change', e => handleFiles(e.target.files));
  removeImage.addEventListener('click', removeSelectedImage);

  objectDetectionBtn.addEventListener('click', () => processImage('object-detection'));
  threeDMapBtn.addEventListener('click', () => processImage('3d-scene-map'));
  closeModal.addEventListener('click', () => fullscreenModal.style.display = 'none');

  function handleFiles(files) {
    const file = files[0];
    if (file && file.type.startsWith('image/')) {
      selectedFile = file;
      const reader = new FileReader();
      reader.onload = e => {
        previewImage.src = e.target.result;
        previewContainer.style.display = 'block';
        uploadArea.style.display = 'none';
      };
      reader.readAsDataURL(file);
    } else {
      alert('Please select a valid image file.');
    }
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

  loadingIndicator.style.display = 'flex';
  resultsSection.style.display = 'block';
  resultContainer.innerHTML = '';

  try {
    const formData = new FormData();

    let apiUrl = '';
    let resultTitle = '';

    if (type === 'object-detection') {
      apiUrl = 'http://localhost:8000/detect-batch';
      formData.append('images', selectedFile); // ✅ backend expects "images"
      resultTitle = 'Object Detection Results';
    } else if (type === '3d-scene-map') {
      apiUrl = 'http://localhost:8000/3d-scene-map';
      formData.append('image', selectedFile); // ✅ backend expects "image"
      formData.append('detection_method', 'auto'); // ✅ include optional form field
      resultTitle = '3D Scene Map Results';
    }

    const response = await fetch(apiUrl, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`API request failed: ${response.status} ${errorText}`);
    }

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
        img.addEventListener('click', () => openFullscreen(img.src));

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

  // Chatbot Functionality
  chatbotToggle.addEventListener('click', toggleChatbot);
  chatbotClose.addEventListener('click', toggleChatbot);
  chatbotSend.addEventListener('click', sendMessage);
  chatbotInput.addEventListener('keypress', e => {
    if (e.key === 'Enter') sendMessage();
  });

  function toggleChatbot() {
    chatbotWindow.style.display = chatbotWindow.style.display === 'flex' ? 'none' : 'flex';
    if (chatbotWindow.style.display === 'flex' && chatbotMessages.innerHTML === '') {
      addBotMessage('Hello! Ask me anything about computer vision.');
    }
  }

  async function sendMessage() {
    const message = chatbotInput.value.trim();
    if (!message) return;

    addUserMessage(message);
    chatbotInput.value = '';
    addBotMessage('Thinking...');

    try {
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message })
      });

      if (!response.ok) throw new Error(`Chat API failed: ${response.status}`);

      const data = await response.json();
      chatbotMessages.lastChild.remove();
      addBotMessage(data.response || 'Sorry, I couldn\'t process that.');
    } catch (err) {
      console.error('Chat Error:', err);
      chatbotMessages.lastChild.remove();
      addBotMessage('Error: Could not connect to the chatbot API.');
    }

    chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
  }

  function addUserMessage(text) {
    const msg = document.createElement('div');
    msg.className = 'message user-message';
    msg.textContent = text;
    chatbotMessages.appendChild(msg);
  }

  function addBotMessage(text) {
    const msg = document.createElement('div');
    msg.className = 'message bot-message';
    msg.textContent = text;
    chatbotMessages.appendChild(msg);
  }
});
