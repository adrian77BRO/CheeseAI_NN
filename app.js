const imageInput = document.getElementById('imageInput');
const sendBtn = document.getElementById('sendBtn');
const previewDiv = document.getElementById('preview');
const resultDiv = document.getElementById('result');
const percentDiv = document.getElementById('percent');

let selectedFile = null;

// Mostrar imagen previa
imageInput.addEventListener('change', () => {
    const file = imageInput.files[0];
    if (!file) return;

    selectedFile = file;
    sendBtn.disabled = false;

    const reader = new FileReader();
    reader.onload = (e) => {
        previewDiv.innerHTML = `<img src='${e.target.result}' alt='Preview' />`;
        resultDiv.textContent = '';
        percentDiv.textContent = ''
    };
    reader.readAsDataURL(file);
});

// Enviar a la API
sendBtn.addEventListener('click', () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('file', selectedFile);

    fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData
    })
        .then(res => res.json())
        .then(data => {
            if (!data.className) {
                resultDiv.textContent = '⚠️ No se pudo clasificar la imagen.';
                return;
            }
            resultDiv.textContent = `Tipo de queso: ${data.className}`;
            percentDiv.textContent = `Predicción: ${(data.confidence * 100).toFixed(2)}%`
        })
        .catch(err => {
            console.error(err);
            resultDiv.textContent = '❌ Error al enviar la imagen.';
        });
});