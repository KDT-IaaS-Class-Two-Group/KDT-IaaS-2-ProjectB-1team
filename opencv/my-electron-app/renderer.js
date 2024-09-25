const { ipcRenderer } = require('electron');

document.getElementById('uploadBtn').addEventListener('click', () => {
    const fileInput = document.getElementById('fileInput');
    const filePath = fileInput.files[0].path;

    ipcRenderer.invoke('process-image', filePath).then((data) => {
        console.log('Landmark Data:', data);
        document.getElementById('output').innerText = JSON.stringify(data, null, 2);
    }).catch(err => {
        console.error('Error:', err);
    });
});
