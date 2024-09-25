const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { execFile } = require('child_process');
const fs = require('fs');

let mainWindow;

app.whenReady().then(() => {
    mainWindow = new BrowserWindow({
        width: 800,
        height: 600,
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            nodeIntegration: true,
            contextIsolation: false,
        }
    });
    mainWindow.loadFile('index.html');
});

ipcMain.handle('process-image', async (event, filePath) => {
    return new Promise((resolve, reject) => {
        const pythonScript = path.join(__dirname, 'process_image.py');
        execFile('python', [pythonScript, filePath], (error, stdout, stderr) => {
            if (error) {
                reject(stderr);
            } else {
                const jsonFile = path.join(__dirname, 'success_directory', 'face_landmarks.json');
                fs.readFile(jsonFile, 'utf8', (err, data) => {
                    if (err) {
                        reject(err);
                    } else {
                        resolve(JSON.parse(data));
                    }
                });
            }
        });
    });
});
