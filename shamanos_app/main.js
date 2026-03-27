// main.js — Electron main process entry point
const { app, BrowserWindow, ipcMain, systemPreferences } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

const stt = require('./backend/stt');
const tts = require('./backend/tts');
const ollama = require('./backend/ollama');
const audioCapture = require('./backend/audio_capture');

let mainWindow = null;
const activeChildren = new Set();

// ── Window creation (task 2.1) ────────────────────────────────────────────────
function createWindow() {
  mainWindow = new BrowserWindow({
    fullscreen: true,
    frame: false,
    backgroundColor: '#020209',
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      preload: path.join(__dirname, 'preload.js'),
    },
  });

  mainWindow.loadFile('renderer/index.html');

  // Open devtools to debug
  mainWindow.webContents.openDevTools();

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// ── App ready (tasks 2.1 + 2.2) ───────────────────────────────────────────────
app.whenReady().then(async () => {
  // Microphone permission on macOS (task 2.2)
  if (process.platform === 'darwin') {
    await systemPreferences.askForMediaAccess('microphone');
  }

  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

// ── Before-quit: kill any running child processes (task 9.5) ─────────────────
app.on('before-quit', () => {
  for (const child of activeChildren) {
    try { child.kill(); } catch (_) {}
  }
  activeChildren.clear();
});

// ── IPC: check-deps (task 2.3) ────────────────────────────────────────────────
ipcMain.handle('check-deps', async () => {
  try {
    const [whisper, kokoro, ollamaOk] = await Promise.all([
      Promise.resolve(stt.checkInstalled()),
      Promise.resolve(tts.checkInstalled()),
      ollama.ping(),
    ]);
    return { whisper, kokoro, ollama: ollamaOk, allReady: whisper && kokoro && ollamaOk };
  } catch (err) {
    return { whisper: false, kokoro: false, ollama: false, allReady: false, error: err.message };
  }
});

// ── IPC: start-listening (task 2.4) ───────────────────────────────────────────
ipcMain.handle('start-listening', async () => {
  try {
    const audioPath = await audioCapture.recordUntilSilence((soxProc) => {
      activeChildren.add(soxProc);
      soxProc.on('close', () => activeChildren.delete(soxProc));
    });
    return { success: true, audioPath };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

// ── IPC: transcribe (task 2.5) ────────────────────────────────────────────────
ipcMain.handle('transcribe', async (_event, audioPath) => {
  try {
    const text = await stt.transcribe(audioPath);
    return { success: true, text };
  } catch (err) {
    return { success: false, error: err.message };
  } finally {
    // Clean up input WAV file after transcription
    if (audioPath) {
      try { fs.unlinkSync(audioPath); } catch (_) {}
    }
  }
});

// ── IPC: query-guide (task 2.6) ───────────────────────────────────────────────
ipcMain.handle('query-guide', async (_event, userText, history) => {
  try {
    const response = await ollama.chat(userText, history);
    return { success: true, response };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

// ── IPC: speak (task 2.7) ─────────────────────────────────────────────────────
ipcMain.handle('speak', async (_event, text) => {
  let wavPath = null;
  try {
    wavPath = await tts.synthesize(text);

    await new Promise((resolve, reject) => {
      const player = spawn('afplay', [wavPath]);
      activeChildren.add(player);

      player.on('close', (code) => {
        activeChildren.delete(player);
        if (code === 0 || code === null) resolve();
        else reject(new Error(`afplay exited with code ${code}`));
      });

      player.on('error', (err) => {
        activeChildren.delete(player);
        reject(err);
      });
    });

    return { success: true };
  } catch (err) {
    return { success: false, error: err.message };
  } finally {
    // Clean up temp WAV file
    if (wavPath) {
      try { fs.unlinkSync(wavPath); } catch (_) {}
    }
  }
});
