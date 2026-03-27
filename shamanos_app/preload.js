// preload.js — Context bridge between main and renderer
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('shaman', {
  checkDeps: () => ipcRenderer.invoke('check-deps'),
  startListening: () => ipcRenderer.invoke('start-listening'),
  transcribe: (audioPath) => ipcRenderer.invoke('transcribe', audioPath),
  queryGuide: (text, history) => ipcRenderer.invoke('query-guide', text, history),
  speak: (text) => ipcRenderer.invoke('speak', text),
});
