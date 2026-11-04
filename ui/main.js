const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');

let win;

function createWindow() {
  win = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 50,
    minHeight: 50,
    transparent: true,
    frame: false,
    resizable: true,
    hasShadow: false,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false
    }
  });

  win.maximize();
  win.loadFile('ui/index.html');

  ipcMain.on('window-controls', (event, command) => {
    if (command === 'minimize') win.minimize();
    if (command === 'maximize') {
      if (win.isMaximized()) win.restore();
      else win.maximize();
    }
    if (command === 'close') win.close();
  });

  ipcMain.on('resize-window', (event, width, height, isOverlay) => {
    win.setAlwaysOnTop(isOverlay, 'screen-saver');
    if (isOverlay) {
      if (win.isMaximized()) win.restore();
      win.setResizable(false);
      win.setSize(80, 80);        // ← exactly the size of the avatar
      win.center();               // center on start
    } else {
      win.setResizable(true);
      win.maximize();
    }
  });

  // Replaces the old move-window handler:
  ipcMain.on('move-window-absolute', (event, x, y) => {
    win.setPosition(x, y);
  });

  ipcMain.on('set-ignore-mouse-events', (event, ignore, options) => {
    if (win) win.setIgnoreMouseEvents(ignore, options || {});
  });
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});