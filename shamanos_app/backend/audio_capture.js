// backend/audio_capture.js — Sox microphone recorder
const { spawn } = require('child_process');
const path = require('path');
const os = require('os');
const fs = require('fs');

const SILENCE_DURATION = 1.5;
const MIN_RECORD_DURATION = 0.8;
const SAMPLE_RATE = 16000;
const MAX_DURATION = 120;

async function recordUntilSilence(onProcess) {
  return new Promise((resolve, reject) => {
    const outputPath = path.join(os.tmpdir(), `shaman_input_${Date.now()}.wav`);
    const startTime = Date.now();

    const args = [
      '-d',
      '-r', String(SAMPLE_RATE),
      '-c', '1',
      '-b', '16',
      outputPath,
      'silence', '1', '0.1', '3%', '1', String(SILENCE_DURATION), '3%',
    ];

    let sox;
    try {
      sox = spawn('sox', args);
    } catch (err) {
      if (err.code === 'ENOENT') {
        return reject(new Error('sox not found. Install it with: brew install sox'));
      }
      return reject(err);
    }

    // Expose the child process so callers can track it for cleanup
    if (typeof onProcess === 'function') onProcess(sox);

    const safetyTimer = setTimeout(() => {
      sox.kill();
    }, MAX_DURATION * 1000);

    sox.on('error', (err) => {
      clearTimeout(safetyTimer);
      if (err.code === 'ENOENT') {
        reject(new Error('sox not found. Install it with: brew install sox'));
      } else {
        reject(err);
      }
    });

    sox.on('close', (code) => {
      clearTimeout(safetyTimer);

      const duration = (Date.now() - startTime) / 1000;

      if (duration < MIN_RECORD_DURATION) {
        return reject(new Error(`Recording too short (${duration.toFixed(2)}s < ${MIN_RECORD_DURATION}s minimum)`));
      }

      if (!fs.existsSync(outputPath)) {
        return reject(new Error('Recording failed: output file not created'));
      }

      resolve(outputPath);
    });
  });
}

module.exports = { recordUntilSilence };
