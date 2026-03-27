// backend/stt.js — Whisper STT wrapper
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const WHISPER_BIN = path.join(__dirname, '../models/whisper.cpp/build/bin/whisper-cli');
const WHISPER_MODEL = path.join(__dirname, '../models/ggml-tiny.en.bin');

const NOISE_TOKENS = ['[BLANK_AUDIO]', '[MUSIC]', '[NOISE]', '(music)', '(noise)'];

function checkInstalled() {
  return fs.existsSync(WHISPER_BIN) && fs.existsSync(WHISPER_MODEL);
}

async function transcribe(audioPath) {
  return new Promise((resolve, reject) => {
    const args = [
      '-m', WHISPER_MODEL,
      '-f', audioPath,
      '-l', 'en',
      '--no-timestamps',
      '-nt',
      '-t', '4',
    ];

    const proc = spawn(WHISPER_BIN, args);
    let stdout = '';
    let stderr = '';

    proc.stdout.on('data', (data) => { stdout += data.toString(); });
    proc.stderr.on('data', (data) => { stderr += data.toString(); });

    proc.on('error', reject);

    proc.on('close', (code) => {
      if (code !== 0) {
        return reject(new Error(`whisper-cli exited with code ${code}: ${stderr}`));
      }

      let text = stdout;
      for (const token of NOISE_TOKENS) {
        text = text.split(token).join('');
      }

      resolve(text.trim());
    });
  });
}

module.exports = { transcribe, checkInstalled };
