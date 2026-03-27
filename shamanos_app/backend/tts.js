// backend/tts.js — Kokoro TTS wrapper
const { spawn, execSync } = require('child_process');
const path = require('path');
const os = require('os');
const fs = require('fs');

const KOKORO_SCRIPT = path.join(__dirname, '../scripts/kokoro_tts.py');

// Find the Python that has kokoro installed
function getPython() {
  const candidates = [
    '/Users/CIM/Library/Python/3.9/bin/python3',
    '/usr/local/bin/python3',
    '/opt/homebrew/bin/python3',
    'python3',
  ]
  for (const p of candidates) {
    try {
      execSync(`"${p}" -c "import kokoro"`, { stdio: 'ignore', timeout: 5000 })
      return p
    } catch (_) {}
  }
  return 'python3' // fallback
}

const PYTHON = getPython()

async function checkInstalled() {
  return new Promise((resolve) => {
    const proc = spawn(PYTHON, ['-c', 'import kokoro']);
    proc.on('error', () => resolve(false));
    proc.on('close', (code) => resolve(code === 0));
  });
}

async function synthesize(text) {
  return new Promise((resolve, reject) => {
    const outputPath = path.join(os.tmpdir(), `shaman_output_${Date.now()}.wav`);

    const args = [
      KOKORO_SCRIPT,
      '--text', text,
      '--output', outputPath,
      '--voice', 'af_sky',
      '--speed', '0.9',
    ];

    const proc = spawn(PYTHON, args);
    let stderr = '';

    proc.stderr.on('data', (data) => { stderr += data.toString(); });

    proc.on('error', reject);

    proc.on('close', (code) => {
      if (code !== 0) {
        return reject(new Error(`kokoro_tts.py failed (exit ${code}): ${stderr.trim()}`));
      }

      if (!fs.existsSync(outputPath)) {
        return reject(new Error(`TTS output file not created. ${stderr.trim()}`));
      }

      resolve(outputPath);
    });
  });
}

module.exports = { synthesize, checkInstalled };
