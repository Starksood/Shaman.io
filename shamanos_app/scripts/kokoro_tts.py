#!/Users/CIM/Library/Python/3.9/bin/python3
# kokoro_tts.py — Kokoro TTS synthesis script
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Kokoro TTS synthesis")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--output", required=True, help="Output WAV file path")
    parser.add_argument("--voice", default="af_sky", help="Voice name (default: af_sky)")
    parser.add_argument("--speed", type=float, default=0.9, help="Speech speed (default: 0.9)")
    args = parser.parse_args()

    try:
        import numpy as np
        import soundfile as sf
        from kokoro import KPipeline

        pipeline = KPipeline(lang_code='a')
        generator = pipeline(
            args.text,
            voice=args.voice,
            speed=args.speed,
            split_pattern=r'\n+'
        )

        chunks = []
        for gs, ps, audio in generator:
            # audio is a torch tensor — convert to numpy
            if hasattr(audio, 'detach'):
                audio = audio.detach().cpu().numpy()
            elif hasattr(audio, 'numpy'):
                audio = audio.numpy()
            audio = np.asarray(audio, dtype=np.float32).flatten()
            if audio.size > 0:
                chunks.append(audio)

        if not chunks:
            print("Error: no audio generated", file=sys.stderr)
            sys.exit(1)

        audio_out = np.concatenate(chunks)
        sf.write(args.output, audio_out, 24000)
        sys.exit(0)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
