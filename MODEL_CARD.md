---
language:
- ckb
license: cc-by-nc-4.0
library_name: onnx
pipeline_tag: text-to-speech
tags:
- text-to-speech
- tts
- kurdish
- sorani
- central-kurdish
- ckb
- onnx
- onnxruntime
- edge
- on-device
- vits
- piper
---

<p align="center">
  <img src="https://huggingface.co/RevgeAI/vekol-tts-ckb-edge/resolve/main/vekol-white.png" alt="Vekol" width="280"/>
</p>

# Vekol-TTS (Sorani, edge)

Central Kurdish (Sorani) text-to-speech that runs offline on CPU. A small ONNX model
(~77 MB) that turns Sorani text into a 22.05 kHz WAV, faster than real time on a laptop
CPU. Part of the Vekol hub by Revge.

- Model: `vekol-tts-ckb-edge`
- Language: Central Kurdish / Sorani (`ckb`), Arabic script
- Voice: one male Sorani voice
- Architecture: VITS (Piper recipe), fine-tuned two-stage, exported to ONNX
- Size: `model.onnx` ≈ 77 MB, 22.05 kHz
- Runtime: ONNX Runtime (CPU)

## License

CC-BY-NC 4.0 (non-commercial). Built with the Piper VITS recipe and trained with audio
from the Vekol server model (built on Coqui XTTS, non-commercial), so it inherits
non-commercial terms. See `NOTICE`. Commercial use needs a model retrained from a
permissively-licensed base on permissively-licensed audio.

## Usage

The simplest path is the `vekol_tts.py` helper from the GitHub repo, which downloads this
model and handles text normalization (variant folding, number reading, no dropped letters):

```bash
pip install onnxruntime numpy scipy huggingface_hub
python3 vekol_tts.py "ئەمڕۆ کەشەکە خۆشە" out.wav
```

It's a standard Piper/VITS ONNX graph, so it also runs directly via ONNX Runtime using the
`model.onnx.json` phoneme map (inputs: `input`, `input_lengths`, `scales`, `sid`), or through
the Piper / sherpa-onnx runtimes.

## Notes

- The character map uses Sorani keheh `ک` (U+06A9) natively — no remapping needed. The
  helper folds typed variants (`ك`→`ک`, `ھ`→`ه`, `ة`→`ە`, `ى`→`ی`, hamza, ZWNJ/tatweel) so
  nothing is dropped, and reads numbers as Kurdish words.
- Clear consonant articulation — including Sorani gaf `گ` and the rarer `ڤ` / `ح` / `غ`,
  which are balanced in training so none is under-pronounced.
- Long sentences are split at their commas automatically, so even paragraph-length input
  reads cleanly with no trailing artifacts.
- Output is 22.05 kHz: clear and accurate, intended for on-device/CPU use.
- Runs through ONNX Runtime on many platforms (C++, Rust, Go, Android, iOS, Raspberry Pi,
  e.g. via sherpa-onnx or Piper).

## Links

- Higher-quality hosted version: https://vekol.krd
- Code: https://github.com/Revge/vekol-tts-ckb-edge
- Training recipe: https://github.com/OHF-Voice/piper1-gpl

## Citation

```bibtex
@software{vekol_tts_ckb_edge,
  title        = {Vekol-TTS: Sorani (Central Kurdish) on-device TTS},
  author       = {Shvan, Darvan},
  organization = {Revge},
  year         = {2026},
  url          = {https://github.com/Revge/vekol-tts-ckb-edge}
}
```

Built by Darvan Shvan at Revge, part of the Vekol hub.
