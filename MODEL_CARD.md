---
language:
- ckb
license: cc-by-nc-4.0
library_name: onnx
pipeline_tag: text-to-speech
base_model: facebook/mms-tts-kmr-script_arabic
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
- mms
---

<p align="center">
  <img src="https://huggingface.co/RevgeAI/vekol-tts-ckb-edge/resolve/main/vekol-white.png" alt="Vekol" width="280"/>
</p>

# Vekol-TTS (Sorani, edge)

Central Kurdish (Sorani) text-to-speech that runs offline on CPU. A small ONNX model
(~114 MB) that turns Sorani text into a 16 kHz WAV, about 8× faster than real time on a
laptop CPU. Part of the Vekol hub by Revge.

- Model: `vekol-tts-ckb-edge`
- Language: Central Kurdish / Sorani (`ckb`), Arabic script
- Voice: one male Sorani voice
- Architecture: VITS, fine-tuned from `facebook/mms-tts-kmr-script_arabic`, exported to ONNX
- Size: ~36 M params, `model.onnx` ≈ 114 MB
- Runtime: ONNX Runtime (CPU)

## License

CC-BY-NC 4.0 (non-commercial). Fine-tuned from Meta MMS-TTS (CC-BY-NC 4.0) and trained
with audio from the Vekol server model (built on Coqui XTTS, non-commercial), so it
inherits non-commercial terms. See `NOTICE`. Commercial use needs a model retrained from
a permissively-licensed base.

## Usage

```bash
pip install onnxruntime numpy scipy transformers huggingface_hub
```

```python
from huggingface_hub import snapshot_download
import os, numpy as np, onnxruntime as ort, scipy.io.wavfile
from transformers import AutoTokenizer

d = snapshot_download("RevgeAI/vekol-tts-ckb-edge")
tok = AutoTokenizer.from_pretrained(d)
sess = ort.InferenceSession(os.path.join(d, "model.onnx"), providers=["CPUExecutionProvider"])

text = "ئەمڕۆ کەشەکە خۆشە".replace("ک", "ك")   # keheh -> kaf (the vocab uses kaf)
ids = tok(text, return_tensors="np")["input_ids"].astype(np.int64)
wav = sess.run(None, {"input_ids": ids})[0][0]
scipy.io.wavfile.write("out.wav", 16000, wav.astype("float32"))
```

Or use the `vekol_tts.py` helper from the GitHub repo, which downloads this model.

## Notes

- Replace Sorani keheh `ک` (U+06A9) with Arabic kaf `ك` (U+0643) before tokenizing; the
  vocabulary uses kaf, so without it words with `ک` lose the letter.
- Output is 16 kHz: clear and accurate, intended for on-device/CPU use.
- Runs through ONNX Runtime on many platforms (C++, Rust, Go, Android, iOS, Raspberry Pi,
  e.g. via sherpa-onnx).

## Links

- Higher-quality hosted version: https://vekol.krd
- Code: https://github.com/Revge/vekol-tts-ckb-edge
- Base model: https://huggingface.co/facebook/mms-tts-kmr-script_arabic

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
