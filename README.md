<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/vekol-white.svg">
  <img src="assets/vekol-black.svg" alt="Vekol" width="300">
</picture>

# Vekol&nbsp;·&nbsp;TTS

### Sorani speech, on every device.

Offline **Central Kurdish (Sorani)** text-to-speech that runs on plain **CPU** —
no GPU, no internet. A small ~77 MB ONNX model that speaks Sorani at 22 kHz in real time.

[![Higher quality](https://img.shields.io/badge/higher%20quality-vekol.krd-6f42c1)](https://vekol.krd)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/license-CC--BY--NC--4.0-555)](LICENSE)
[![Model](https://img.shields.io/badge/🤗-RevgeAI%2Fvekol--tts--ckb--edge-yellow)](https://huggingface.co/RevgeAI/vekol-tts-ckb-edge)
![Language](https://img.shields.io/badge/lang-ckb%20(Sorani)-1f6feb)
![Runtime](https://img.shields.io/badge/ONNX-CPU-2ea043)

This is the free, open-source **edge** model. For higher-quality Sorani speech, try the
hosted version at **[vekol.krd](https://vekol.krd)** · part of **Vekol**, Revge's Kurdish AI hub.

</div>

---

> **Non-commercial (CC-BY-NC 4.0).** Built with the Piper VITS recipe and trained with audio
> from the Vekol server model (built on Coqui XTTS, non-commercial) — so this inherits
> non-commercial terms. See [`NOTICE`](NOTICE). Commercial use needs a model retrained from
> a permissively-licensed base on permissively-licensed audio.

## What it is

| | |
|---|---|
| Model | `vekol-tts-ckb-edge` |
| Language | Central Kurdish / Sorani (`ckb`), Arabic script |
| Voice | one male Sorani voice |
| Output | 22.05 kHz mono WAV |
| Architecture | VITS (Piper), fine-tuned two-stage, exported to ONNX |
| Size | `model.onnx`, ~77 MB |
| Weights | [RevgeAI/vekol-tts-ckb-edge](https://huggingface.co/RevgeAI/vekol-tts-ckb-edge) |

## Install

```bash
pip install onnxruntime numpy scipy huggingface_hub
```

The weights live on Hugging Face. The script downloads `model.onnx` on first run, or grab
it yourself:

```bash
huggingface-cli download RevgeAI/vekol-tts-ckb-edge model.onnx --local-dir .
```

## Usage

```bash
# one sentence
python3 vekol_tts.py "کوردستانم وڵاتێکە و نابێت بە چوار" out.wav

# interactive — type lines, empty line to quit
python3 vekol_tts.py
```

From Python:

```python
from vekol_tts import speak
speak("ڕێڤگە کۆمپانیایەکی ژیریی دەستکردە لە شاری هەولێر لە هەرێمی کوردستانی عێراق.", "revge.wav")
```

## Samples

In [`samples/`](samples/) — each sentence has the **edge** model (this repo) and a
**hosted, higher-quality** version (from [vekol.krd](https://vekol.krd)), so you can hear
the difference:

| Text | Edge — 22 kHz (this model) | Hosted — higher quality |
|------|------|------|
| ڕێڤگە کۆمپانیایەکی ژیریی دەستکردە لە شاری هەولێر… | `sample1.wav` | `sample1-hosted.wav` |
| کوردستانم وڵاتێکە و نابێت بە چوار | `sample2.wav` | `sample2-hosted.wav` |
| بەخێربێن بۆ ڤەکۆڵ… (paragraph) | `sample3-long.wav` | — |

The edge files run fully offline on CPU. The hosted files are higher quality (24 kHz) —
listen to them or try your own text at **[vekol.krd](https://vekol.krd)**.

## How it was built

Two stages with the Piper VITS recipe. First a base is fine-tuned on about 27 hours of real
Central Kurdish speech so it pronounces Sorani properly. Then it's fine-tuned on a single
speaker using a larger set of audio from the Vekol server model, for a consistent voice.
The generator is exported to ONNX at 22.05 kHz. The released model is a plain VITS in ONNX
with no dependency on the server model at runtime.

## Letters & text handling

The character map uses Sorani keheh `ک` (U+06A9) natively — no remapping needed. `vekol_tts.py`
normalizes typed input so nothing is dropped: it folds common variants onto the letters the
model knows (Arabic kaf `ك`→`ک`, heh-doachashmee `ھ`→`ه`, teh-marbuta `ة`→`ە`, alef-maksura
`ى`→`ی`, hamza variants, ZWNJ/tatweel), and **reads numbers as Kurdish words** (`٢٠٢٤` →
«دوو هەزار و بیست و چوار»). Any character still outside the map prints a warning instead of
vanishing silently. The model is character based: a pad id sits between every character.

## Running it elsewhere

`model.onnx` is a standard Piper/VITS ONNX graph (with its `model.onnx.json` config), so
ONNX Runtime can run it from C++, Rust, Go, Java, C#, JavaScript, Android, iOS or a
Raspberry Pi — for example through [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) or
the [Piper](https://github.com/OHF-Voice/piper1-gpl) runtime.

## Limitations

Output is 22 kHz: clear and accurate, but a small on-device model, not studio quality. One
voice, one language, neutral narration (no acted emotion). Long sentences are split at their
commas automatically, so paragraph-length text reads cleanly without trailing artifacts.

For higher, studio-grade quality and expressive delivery, use the hosted version at
[vekol.krd](https://vekol.krd).

## License & credits

CC-BY-NC 4.0 (non-commercial) — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). Built with
the [Piper](https://github.com/OHF-Voice/piper1-gpl) VITS training recipe.

```bibtex
@software{vekol_tts_ckb_edge,
  title        = {Vekol-TTS: Sorani (Central Kurdish) on-device TTS},
  author       = {Shvan, Darvan},
  organization = {Revge},
  year         = {2026},
  url          = {https://github.com/Revge/vekol-tts-ckb-edge}
}
```

<div align="center">

Built by **Darvan Shvan** · **[Revge](https://github.com/Revge)** · part of the **Vekol** hub

</div>
