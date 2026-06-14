<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/vekol-white.svg">
  <img src="assets/vekol-black.svg" alt="Vekol" width="300">
</picture>

# Vekol&nbsp;·&nbsp;TTS

### Sorani speech, on every device.

Offline **Central Kurdish (Sorani)** text-to-speech that runs on plain **CPU** —
no GPU, no internet. A small ONNX model that speaks Sorani in real time.

[![Higher quality](https://img.shields.io/badge/higher%20quality-vekol.krd-6f42c1)](https://vekol.krd)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/license-CC--BY--NC--4.0-555)](LICENSE)
[![Model](https://img.shields.io/badge/🤗-RevgeAI%2Fvekol--tts--ckb--edge-yellow)](https://huggingface.co/RevgeAI/vekol-tts-ckb-edge)
![Language](https://img.shields.io/badge/lang-ckb%20(Sorani)-1f6feb)
![Runtime](https://img.shields.io/badge/ONNX-CPU-2ea043)

This is the free, open-source **edge** model. For higher-quality Sorani speech, try the
hosted version at **[vekol.krd](https://vekol.krd)** · part of **Vekol**, Revge's Kurdish AI hub.

</div>

---

> **Non-commercial (CC-BY-NC 4.0).** Fine-tuned from Meta's MMS-TTS and trained with audio
> from the Vekol server model (built on Coqui XTTS), both non-commercial — so this inherits
> non-commercial terms. See [`NOTICE`](NOTICE). Commercial use needs a model retrained from
> a permissively-licensed base.

## What it is

| | |
|---|---|
| Model | `vekol-tts-ckb-edge` |
| Language | Central Kurdish / Sorani (`ckb`), Arabic script |
| Voice | one male Sorani voice |
| Output | 16 kHz mono WAV |
| Architecture | VITS, fine-tuned from `facebook/mms-tts-kmr-script_arabic`, exported to ONNX |
| Size | `model.onnx`, ~114 MB (≈36 M params) |
| Weights | [RevgeAI/vekol-tts-ckb-edge](https://huggingface.co/RevgeAI/vekol-tts-ckb-edge) |

## Install

```bash
pip install onnxruntime numpy scipy transformers huggingface_hub
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

| Text | Edge — 16 kHz (this model) | Hosted — higher quality |
|------|------|------|
| ڕێڤگە کۆمپانیایەکی ژیریی دەستکردە لە شاری هەولێر… | `sample1.wav` | `sample1-hosted.wav` |
| کوردستانم وڵاتێکە و نابێت بە چوار | `sample2.wav` | `sample2-hosted.wav` |

The edge files run fully offline on CPU. The hosted files are higher quality (24 kHz) —
listen to them or try your own text at **[vekol.krd](https://vekol.krd)**.

## How it was built

Two stages. First, MMS-TTS (`kmr-script_arabic`) is fine-tuned on about 27 hours of real
Central Kurdish speech so it pronounces Sorani properly. Then it's fine-tuned on a single
speaker using a larger set of audio from the Vekol server model, for a consistent voice.
The generator is exported to ONNX at 16 kHz. The released model is a plain VITS in ONNX
with no dependency on the server model at runtime.

## The letter ک

The model's vocabulary uses Arabic kaf `ك` (U+0643), not Sorani keheh `ک` (U+06A9), so the
script replaces `ک` with `ك` before tokenizing — `vekol_tts.py` handles this. If you write
your own tokenizer, keep that replacement, or words with `ک` lose the letter. The tokenizer
is character based: the pad token `|` (id 0) sits between every character. `transformers`
is used only for this step and can be reimplemented from `vocab.json`.

## Running it elsewhere

`model.onnx` is a standard ONNX graph, so ONNX Runtime can run it from C++, Rust, Go, Java,
C#, JavaScript, Android, iOS or a Raspberry Pi — for example through
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx).

## Limitations

Output is 16 kHz: clear and accurate, but not studio quality. One voice, one language.
Numbers and abbreviations are read as written, so spell them out, and end sentences with
`.`, `؟` or `!` for better phrasing.

For higher, studio-grade quality, use the hosted version at [vekol.krd](https://vekol.krd).

## License & credits

CC-BY-NC 4.0 (non-commercial) — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). Built on
Meta's MMS-TTS and the [finetune-hf-vits](https://github.com/ylacombe/finetune-hf-vits) recipe.

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
