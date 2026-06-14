#!/usr/bin/env python3
"""Vekol-TTS — Sorani (Central Kurdish, ckb) text-to-speech, edge build.

Tiny offline voice that runs on CPU (no GPU, no internet after the model is local).
Model = model.onnx (~114 MB, 16 kHz). Part of the Vekol hub by Revge.

Usage:
    python3 vekol_tts.py "ئەمڕۆ کەشەکە خۆشە" out.wav
    python3 vekol_tts.py            # interactive

Deps:  pip install onnxruntime numpy scipy transformers huggingface_hub
(onnxruntime runs the model; transformers is used only for the char tokenizer.)

The model weights (model.onnx) are hosted on Hugging Face: RevgeAI/vekol-tts-ckb-edge.
If model.onnx isn't next to this script, it's downloaded automatically on first run.
"""
import os, sys, re
import numpy as np
import onnxruntime as ort
import scipy.io.wavfile
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
SR = 16000
HF_REPO = "RevgeAI/vekol-tts-ckb-edge"


def _model_path():
    local = os.path.join(HERE, "model.onnx")
    if os.path.exists(local):
        return local
    from huggingface_hub import hf_hub_download
    print("model.onnx not found locally — downloading from", HF_REPO, "...")
    return hf_hub_download(repo_id=HF_REPO, filename="model.onnx")


_tok = AutoTokenizer.from_pretrained(HERE)
_sess = ort.InferenceSession(_model_path(), providers=["CPUExecutionProvider"])


def speak(text, out="out.wav"):
    # Sorani keheh 'ک'(U+06A9) -> Arabic kaf 'ك'(U+0643): the model's vocab uses kaf;
    # without this, every ک-word loses the letter. (Matches training.)
    text = text.replace("ک", "ك")
    # split long text at punctuation so paragraphs read naturally; stitch with a short pause
    chunks = [c.strip() for c in re.split(r"[.،؛؟!\n]+", text) if c.strip()] or [text]
    gap = np.zeros(int(0.18 * SR), dtype="float32")
    pieces = []
    for c in chunks:
        ids = _tok(c, return_tensors="np")["input_ids"].astype(np.int64)
        pieces.append(_sess.run(None, {"input_ids": ids})[0][0].astype("float32"))
        pieces.append(gap)
    wav = np.concatenate(pieces[:-1]) if len(pieces) > 1 else pieces[0]
    scipy.io.wavfile.write(out, SR, wav)
    return out, len(wav) / SR


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        out = sys.argv[2] if len(sys.argv) > 2 else "out.wav"
        path, dur = speak(sys.argv[1], out)
        print(f"wrote {path}  ({dur:.1f}s)")
    else:
        print("Vekol-TTS (Sorani) — type text, blank line to quit:")
        n = 1
        while True:
            try:
                t = input("> ").strip()
            except EOFError:
                break
            if not t:
                break
            path, dur = speak(t, f"out_{n}.wav")
            print(f"  wrote {path}  ({dur:.1f}s)")
            n += 1
