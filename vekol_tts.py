#!/usr/bin/env python3
"""Vekol-TTS — Sorani (Central Kurdish, ckb) text-to-speech, edge build.

Tiny offline voice that runs on CPU (no GPU, no internet after the model is local).
Model = model.onnx (~77 MB, 22.05 kHz, Piper/VITS). Part of the Vekol hub by Revge.

Usage:
    python3 vekol_tts.py "ئەمڕۆ کەشەکە خۆشە" out.wav
    python3 vekol_tts.py            # interactive

Deps:  pip install onnxruntime numpy scipy huggingface_hub
(onnxruntime runs the model; no heavyweight ML deps — the tokenizer is the
character map shipped in model.onnx.json.)

The model weights (model.onnx) are hosted on Hugging Face: RevgeAI/vekol-tts-ckb-edge.
If model.onnx isn't next to this script, it's downloaded automatically on first run.
"""
import os, re, sys, json, unicodedata
import numpy as np
import onnxruntime as ort
import scipy.io.wavfile
from scipy.signal import stft, istft

HERE = os.path.dirname(os.path.abspath(__file__))
HF_REPO = "RevgeAI/vekol-tts-ckb-edge"
SCALES = [0.667, 1.0, 0.0]   # [noise_scale, length_scale, noise_w] — accurate, deterministic


def _asset(name):
    local = os.path.join(HERE, name)
    if os.path.exists(local):
        return local
    from huggingface_hub import hf_hub_download
    print(f"{name} not found locally — downloading from {HF_REPO} ...")
    return hf_hub_download(repo_id=HF_REPO, filename=name)


_cfg = json.load(open(_asset("model.onnx.json"), encoding="utf-8"))
_pm = _cfg["phoneme_id_map"]
SR = _cfg["audio"]["sample_rate"]
_sess = ort.InferenceSession(_asset("model.onnx"), providers=["CPUExecutionProvider"])
PAD, BOS, EOS = _pm["_"][0], _pm["^"][0], _pm["$"][0]

# --- text normalization: fold typed variants onto in-map letters so nothing drops ---
_NORM = {
    "ك": "ک",                                   # Arabic kaf -> Sorani keheh
    "ھ": "ه", "ہ": "ه", "ۀ": "ە", "ة": "ە",     # heh variants / teh-marbuta
    "ى": "ی", "ﻯ": "ی", "ﺉ": "ئ", "ٸ": "ئ",
    "ؤ": "و", "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ڭ": "گ",
    "‌": "", "‍": "", "ـ": "",         # ZWNJ/ZWJ -> join, tatweel -> drop
    "“": '"', "”": '"', "’": "'", "‘": "'", "—": "-", "–": "-", "…": ".",
}
_DIG = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_ONES = ["", "یەک", "دوو", "سێ", "چوار", "پێنج", "شەش", "حەوت", "هەشت", "نۆ"]
_TEENS = ["دە", "یازدە", "دوازدە", "سێزدە", "چواردە", "پازدە", "شازدە", "حەڤدە", "هەژدە", "نۆزدە"]
_TENS = ["", "", "بیست", "سی", "چل", "پەنجا", "شەست", "حەفتا", "هەشتا", "نەوەد"]


def _u100(n):
    if n < 10: return _ONES[n]
    if n < 20: return _TEENS[n - 10]
    t, r = _TENS[n // 10], n % 10
    return t if r == 0 else f"{t} و {_ONES[r]}"


def _u1000(n):
    if n < 100: return _u100(n)
    h, r = n // 100, n % 100
    hw = "سەد" if h == 1 else f"{_ONES[h]}سەد"
    return hw if r == 0 else f"{hw} و {_u100(r)}"


def _spell(n):
    if n == 0: return "سفر"
    parts = []
    for div, word in ((1_000_000_000, "ملیار"), (1_000_000, "ملیۆن"), (1000, "هەزار")):
        if n >= div:
            q, n = divmod(n, div)
            parts.append(word if q == 1 else f"{_u1000(q)} {word}")
    if n: parts.append(_u1000(n))
    return " و ".join(parts)


def normalize(text):
    text = "".join(_NORM.get(c, c) for c in text).translate(_DIG)
    return re.sub(r"\d+", lambda m: f" {_spell(int(m.group()))} ", text)


def _ids(text):
    seq = [BOS, PAD]
    for ch in unicodedata.normalize("NFD", normalize(text)):
        if ch in _pm:
            seq += [_pm[ch][0], PAD]
        elif not unicodedata.combining(ch):
            sys.stderr.write(f"[warn] dropped '{ch}' U+{ord(ch):04X} (not in map)\n")
    return np.array([seq + [EOS]], dtype=np.int64)


def _denoise(a, reduce_db=16):
    """Spectral noise-gate: attenuate the vocoder's grainy background hiss."""
    _, _, Z = stft(a, SR, nperseg=1024, noverlap=768)
    mag, ph = np.abs(Z), np.angle(Z)
    noise = np.percentile(mag, 10, axis=1, keepdims=True)
    g = 10 ** (-reduce_db / 20)
    mask = g + (1 - g) * np.clip((mag / (noise + 1e-9) - 1.0) / 2.0, 0, 1)
    _, y = istft(mag * mask * np.exp(1j * ph), SR, nperseg=1024, noverlap=768)
    return (y[:len(a)] / (np.abs(y).max() or 1))


def _trim_tail(a, gap=0.35, tail=0.20, th=0.02):
    """Drop a trailing blob sitting after a long silence (a hallucinated tail)."""
    env = np.convolve(np.abs(a), np.ones(int(0.02 * SR)) / int(0.02 * SR), "same")
    loud = env > th * (env.max() or 1)
    if not loud.any():
        return a
    last = np.where(loud)[0][-1]
    i = last
    while i > 0:
        if not loud[i]:
            j = i
            while j > 0 and not loud[j]:
                j -= 1
            if (i - j) / SR > gap:
                last = j
            i = j
        else:
            i -= 1
    return a[:min(len(a), last + int(tail * SR))]


def speak(text, out="out.wav"):
    ids = _ids(text)
    a = _sess.run(None, {
        "input": ids,
        "input_lengths": np.array([ids.shape[1]], dtype=np.int64),
        "scales": np.array(SCALES, dtype=np.float32),
        "sid": np.array([0], dtype=np.int64),
    })[0].squeeze()
    a = a / (np.abs(a).max() or 1)
    a = _trim_tail(_denoise(a))
    scipy.io.wavfile.write(out, SR, (a * 32767).astype(np.int16))
    return out, len(a) / SR


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
