#!/usr/bin/env python3
"""ASR 4-engine WAV benchmark.

Usage:
    python asr_benchmark_wav.py [--engines ENGINE,...] [--langs LANG,...] [--wav-dir DIR]

Engines (comma-separated, default: qwen3-asr,sensevoice,funasr-nano):
    qwen3-asr       Qwen3-ASR-1.7B (ONNX+GGUF)
    sensevoice      SenseVoice Small
    funasr-nano     Fun-ASR-Nano-2512

Languages (comma-separated, default: zh,en,ja):
    zh  Chinese
    en  English
    ja  Japanese

The script downloads a small set of reference WAV clips on first run (stored in
tests/asr_bench_clips/) and prints per-clip transcription + RTF + CER table.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

CLIPS: list[dict] = [
    {
        "lang": "zh",
        "id": "aishell_S0764_BAC009S0764W0121",
        "ref": "甚至出现交易几乎停滞的情况",
        "url": "https://www.openslr.org/resources/33/data_aishell/wav/test/S0764/BAC009S0764W0121.wav",
        "local": "zh_aishell_BAC009S0764W0121.wav",
    },
    {
        "lang": "zh",
        "id": "aishell_S0764_BAC009S0764W0122",
        "ref": "一些公司的股价甚至翻了一番",
        "url": "https://www.openslr.org/resources/33/data_aishell/wav/test/S0764/BAC009S0764W0122.wav",
        "local": "zh_aishell_BAC009S0764W0122.wav",
    },
    {
        "lang": "en",
        "id": "librispeech_1089-134686-0000",
        "ref": "he hoped there would be stew for dinner turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick peppered flour fattened sauce",
        "url": "https://www.openslr.org/resources/12/test-clean/1089/134686/1089-134686-0000.flac",
        "local": "en_libri_1089-134686-0000.wav",
    },
    {
        "lang": "en",
        "id": "librispeech_1089-134686-0001",
        "ref": "stuff it into you his belly counselled him",
        "url": "https://www.openslr.org/resources/12/test-clean/1089/134686/1089-134686-0001.flac",
        "local": "en_libri_1089-134686-0001.wav",
    },
    {
        "lang": "ja",
        "id": "jsut_BASIC5000_0001",
        "ref": "水をマレーシアから買わなければならないのです",
        "url": "https://ss-takashi.sakura.ne.jp/corpus/jsut_ver1.1/basic5000/wav/BASIC5000_0001.wav",
        "local": "ja_jsut_BASIC5000_0001.wav",
    },
    {
        "lang": "ja",
        "id": "jsut_BASIC5000_0002",
        "ref": "もちろん私はヴィランではありません",
        "url": "https://ss-takashi.sakura.ne.jp/corpus/jsut_ver1.1/basic5000/wav/BASIC5000_0002.wav",
        "local": "ja_jsut_BASIC5000_0002.wav",
    },
]

BENCH_DIR = Path(__file__).parent / "tests" / "asr_bench_clips"


def _load_wav_as_f32(path: Path) -> np.ndarray:
    import soundfile as sf
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        except ImportError:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(16000, sr)
            audio = resample_poly(audio, 16000 // g, sr // g).astype(np.float32)
    return audio


def _fetch_clip(clip: dict) -> Path:
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    local = BENCH_DIR / clip["local"]
    if local.exists():
        return local
    url = clip["url"]
    print(f"  Downloading {url} ...", flush=True)
    try:
        urllib.request.urlretrieve(url, local)
    except Exception as exc:
        print(f"  WARN: download failed ({exc}), skipping {clip['id']}", file=sys.stderr)
        return None
    return local


def _load_sensevoice(device: str = "cpu") -> Any:
    from model_manager import apply_cache_env
    apply_cache_env()
    from asr_sensevoice import SenseVoiceEngine
    return SenseVoiceEngine(device=device, hub="ms")


def _load_funasr_nano(device: str = "cpu") -> Any:
    from model_manager import apply_cache_env
    apply_cache_env()
    from asr_funasr_nano import FunASRNanoEngine
    return FunASRNanoEngine(device=device, hub="ms", engine_type="funasr-nano")


def _load_qwen3_asr(device: str = "cpu") -> Any:
    from model_manager import apply_cache_env, get_qwen3_asr_model_dir
    apply_cache_env()
    from asr_qwen3 import Qwen3ASREngine
    return Qwen3ASREngine(model_dir=get_qwen3_asr_model_dir(), use_dml=False, chunk_size=10.0)


ENGINE_LOADERS = {
    "qwen3-asr": _load_qwen3_asr,
    "sensevoice": _load_sensevoice,
    "funasr-nano": _load_funasr_nano,
}

ENGINE_DISPLAY = {
    "qwen3-asr": "Qwen3-ASR-1.7B",
    "sensevoice": "SenseVoice Small",
    "funasr-nano": "Fun-ASR-Nano",
}


def _char_error_rate(ref: str, hyp: str) -> float:
    ref = ref.replace(" ", "")
    hyp = hyp.replace(" ", "")
    if not ref:
        return 0.0 if not hyp else 1.0
    m, n = len(ref), len(hyp)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n] / len(ref)


def run_benchmark(engines: list[str], langs: list[str], device: str = "cpu"):
    clips = [c for c in CLIPS if c["lang"] in langs]

    print(f"\n{'=' * 70}")
    print(f"ASR Benchmark  engines={engines}  langs={langs}  clips={len(clips)}")
    print(f"{'=' * 70}\n")

    print("Fetching reference clips...")
    available_clips = []
    for clip in clips:
        p = _fetch_clip(clip)
        if p is not None:
            available_clips.append((clip, p))
    print(f"Ready: {len(available_clips)}/{len(clips)} clips\n")

    if not available_clips:
        print("No clips available.")
        return

    results: dict[str, list[dict]] = {e: [] for e in engines}

    for engine_key in engines:
        display = ENGINE_DISPLAY.get(engine_key, engine_key)
        print(f"{'─' * 70}")
        print(f"Loading engine: {display}")
        loader = ENGINE_LOADERS.get(engine_key)
        if loader is None:
            print(f"  Unknown engine '{engine_key}', skipping.")
            continue

        try:
            t0 = time.perf_counter()
            engine = loader(device)
            load_ms = (time.perf_counter() - t0) * 1000
            print(f"  Load time: {load_ms:.0f} ms\n")
        except Exception as exc:
            print(f"  FAILED to load: {exc}\n", file=sys.stderr)
            continue

        for clip, path in available_clips:
            try:
                audio = _load_wav_as_f32(path)
            except Exception as exc:
                print(f"  [{clip['id']}] audio load error: {exc}", file=sys.stderr)
                continue

            dur_s = len(audio) / 16000
            lang = clip["lang"]
            if hasattr(engine, "set_language"):
                engine.set_language(lang)

            t0 = time.perf_counter()
            try:
                res = engine.transcribe(audio)
            except Exception as exc:
                print(f"  [{clip['id']}] transcribe error: {exc}", file=sys.stderr)
                res = None
            elapsed = time.perf_counter() - t0
            rtf = elapsed / dur_s if dur_s > 0 else 0.0

            hyp = (res.get("text") or "").strip() if res else ""
            ref = clip["ref"]
            cer = _char_error_rate(ref, hyp)

            results[engine_key].append({
                "id": clip["id"], "lang": lang, "ref": ref, "hyp": hyp,
                "dur_s": dur_s, "elapsed_s": elapsed, "rtf": rtf, "cer": cer,
            })

            print(f"  [{lang}] {clip['id']}")
            print(f"    REF: {ref}")
            print(f"    HYP: {hyp or '(empty)'}")
            print(f"    RTF={rtf:.2f}  CER={cer:.1%}  dur={dur_s:.1f}s  t={elapsed*1000:.0f}ms\n")

        if hasattr(engine, "unload"):
            try:
                engine.unload()
            except Exception:
                pass

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'Engine':<22} {'Lang':<5} {'Clips':<6} {'Avg RTF':<10} {'Avg CER'}")
    print(f"{'─' * 22} {'─' * 5} {'─' * 6} {'─' * 10} {'─' * 8}")
    for engine_key in engines:
        rows = results.get(engine_key, [])
        if not rows:
            print(f"{ENGINE_DISPLAY.get(engine_key, engine_key):<22} {'—':<5} 0")
            continue
        display = ENGINE_DISPLAY.get(engine_key, engine_key)
        for lang in langs:
            lang_rows = [r for r in rows if r["lang"] == lang]
            if not lang_rows:
                continue
            avg_rtf = sum(r["rtf"] for r in lang_rows) / len(lang_rows)
            avg_cer = sum(r["cer"] for r in lang_rows) / len(lang_rows)
            print(f"{display:<22} {lang:<5} {len(lang_rows):<6} {avg_rtf:<10.3f} {avg_cer:.1%}")
    print(f"{'=' * 70}\n")


def main():
    parser = argparse.ArgumentParser(description="ASR multi-engine WAV benchmark")
    parser.add_argument("--engines", default="qwen3-asr,sensevoice,funasr-nano")
    parser.add_argument("--langs", default="zh,en,ja")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--wav-dir", default=None)
    args = parser.parse_args()

    if args.wav_dir:
        global BENCH_DIR
        BENCH_DIR = Path(args.wav_dir)

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    langs = [l.strip() for l in args.langs.split(",") if l.strip()]

    repo_root = str(Path(__file__).parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    run_benchmark(engines=engines, langs=langs, device=args.device)


if __name__ == "__main__":
    main()
