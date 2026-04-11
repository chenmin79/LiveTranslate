# Qwen3-ASR Troubleshooting

## Summary

This note records a real startup failure we hit while enabling `Qwen3-ASR` in `LiveTranslate`, and the fix that made the engine load successfully again.

The failure was **not** caused by `main.py`, audio capture, VAD, or translation.
The direct cause was a **mixed-version llama.cpp runtime** inside `qwen_asr_gguf/inference/bin`.

## Symptom

`Qwen3-ASR` failed during initialization with an access violation like this:

```text
Model load failed: D:\work\prj\LiveTranslate\models\qwen3-asr\qwen3_asr_llm.q4_k.gguf
OSError: exception: access violation reading 0x0000000000000034
```

Typical stack:

- `asr_qwen3.py`
- `qwen_asr_gguf/inference/asr.py`
- `qwen_asr_gguf/inference/llama.py`
- crash around `llama_model_n_embd(self.ptr)`

## What We Verified

### 1. The GGUF model file was not obviously corrupted

We verified that Python could read the GGUF header and metadata normally.
The file contained valid metadata and tensors, so the model file itself was unlikely to be the main problem.

Relevant file:

- `models/qwen3-asr/qwen3_asr_llm.q4_k.gguf`

### 2. The crash reproduced outside the main app

A standalone test script reproduced the same failure without running the full UI pipeline.
That proved the problem was below the application layer.

Standalone test script:

- `D:\work\prj\realtimeTranslator\test_qwen_asr_standalone.py`

### 3. The failure was below ASR business logic

Even directly constructing `qwen_asr_gguf.inference.llama.LlamaModel(...)` reproduced the same failure.
That narrowed the issue to the local GGUF runtime layer.

## Root Cause

`LiveTranslate` was mixing **different versions** of llama.cpp / ggml runtime DLLs.

The project had already downloaded one set of DLLs into:

- `qwen_asr_gguf/inference/bin`

Later, only part of that runtime was replaced. For example, some of these files were updated:

- `llama.dll`
- `ggml.dll`
- `ggml-base.dll`
- `ggml-vulkan.dll`

But other related DLLs still came from a different version, such as:

- `ggml-cpu-*.dll`
- `ggml-rpc.dll`

This mixed runtime caused the low-level loader / binding layer to behave incorrectly and eventually crash with an access violation.

## Why This Happened

`model_manager.py` downloads a llama.cpp Windows package for Qwen3-ASR.
The risky part is that runtime binaries may come from a newer release than the version expected by the bundled `qwen_asr_gguf/inference/llama.py` ctypes bindings.

Relevant code:

- `model_manager.py`
- `qwen_asr_gguf/inference/llama.py`

The project currently uses the latest release lookup first, with fallback code mentioning `b8391`.
If only part of the DLL set is replaced, version skew is very easy to introduce.

## Fix That Worked

The working fix was:

1. Download a single pinned llama.cpp Windows package.
2. Extract it.
3. Replace the **entire** DLL set in `qwen_asr_gguf/inference/bin` with files from the same package.
4. Do not mix DLLs from different llama.cpp releases.

The pinned version that worked in this environment was:

- `b8391`

After replacing the whole DLL set with the `b8391` package, Qwen3-ASR initialized successfully again.

## Important Detail

Replacing only these files was **not enough**:

- `llama.dll`
- `ggml.dll`
- `ggml-base.dll`
- `ggml-vulkan.dll`

The successful repair required replacing the rest of the DLL family too, including:

- `ggml-cpu-alderlake.dll`
- `ggml-cpu-cannonlake.dll`
- `ggml-cpu-cascadelake.dll`
- `ggml-cpu-cooperlake.dll`
- `ggml-cpu-haswell.dll`
- `ggml-cpu-icelake.dll`
- `ggml-cpu-ivybridge.dll`
- `ggml-cpu-piledriver.dll`
- `ggml-cpu-sandybridge.dll`
- `ggml-cpu-sapphirerapids.dll`
- `ggml-cpu-skylakex.dll`
- `ggml-cpu-sse42.dll`
- `ggml-cpu-x64.dll`
- `ggml-cpu-zen4.dll`
- `ggml-rpc.dll`
- `ggml-vulkan.dll`
- `llama.dll`
- `mtmd.dll`

In practice, the safe rule is:

> Treat `qwen_asr_gguf/inference/bin/*.dll` as one versioned unit.

## Validation Steps

The fix was validated in three layers.

### Layer 1: Standalone initialization

`test_qwen_asr_standalone.py` succeeded with:

- `--device cpu --no-vulkan`
- `--device cpu`
- `--device dml`

### Layer 2: Main application startup

`LiveTranslate` logs showed successful initialization:

```text
Qwen3-ASR loaded: D:\work\prj\LiveTranslate\models\qwen3-asr (DML=False)
Qwen3-ASR language: None -> en
ASR engine ready: qwen3-asr on cpu
```

### Layer 3: Pipeline readiness

The main app no longer crashed immediately during the Qwen3-ASR load path.
It reached the normal running state after ASR initialization.

## Recommended Operational Rule

For future maintenance, do this when touching Qwen3-ASR runtime binaries:

1. Pin a llama.cpp release version.
2. Replace the full DLL set together.
3. Avoid partial upgrades of only a few DLLs.
4. If Qwen3-ASR starts crashing with `access violation`, first suspect DLL version skew before suspecting the GGUF model file.

## Suggested Future Improvement

A small future improvement would be to make `model_manager.py` install a **fully pinned** llama.cpp package for Qwen3-ASR instead of preferring the latest release.

That would reduce the chance of runtime drift between:

- `qwen_asr_gguf/inference/llama.py`
- downloaded `llama.cpp` DLLs
- the shipped GGUF model format expectations

## Related Paths

- `main.py`
- `asr_qwen3.py`
- `model_manager.py`
- `qwen_asr_gguf/inference/asr.py`
- `qwen_asr_gguf/inference/llama.py`
- `qwen_asr_gguf/inference/bin`
- `models/qwen3-asr`
