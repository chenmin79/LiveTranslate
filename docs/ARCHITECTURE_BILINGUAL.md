# LiveTranslate Architecture for Bilingual Overlay

## Scope

This document is based on the current repository code, with emphasis on these files:

- `main.py`
- `audio_capture.py`
- `vad_processor.py`
- `translator.py`
- `subtitle_overlay.py`
- `subtitle_window.py`
- `subtitle_settings.py`
- `asr_engine.py`
- `asr_sensevoice.py`
- `asr_funasr_nano.py`
- `asr_qwen3.py`

The goal here is to describe the architecture as it exists today and identify the smallest future extension points for an `English + Chinese bilingual floating overlay`, without changing current behavior.

---

## 1. Main Execution Chain

The runtime entry point is `main()` in `main.py`.

At startup, `main()` does the following:

1. Initializes logging with `setup_logging()`.
2. Loads static config through `load_config()`.
3. Loads persisted user settings through `control_panel._load_saved_settings()`.
4. Creates the Qt application and top-level UI objects:
   - `ControlPanel`
   - `SubtitleOverlay`
   - `SubtitleWindow`
   - `LogWindow`
5. Creates the runtime coordinator `LiveTranslateApp`.
6. Injects UI dependencies into `LiveTranslateApp` through:
   - `LiveTranslateApp.set_overlay()`
   - `LiveTranslateApp.set_subtitle_window()`
   - `LiveTranslateApp.set_panel()`
7. Connects tray actions and overlay signals to lifecycle operations such as start, pause, resume, hide, and quit.
8. Calls `on_start()` via `QTimer.singleShot(500, on_start)`, which invokes `LiveTranslateApp.start()`.

### Runtime pipeline inside `LiveTranslateApp`

`LiveTranslateApp` in `main.py` is the central orchestrator. Its effective runtime chain is:

`AudioCapture -> VADProcessor -> ASR queue -> ASR engine -> Translator -> SubtitleOverlay / SubtitleWindow`

More concretely:

1. `LiveTranslateApp.start()`
   - starts `AudioCapture.start()`
   - starts `_capture_thread` running `_capture_loop()`
   - starts `_asr_thread` running `_asr_loop()`

2. `LiveTranslateApp._capture_loop()`
   - pulls audio chunks from `AudioCapture.get_audio()`
   - pushes them into `VADProcessor.process_chunk()`
   - when speech is not yet complete, it may trigger incremental recognition by enqueueing an `"interim"` task
   - when speech becomes a segment, it enqueuees a `"vad_flush"` task

3. `LiveTranslateApp._asr_loop()`
   - consumes the ASR queue
   - routes `"vad_flush"` to:
     - `_process_segment()` if there was no prior interim output
     - `_process_interim_final()` if incremental ASR already emitted earlier sentences
   - routes `"interim"` to `_do_interim_asr()`

4. After ASR text is available
   - `_process_segment()`, `_process_segment_text()`, or `_process_interim_final()` decides whether translation is needed
   - if translation is needed, it submits `_translate_async()` to the thread pool executor
   - if translation is not needed, it updates UI directly

5. `_translate_async()`
   - uses `Translator.translate_iter()`
   - streams partial translation text to the overlay
   - emits the final translation to the overlay
   - updates `SubtitleWindow` with a translation dictionary

---

## 2. Where Audio Enters the System

Audio enters through `AudioCapture` in `audio_capture.py`.

### Primary entry point

- Class: `AudioCapture`
- Main methods:
  - `start()`
  - `_read_loop()`
  - `get_audio()`

### Input sources

`AudioCapture` currently supports two inputs:

1. System playback audio
   - captured through WASAPI loopback
   - device discovery is handled by `_find_loopback_device()`
   - stream opening is handled by `_open_stream()`

2. Optional microphone audio
   - selected via `set_mic_device()`
   - stream opening is handled by `_open_mic_stream()`

### How audio is normalized

Inside `_read_loop()`:

- loopback audio is read from PyAudioWPatch
- raw bytes are converted in `_resample_to_mono()`
- output is normalized to:
  - `float32`
  - mono
  - target sample rate `self.sample_rate`, currently driven from config as 16kHz

If microphone is enabled:

- mic data is also resampled to the same mono 16kHz format
- mic data is buffered in `self._mic_buf`
- loopback and mic chunks are mixed by summation before enqueueing

### Handoff to the rest of the app

The only public handoff from audio capture is:

- `AudioCapture.get_audio(timeout=1.0)`

This returns a tuple:

- `(audio_chunk, mic_rms)`

That tuple is consumed in `LiveTranslateApp._capture_loop()` in `main.py`.

So the actual audio ingress boundary for the pipeline is:

- producer: `AudioCapture._read_loop()`
- queue: `AudioCapture.audio_queue`
- consumer: `LiveTranslateApp._capture_loop()`

---

## 3. Where ASR Is Processed

ASR is coordinated in `main.py`, but implemented by pluggable engine classes.

### ASR coordination layer

The ASR coordination lives in `LiveTranslateApp`:

- `_switch_asr_engine()`
- `_asr_loop()`
- `_process_segment()`
- `_do_interim_asr()`
- `_process_segment_text()`
- `_process_interim_final()`

### VAD stage before ASR

Before ASR receives anything, audio goes through `VADProcessor` in `vad_processor.py`.

Relevant methods:

- `process_chunk()`
- `peek_buffer()`
- `trim_front()`
- `flush()`
- `force_flush()`

`VADProcessor` is responsible for:

- deciding when speech starts and stops
- buffering speech chunks
- splitting long speech at a likely pause with `_split_at_best_pause()`
- supporting incremental ASR by exposing `peek_buffer()` and `trim_front()`

This means ASR does not read raw audio directly from `AudioCapture`; it only sees speech segments that passed through VAD.

### ASR queue boundary

ASR work is serialized through `self._asr_queue` in `LiveTranslateApp`.

Items are enqueued by:

- `_enqueue_asr("vad_flush", speech_segment)`
- `_enqueue_asr("interim", None)`

Items are consumed by:

- `_asr_loop()`

### Actual ASR engines

The currently supported engine classes are:

- `ASREngine` in `asr_engine.py`
  - wraps `faster_whisper.WhisperModel`
  - main method: `transcribe(audio, word_timestamps=False)`

- `SenseVoiceEngine` in `asr_sensevoice.py`
  - wraps `funasr.AutoModel`
  - main method: `transcribe(audio)`

- `FunASRNanoEngine` in `asr_funasr_nano.py`
  - wraps `funasr.AutoModel`
  - main method: `transcribe(audio)`

- `Qwen3ASREngine` in `asr_qwen3.py`
  - wraps `qwen_asr_gguf.inference.asr.QwenASREngine`
  - main method: `transcribe(audio)`

Engine switching happens in `LiveTranslateApp._switch_asr_engine()`.

### ASR result shape

All engines normalize their output to the same shape:

- `text`
- `language`
- `language_name`

Whisper may also return:

- `words`

This common contract is what allows `LiveTranslateApp` to switch engines with minimal branching.

---

## 4. Where Translation Happens

Translation happens after ASR text is produced and filtered, not during audio or VAD processing.

### Translation coordinator

The translation decision point is in:

- `LiveTranslateApp._process_segment()`
- `LiveTranslateApp._process_segment_text()`
- `LiveTranslateApp._process_interim_final()`

These functions:

1. receive recognized text
2. filter empty/noisy results
3. determine `source_lang`
4. compare `source_lang` against the current `target_language`
5. dispatch translation if needed

### Translation worker

Actual translation is done in:

- `LiveTranslateApp._translate_async()`

This method:

1. calls `self._translator.translate_iter(text, source_lang)`
2. streams partial outputs to `SubtitleOverlay.update_streaming()`
3. records token usage and timing
4. sends final translation to:
   - `SubtitleOverlay.update_translation()`
   - `SubtitleWindow.update_text()`

### Translator implementation

The translation client lives in `translator.py`.

Relevant class and methods:

- `Translator`
  - `translate()`
  - `translate_iter()`
  - `with_target_language()`
  - `_translate_sync()`
  - `_translate_streaming()`

Key points:

- it uses an OpenAI-compatible chat completion API
- it supports streaming and non-streaming modes
- it builds prompts through `_build_system_prompt()`
- it builds message payloads through `_build_messages()`
- it supports optional context history

### Multi-language subtitle window support

`SubtitleWindow` already supports more than one translation target.

This is handled by:

- `LiveTranslateApp._translate_extra_langs()`
- `LiveTranslateApp._translate_subwin_only()`

The logic is:

- primary target language uses `self._translator`
- additional subtitle-window languages are translated in parallel using `Translator.with_target_language()`
- the resulting translations are assembled into a dictionary
- that dictionary is passed to `SubtitleWindow.update_text(original, tl_dict)`

This is important because it means the project already has a translation dictionary model for one UI path.

---

## 5. How the Overlay Subtitle Layer Updates

The floating overlay is implemented by `SubtitleOverlay` in `subtitle_overlay.py`.

### Current overlay data model

The overlay is message-based, not line-based.

Each ASR segment becomes one `ChatMessage`, created by:

- `SubtitleOverlay._on_add_message()`

Each `ChatMessage` stores:

- original text
- source language
- translation text
- timestamp
- ASR timing
- translation timing

### Thread-safe update API

`SubtitleOverlay` exposes thread-safe methods that emit Qt signals:

- `add_message(msg_id, timestamp, original, source_lang, asr_ms)`
- `update_translation(msg_id, translated, translate_ms)`
- `update_streaming(msg_id, partial_text)`
- `update_monitor(rms, vad_conf, mic_rms=None)`
- `update_stats(...)`
- `update_asr_device(device)`

These are called from worker threads in `LiveTranslateApp`.

They are received on the UI side by:

- `_on_add_message()`
- `_on_update_translation()`
- `_on_update_streaming()`
- `_on_update_monitor()`
- `_on_update_stats()`
- `_on_update_asr_device()`

### Overlay update flow

The actual flow for one recognized sentence is:

1. `LiveTranslateApp._process_segment()` calls:
   - `self._overlay.add_message(...)`

2. `SubtitleOverlay.add_message()` emits `add_message_signal`

3. `SubtitleOverlay._on_add_message()` creates a `ChatMessage`

4. During translation:
   - `LiveTranslateApp._translate_async()` repeatedly calls `self._overlay.update_streaming(msg_id, partial)`

5. `SubtitleOverlay._on_update_streaming()` calls `ChatMessage.update_streaming(partial_text)`

6. After translation completes:
   - `LiveTranslateApp._translate_async()` calls `self._overlay.update_translation(msg_id, translated, tl_ms)`

7. `SubtitleOverlay._on_update_translation()` calls `ChatMessage.set_translation(translated, translate_ms)`

### Important architectural point

The current floating overlay is not the same thing as `SubtitleWindow`.

- `SubtitleOverlay` is a chat/log style floating control overlay
- `SubtitleWindow` is a clean OBS-friendly subtitle renderer

This matters for future bilingual overlay work because there are two possible product directions:

1. extend the current chat-style floating overlay
2. create or adapt a clean text-only floating bilingual layer using the subtitle-window style model

---

## 6. Smallest Invasive Change Points for a Future "English + Chinese Bilingual Floating Overlay"

This section is intentionally about minimal-intrusion extension points, not implementation.

### Current state

Today:

- `SubtitleOverlay` already stores original text and translated text in each `ChatMessage`
- `SubtitleWindow` already supports structured multi-line rendering with line configs
- `LiveTranslateApp` already knows both:
  - the original ASR text
  - the translated result

So the bilingual data itself already exists in the pipeline.

### Best minimal-intrusion strategy

The least invasive path is to extend the existing `SubtitleOverlay` display contract rather than altering audio, VAD, or ASR.

The smallest change points are:

1. `LiveTranslateApp._process_segment()` and `LiveTranslateApp._process_segment_text()`
   - these already send original text to `SubtitleOverlay.add_message()`
   - no audio/ASR contract changes are needed here

2. `LiveTranslateApp._translate_async()`
   - this already streams partial translation and final translation to the overlay
   - if a bilingual floating presentation is needed, this is the ideal place to keep the overlay fed with both original and translated text

3. `subtitle_overlay.py`, specifically:
   - `ChatMessage`
   - `SubtitleOverlay._on_add_message()`
   - `SubtitleOverlay._on_update_streaming()`
   - `SubtitleOverlay._on_update_translation()`

This is the most local UI-only change surface for a bilingual floating overlay.

### Why this is minimal

Because `ChatMessage` already has both layers:

- original line in `_header_label`
- translated line in `_trans_label`

In other words, the current overlay is already logically bilingual per message. The likely future requirement is not "add bilingual data", but "change the floating presentation style".

### Two realistic extension paths

#### Path A: Extend `SubtitleOverlay`

Use this if the target is:

- still a chat-style floating panel
- but more explicitly bilingual
- perhaps more compact and subtitle-like

Minimal touch points:

- `ChatMessage` rendering
- style configuration passed through `SubtitleOverlay.apply_style()`
- optional new overlay mode in `DragHandle` / `SubtitleOverlay`

This path avoids changing `Translator`, `AudioCapture`, and the ASR engines.

#### Path B: Reuse `SubtitleWindow` concepts for a second floating layer

Use this if the target is:

- a clean bilingual floating subtitle bar
- original English on one line
- Chinese translation on the next line
- not a chat history panel

Minimal touch points:

- keep `LiveTranslateApp` producing `original + translation`
- introduce a new overlay-like window that follows `SubtitleWindow`'s line-config model
- reuse `SubtitleWindow.update_text(original, translations)` semantics

This path is slightly larger than Path A, but cleaner if the product requirement is "subtitle bar" rather than "message log".

### Strong recommendation

If the future feature is specifically "English + Chinese bilingual floating overlay", the least invasive architectural insertion point is the UI layer, not the pipeline layer.

That means:

- do not change `AudioCapture`
- do not change `VADProcessor`
- do not change ASR engine interfaces
- do not change `Translator` request/response shape first

Instead, treat the feature as a new presentation target fed by the existing:

- original ASR text
- final translated text
- optional streaming partial translation

---

## 7. Suggested New Modules for the Future, Without Implementing Them Yet

The current codebase already works, so new modules should mainly reduce coupling between pipeline logic and UI presentation.

### 1. `overlay_presenter.py`

Purpose:

- centralize how recognized text and translated text are converted into UI update events

Why:

- today `LiveTranslateApp` directly calls both `SubtitleOverlay` and `SubtitleWindow`
- that makes UI fan-out logic live inside the pipeline coordinator

Possible responsibility:

- accept events like `on_asr_result(...)` and `on_translation_result(...)`
- dispatch them to one or more display targets

### 2. `message_models.py`

Purpose:

- define stable data objects for:
  - ASR result
  - translation result
  - subtitle payload

Why:

- current code passes multiple primitive arguments around
- message identity, original text, source language, and translations are spread across methods

Possible responsibility:

- formalize a payload such as:
  - segment id
  - timestamp
  - original text
  - source language
  - primary translation
  - extra translations
  - timing metadata

### 3. `bilingual_overlay.py`

Purpose:

- host a clean bilingual floating subtitle layer separate from chat history

Why:

- `SubtitleOverlay` and `SubtitleWindow` currently serve different UX goals
- a dedicated bilingual floating layer would avoid overloading either class

Possible responsibility:

- render one active bilingual subtitle block
- support compact always-on-top display
- optionally support streaming partial translation

### 4. `translation_dispatcher.py`

Purpose:

- separate translation orchestration from `LiveTranslateApp`

Why:

- today `_translate_async()`, `_translate_extra_langs()`, and `_translate_subwin_only()` are embedded in `main.py`
- this makes `LiveTranslateApp` own both pipeline orchestration and translation fan-out logic

Possible responsibility:

- manage primary and extra target languages
- handle translator cloning via `Translator.with_target_language()`
- return a unified translation dictionary for all UI consumers

### 5. `display_targets.py`

Purpose:

- define a small adapter interface for display sinks

Why:

- current display targets are hardcoded:
  - `SubtitleOverlay`
  - `SubtitleWindow`

Possible responsibility:

- represent sinks such as:
  - overlay history panel
  - OBS subtitle window
  - future bilingual floating bar

This would make future UI additions less invasive.

---

## Summary

The current architecture is already split cleanly enough for a bilingual overlay feature:

- audio enters in `audio_capture.py`
- speech segmentation happens in `vad_processor.py`
- ASR is coordinated in `main.py` and implemented by pluggable engines
- translation happens through `translator.py`, dispatched from `main.py`
- overlay updates happen in `subtitle_overlay.py`
- structured multi-line subtitle rendering already exists in `subtitle_window.py`

The smallest invasive future change is to treat bilingual floating subtitles as a presentation-layer extension, not a pipeline rewrite.

The most localized current extension points are:

- `LiveTranslateApp._translate_async()` in `main.py`
- `ChatMessage` and `SubtitleOverlay` in `subtitle_overlay.py`
- optionally the line-based rendering model in `subtitle_window.py`
