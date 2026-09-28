//! # fluidaudio-rs
//!
//! Rust bindings for [FluidAudio](https://github.com/FluidInference/FluidAudio) -
//! a Swift library for ASR, VAD, Speaker Diarization, and TTS on Apple platforms.
//!
//! ## Features
//!
//! - **ASR (Automatic Speech Recognition)** - High-quality speech-to-text using Parakeet TDT models
//! - **VAD (Voice Activity Detection)** - Detect speech segments in audio
//! - **Speaker Diarization** - Identify and label different speakers in audio
//!
//! ## Requirements
//!
//! - macOS 14+ or iOS 17+
//! - Apple Silicon (M1/M2/M3) recommended
//!
//! ## Example
//!
//! ```rust,no_run
//! use fluidaudio_rs::FluidAudio;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let audio = FluidAudio::new()?;
//!
//!     // Transcribe an audio file
//!     audio.init_asr()?;
//!     let result = audio.transcribe_file("audio.wav")?;
//!     println!("Text: {}", result.text);
//!     println!("Confidence: {:.2}%", result.confidence * 100.0);
//!
//!     Ok(())
//! }
//! ```

mod ffi;

use std::path::Path;
use thiserror::Error;

// Re-export FFI types
pub use ffi::{AsrResult, DiarizationSegment, SystemInfo, VadFrame};

/// Parakeet TDT model versions that [`FluidAudio::init_asr_with_version`] can load.
///
/// The discriminants are the FFI codes understood by the Swift bridge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum AsrModelVersion {
    /// Parakeet TDT v3 (0.6B, 25 languages). FluidAudio's default.
    #[default]
    V3 = 0,
    /// Parakeet Ultra: post-trained v3 with the same languages and speed.
    Ultra = 1,
    /// Parakeet Redux: 2-bit re-training of v3, about 220 MB. macOS 15+.
    Redux = 2,
    /// Parakeet TDT v2: English-only predecessor of v3.
    V2 = 3,
}

/// Errors that can occur when using FluidAudio
#[derive(Error, Debug)]
pub enum FluidAudioError {
    #[error("FluidAudio not initialized: {0}")]
    NotInitialized(String),

    #[error("Transcription failed: {0}")]
    TranscriptionFailed(String),

    #[error("Processing failed: {0}")]
    ProcessingFailed(String),

    #[error("Audio file not found: {0}")]
    FileNotFound(String),

    #[error("Swift bridge error: {0}")]
    BridgeError(String),
}

impl From<String> for FluidAudioError {
    fn from(s: String) -> Self {
        FluidAudioError::BridgeError(s)
    }
}

/// Main FluidAudio interface for Rust
///
/// Provides access to ASR and VAD functionality.
pub struct FluidAudio {
    bridge: ffi::FluidAudioBridge,
}

impl FluidAudio {
    /// Create a new FluidAudio instance
    pub fn new() -> Result<Self, FluidAudioError> {
        let bridge = ffi::FluidAudioBridge::new()
            .ok_or_else(|| FluidAudioError::BridgeError("Failed to create bridge".to_string()))?;
        Ok(Self { bridge })
    }

    // ========== ASR Methods ==========

    /// Initialize the ASR (Automatic Speech Recognition) engine
    ///
    /// This downloads and loads the ASR models. First run may take 20-30 seconds
    /// as models are compiled for the Neural Engine.
    pub fn init_asr(&self) -> Result<(), FluidAudioError> {
        self.bridge.initialize_asr().map_err(FluidAudioError::from)
    }

    /// Initialize the ASR engine with a specific Parakeet model version.
    ///
    /// Downloads the models on first use. [`AsrModelVersion::Ultra`] is a
    /// post-trained v3 with lower word error rates at the same speed; see
    /// the FluidAudio documentation for details.
    pub fn init_asr_with_version(&self, version: AsrModelVersion) -> Result<(), FluidAudioError> {
        self.bridge
            .initialize_asr_with_version(version as i32)
            .map_err(FluidAudioError::from)
    }

    /// Transcribe an audio file
    ///
    /// # Arguments
    /// * `path` - Path to the audio file (WAV, M4A, MP3, etc.)
    ///
    /// # Returns
    /// * `AsrResult` containing the transcribed text and metadata
    pub fn transcribe_file<P: AsRef<Path>>(&self, path: P) -> Result<AsrResult, FluidAudioError> {
        let path_str = path.as_ref().to_string_lossy();

        if !path.as_ref().exists() {
            return Err(FluidAudioError::FileNotFound(path_str.to_string()));
        }

        self.bridge
            .transcribe_file(&path_str)
            .map_err(FluidAudioError::from)
    }

    /// Transcribe audio samples directly
    ///
    /// This method accepts raw 16kHz mono audio samples, making it ideal for
    /// real-time audio applications where audio is captured from a microphone
    /// or other streaming source.
    ///
    /// # Arguments
    /// * `samples` - Slice of f32 audio samples (16kHz mono, normalized to -1.0 to 1.0)
    ///
    /// # Returns
    /// * `AsrResult` containing the transcribed text and metadata
    ///
    /// # Example
    /// ```rust,no_run
    /// use fluidaudio_rs::FluidAudio;
    ///
    /// fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let audio = FluidAudio::new()?;
    ///     audio.init_asr()?;
    ///
    ///     // Simulated audio buffer (16kHz mono)
    ///     let samples: Vec<f32> = vec![0.0; 16000]; // 1 second of silence
    ///
    ///     let result = audio.transcribe_samples(&samples)?;
    ///     println!("Text: {}", result.text);
    ///
    ///     Ok(())
    /// }
    /// ```
    pub fn transcribe_samples(&self, samples: &[f32]) -> Result<AsrResult, FluidAudioError> {
        self.bridge
            .transcribe_samples(samples)
            .map_err(FluidAudioError::from)
    }

    /// Check if ASR is initialized and ready
    pub fn is_asr_available(&self) -> bool {
        self.bridge.is_asr_available()
    }

    // ========== Streaming ASR Methods ==========

    /// Initialize Streaming ASR (memory-efficient, uses 99.5% less memory than regular ASR)
    ///
    /// Streaming ASR is ideal for long audio files or real-time transcription where
    /// memory usage is a concern. It processes audio in chunks rather than loading
    /// the entire file into memory.
    ///
    /// # Example
    /// ```rust,no_run
    /// use fluidaudio_rs::FluidAudio;
    ///
    /// fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let audio = FluidAudio::new()?;
    ///
    ///     // Initialize streaming ASR
    ///     audio.init_streaming_asr()?;
    ///
    ///     // Use session-based API for real-time streaming
    ///     audio.streaming_asr_start()?;
    ///
    ///     // Feed audio chunks as they become available
    ///     let chunk1: Vec<f32> = vec![0.0; 16000]; // 1 second
    ///     audio.streaming_asr_feed(&chunk1)?;
    ///
    ///     let chunk2: Vec<f32> = vec![0.0; 16000]; // another second
    ///     audio.streaming_asr_feed(&chunk2)?;
    ///
    ///     // Get final transcription
    ///     let text = audio.streaming_asr_finish()?;
    ///     println!("Transcription: {}", text);
    ///
    ///     Ok(())
    /// }
    /// ```
    pub fn init_streaming_asr(&self) -> Result<(), FluidAudioError> {
        self.bridge
            .initialize_streaming_asr()
            .map_err(FluidAudioError::from)
    }

    /// Start a streaming ASR session
    ///
    /// Call this before feeding audio chunks. Use `streaming_asr_feed()` to process
    /// audio chunks, then `streaming_asr_finish()` to get the final result.
    pub fn streaming_asr_start(&self) -> Result<(), FluidAudioError> {
        self.bridge
            .streaming_asr_start()
            .map_err(FluidAudioError::from)
    }

    /// Feed audio samples to the streaming ASR session
    ///
    /// # Arguments
    /// * `samples` - Slice of f32 audio samples (16kHz mono, normalized to -1.0 to 1.0)
    ///
    /// Call this multiple times to process audio in chunks. The transcription engine
    /// will process the audio incrementally.
    pub fn streaming_asr_feed(&self, samples: &[f32]) -> Result<(), FluidAudioError> {
        self.bridge
            .streaming_asr_feed(samples)
            .map_err(FluidAudioError::from)
    }

    /// Finish the streaming ASR session and get the transcription result
    ///
    /// # Returns
    /// * `String` containing the complete transcribed text
    ///
    /// This finalizes processing and returns the full transcription. After calling
    /// this, you must call `streaming_asr_start()` again to start a new session.
    pub fn streaming_asr_finish(&self) -> Result<String, FluidAudioError> {
        self.bridge
            .streaming_asr_finish()
            .map_err(FluidAudioError::from)
    }

    /// Transcribe an audio file using streaming ASR (memory-efficient wrapper)
    ///
    /// This is a convenience method that handles the session lifecycle for you.
    /// For long files or when memory usage is critical, this uses significantly
    /// less memory than `transcribe_file()`.
    ///
    /// # Arguments
    /// * `path` - Path to the audio file (WAV, M4A, MP3, etc.)
    ///
    /// # Returns
    /// * `AsrResult` containing the transcribed text and metadata
    ///
    /// # Example
    /// ```rust,no_run
    /// use fluidaudio_rs::FluidAudio;
    ///
    /// fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let audio = FluidAudio::new()?;
    ///     audio.init_streaming_asr()?;
    ///
    ///     let result = audio.transcribe_file_streaming("long_audio.wav")?;
    ///     println!("Text: {}", result.text);
    ///     println!("RTFx: {:.2}x", result.rtfx);
    ///
    ///     Ok(())
    /// }
    /// ```
    pub fn transcribe_file_streaming<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<AsrResult, FluidAudioError> {
        let path_str = path.as_ref().to_string_lossy();

        if !path.as_ref().exists() {
            return Err(FluidAudioError::FileNotFound(path_str.to_string()));
        }

        self.bridge
            .transcribe_file_streaming(&path_str)
            .map_err(FluidAudioError::from)
    }

    /// Check if streaming ASR is initialized and ready
    pub fn is_streaming_asr_available(&self) -> bool {
        self.bridge.is_streaming_asr_available()
    }

    // ========== VAD Methods ==========

    /// Initialize the VAD (Voice Activity Detection) engine
    ///
    /// # Arguments
    /// * `threshold` - Detection threshold (0.0-1.0, default 0.85)
    pub fn init_vad(&self, threshold: f32) -> Result<(), FluidAudioError> {
        self.bridge
            .initialize_vad(threshold)
            .map_err(FluidAudioError::from)
    }

    /// Check if VAD is initialized and ready
    pub fn is_vad_available(&self) -> bool {
        self.bridge.is_vad_available()
    }

    /// Run VAD over an audio file.
    ///
    /// The audio is automatically resampled to 16 kHz mono Float32 and processed
    /// in 4096-sample (256 ms) chunks. One [`VadFrame`] is returned per chunk.
    ///
    /// # Arguments
    /// * `path` - Path to an audio file (WAV, M4A, MP3, etc.)
    ///
    /// # Example
    /// ```rust,no_run
    /// use fluidaudio_rs::FluidAudio;
    ///
    /// fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let audio = FluidAudio::new()?;
    ///     audio.init_vad(0.5)?;
    ///     let frames = audio.vad_process_file("speech.wav")?;
    ///     let voiced = frames.iter().filter(|f| f.is_voice_active).count();
    ///     println!("{} / {} chunks classified as voice", voiced, frames.len());
    ///     Ok(())
    /// }
    /// ```
    pub fn vad_process_file<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<Vec<VadFrame>, FluidAudioError> {
        let path_str = path.as_ref().to_string_lossy();

        if !path.as_ref().exists() {
            return Err(FluidAudioError::FileNotFound(path_str.to_string()));
        }

        self.bridge
            .vad_process_file(&path_str)
            .map_err(FluidAudioError::from)
    }

    /// Run VAD over raw 16 kHz mono Float32 samples.
    ///
    /// One [`VadFrame`] is returned for every 4096-sample (256 ms) chunk; the
    /// trailing partial chunk (if any) is padded internally.
    pub fn vad_process_samples(&self, samples: &[f32]) -> Result<Vec<VadFrame>, FluidAudioError> {
        self.bridge
            .vad_process_samples(samples)
            .map_err(FluidAudioError::from)
    }

    // ========== Diarization Methods ==========

    /// Initialize the speaker diarization engine
    ///
    /// This downloads and loads the diarization models. First run may take
    /// some time as models are compiled for the Neural Engine.
    ///
    /// # Arguments
    /// * `threshold` - Clustering threshold (0.0-1.0, default 0.6). Lower values
    ///   produce more speakers, higher values merge speakers more aggressively.
    pub fn init_diarization(&self, threshold: f64) -> Result<(), FluidAudioError> {
        self.bridge
            .initialize_diarization(threshold)
            .map_err(FluidAudioError::from)
    }

    /// Diarize an audio file using a pre-staged Sortformer model (`.mlpackage`
    /// path), with **no** network download. The caller is responsible for
    /// provisioning and verifying the model. Config is `.balancedV2` (matches
    /// the `SortformerNvidiaLow_v2.mlpackage`). Unlike [`Self::diarize_file`],
    /// this never touches HuggingFace and needs no prior `init_diarization`.
    ///
    /// The compiled model is cached in a writable per-user directory (never next
    /// to `model_path`, so a read-only / air-gapped model location works) and the
    /// loaded handle is retained in-memory, so repeated calls in the same process
    /// reuse it with no reload.
    ///
    /// # Arguments
    /// * `audio` - Path to the audio file (WAV, M4A, MP3, etc.)
    /// * `model_path` - Path to the Sortformer `.mlpackage`
    pub fn diarize_file_with_models<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        audio: P,
        model_path: Q,
    ) -> Result<Vec<DiarizationSegment>, FluidAudioError> {
        let audio_str = audio.as_ref().to_string_lossy();
        if !audio.as_ref().exists() {
            return Err(FluidAudioError::FileNotFound(audio_str.to_string()));
        }
        let model_str = model_path.as_ref().to_string_lossy();
        if !model_path.as_ref().exists() {
            return Err(FluidAudioError::FileNotFound(model_str.to_string()));
        }
        self.bridge
            .diarize_file_with_models(&audio_str, &model_str)
            .map_err(FluidAudioError::from)
    }

    /// Pre-compile the Sortformer `.mlpackage` and warm it, so the first real
    /// diarization is fast. The compiled `.mlmodelc` is written to a writable
    /// per-user cache directory (keyed by a fingerprint of the model — its path,
    /// total size, and newest mtime — never written next
    /// to `model_path`, so a read-only / air-gapped model location works), paying
    /// the one-time ~100s ANE compile up front (e.g. at install time). The loaded
    /// model is also retained in-memory, so within the **same process** subsequent
    /// [`Self::diarize_file_with_models`] calls reuse it with no reload. Across
    /// **separate processes** only the on-disk compiled cache carries over, so the
    /// first diarize in a new process still pays the ~4s warm `MLModel` load (vs
    /// ~100s cold). No audio is processed.
    ///
    /// # Arguments
    /// * `model_path` - Path to the Sortformer `.mlpackage`
    pub fn compile_diarization_model<Q: AsRef<Path>>(
        &self,
        model_path: Q,
    ) -> Result<(), FluidAudioError> {
        let model_str = model_path.as_ref().to_string_lossy();
        if !model_path.as_ref().exists() {
            return Err(FluidAudioError::FileNotFound(model_str.to_string()));
        }
        self.bridge
            .compile_diarization_model(&model_str)
            .map_err(FluidAudioError::from)
    }

    /// Diarize an audio file to identify speaker segments
    ///
    /// # Arguments
    /// * `path` - Path to the audio file (WAV, M4A, MP3, etc.)
    ///
    /// # Returns
    /// * `Vec<DiarizationSegment>` containing speaker-labeled time segments
    pub fn diarize_file<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<Vec<DiarizationSegment>, FluidAudioError> {
        let path_str = path.as_ref().to_string_lossy();

        if !path.as_ref().exists() {
            return Err(FluidAudioError::FileNotFound(path_str.to_string()));
        }

        self.bridge
            .diarize_file(&path_str)
            .map_err(FluidAudioError::from)
    }

    /// Check if diarization is initialized and ready
    pub fn is_diarization_available(&self) -> bool {
        self.bridge.is_diarization_available()
    }

    // ========== TTS (Kokoro) Methods ==========

    /// Initialize the Kokoro TTS engine with a default voice and language.
    ///
    /// `lang` selects the KokoroAne variant in the Swift bridge (`zh` → Mandarin,
    /// everything else → English). Downloads the variant's model on first run
    /// (FluidAudio-managed cache).
    pub fn init_kokoro(&self, default_voice: &str, lang: &str) -> Result<(), FluidAudioError> {
        self.bridge
            .initialize_kokoro(default_voice, lang)
            .map_err(FluidAudioError::from)
    }

    /// Synthesize text via Kokoro TTS in the language the engine was initialized
    /// with (`init_kokoro`'s `lang`): English by default, or Mandarin when
    /// `lang` was `zh` (the `.mandarin` KokoroAne variant).
    ///
    /// # Arguments
    /// * `text` - Text to synthesize
    /// * `voice` - Kokoro voice id (e.g. `am_michael`)
    /// * `speed` - Speech rate (0.5-2.0; clamped by FluidAudio)
    ///
    /// # Returns
    /// * `Vec<u8>` - a complete WAV (24 kHz mono, 16-bit PCM)
    pub fn synthesize_kokoro(
        &self,
        text: &str,
        voice: &str,
        speed: f32,
    ) -> Result<Vec<u8>, FluidAudioError> {
        self.bridge
            .kokoro_synthesize(text, voice, speed)
            .map_err(FluidAudioError::from)
    }

    /// Check if Kokoro TTS is initialized and ready.
    pub fn is_kokoro_available(&self) -> bool {
        self.bridge.is_kokoro_available()
    }

    // ========== System Info ==========

    /// Get system information
    pub fn system_info(&self) -> SystemInfo {
        self.bridge.system_info()
    }

    /// Check if running on Apple Silicon
    pub fn is_apple_silicon(&self) -> bool {
        self.bridge.is_apple_silicon()
    }

    /// Check if running on an Intel Mac (x86_64).
    ///
    /// Local Apple-Silicon-only models (Parakeet, CoreML TTS) will fail
    /// to load on Intel Macs. Apps can use this to gate UI selection of those
    /// models and steer Intel users to cloud transcription instead.
    pub fn is_intel_mac(&self) -> bool {
        self.bridge.is_intel_mac()
    }

    // ========== ITN (Inverse Text Normalization) ==========

    /// Normalize a short spoken-form expression to written form.
    ///
    /// Examples:
    /// - `"two hundred thirty two"` → `"232"`
    /// - `"five dollars and fifty cents"` → `"$5.50"`
    /// - `"period"` → `"."`
    ///
    /// Intended for short fragments. For full sentences with mixed punctuation
    /// commands and ordinary prose, prefer [`itn_normalize_sentence`](Self::itn_normalize_sentence).
    pub fn itn_normalize(&self, text: &str) -> Result<String, FluidAudioError> {
        self.bridge
            .itn_normalize(text)
            .map_err(FluidAudioError::from)
    }

    /// Sentence-mode normalization with sliding-window span matching.
    ///
    /// Uses Apple's NaturalLanguage framework to disambiguate words like
    /// `"period"` between punctuation commands and ordinary nouns/verbs.
    pub fn itn_normalize_sentence(&self, text: &str) -> Result<String, FluidAudioError> {
        self.bridge
            .itn_normalize_sentence(text)
            .map_err(FluidAudioError::from)
    }

    /// Sentence-mode normalization with a caller-controlled maximum span size
    /// (in tokens). Larger spans catch longer multi-word numbers/dates at the
    /// cost of more work per sentence.
    pub fn itn_normalize_sentence_max_span(
        &self,
        text: &str,
        max_span_tokens: u32,
    ) -> Result<String, FluidAudioError> {
        self.bridge
            .itn_normalize_sentence_max_span(text, max_span_tokens)
            .map_err(FluidAudioError::from)
    }

    /// Whether the underlying native NeMo ITN library is loaded.
    ///
    /// When `false`, the Swift-side normalizer falls back to its
    /// Apple-NaturalLanguage-only path (still functional, but with reduced
    /// coverage on long multi-token expressions).
    pub fn itn_is_native_available(&self) -> bool {
        self.bridge.itn_is_native_available()
    }

    // ========== Cleanup ==========

    /// Release all resources
    pub fn cleanup(&self) {
        self.bridge.cleanup()
    }
}

impl Drop for FluidAudio {
    fn drop(&mut self) {
        self.cleanup();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_instance() {
        // Note: This test will fail until Swift bridge is properly linked
        // For now, just test the types exist
        let _ = FluidAudioError::NotInitialized("test".to_string());
    }
}
