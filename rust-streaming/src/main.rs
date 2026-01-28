//! Dia2 WebSocket Streaming Server
//!
//! This server provides a WebSocket interface for streaming TTS generation.
//! It communicates with a Python subprocess that runs the actual model inference.

mod session;
mod tts_bridge;

use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    extract::{Query, State},
    http::header,
    response::IntoResponse,
    routing::get,
    Router,
};
use futures::{sink::SinkExt, stream::StreamExt};
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_http::cors::{Any, CorsLayer};
use tower_http::services::ServeDir;
use tracing::{error, info, warn};

use session::SessionManager;
use tts_bridge::{TTSBridge, TTSEvent, TTSRequest};

/// Query parameters for test WAV endpoint
#[derive(serde::Deserialize)]
struct TestWavParams {
    /// Test type: silence, sine_zero_end, sine_nonzero_end, dc_offset
    #[serde(default = "default_test_type")]
    test_type: String,
    /// Duration in milliseconds
    #[serde(default = "default_duration")]
    duration_ms: u32,
}

fn default_test_type() -> String {
    "sine_nonzero_end".to_string()
}

fn default_duration() -> u32 {
    160
}

/// Application state shared across all connections
pub struct AppState {
    session_manager: SessionManager,
    tts_bridge: TTSBridge,
}

impl AppState {
    pub fn new(python_bridge_path: String) -> Self {
        Self {
            session_manager: SessionManager::new(),
            tts_bridge: TTSBridge::new(&python_bridge_path),
        }
    }
}

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "dia2_streaming_server=info,tower_http=info".into()),
        )
        .init();

    // Get bridge path from environment or use default
    let bridge_path = std::env::var("TTS_BRIDGE_PATH")
        .unwrap_or_else(|_| "./tts_bridge.py".to_string());

    let state = Arc::new(RwLock::new(AppState::new(bridge_path)));

    // Spawn warmup task to pre-load the model in the background
    // This runs while the server starts accepting connections
    let warmup_state = state.clone();
    tokio::spawn(async move {
        info!("Starting model warmup in background...");
        let state_guard = warmup_state.read().await;
        if let Err(e) = state_guard.tts_bridge.warmup().await {
            error!("Failed to warmup TTS bridge: {}", e);
        } else {
            info!("Model warmup completed successfully");
        }
    });

    // CORS configuration
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // Get frontend path from environment or use default
    let frontend_path = std::env::var("FRONTEND_PATH")
        .unwrap_or_else(|_| "./frontend".to_string());

    // Build router
    let app = Router::new()
        .route("/ws", get(ws_handler))
        .route("/ws/stream", get(ws_handler))
        .route("/health", get(health_handler))
        .route("/test-chunk.wav", get(test_wav_handler))
        .nest_service("/", ServeDir::new(&frontend_path))
        .layer(cors)
        .with_state(state);

    // Get port from environment
    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);

    let addr = format!("0.0.0.0:{}", port);
    info!("Starting Dia2 Streaming Server on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

/// Health check endpoint
async fn health_handler() -> impl IntoResponse {
    axum::Json(serde_json::json!({
        "status": "ok",
        "service": "dia2-streaming-server",
        "version": env!("CARGO_PKG_VERSION")
    }))
}

/// Test WAV file endpoint - returns a downloadable WAV file for testing
///
/// Usage: GET /test-chunk.wav?test_type=sine_nonzero_end&duration_ms=160
///
/// Test types:
/// - silence: Pure silence (all zeros) - should NOT pop
/// - sine_zero_end: 440Hz sine that ends EXACTLY at zero - should NOT pop
/// - sine_fade_end: 440Hz sine with 5ms fade-out at end - should NOT pop
/// - sine_nonzero_end: 440Hz sine that ends at non-zero - WILL pop
/// - dc_offset: Constant non-zero value - WILL pop
async fn test_wav_handler(Query(params): Query<TestWavParams>) -> impl IntoResponse {
    let sample_rate = 24000u32;
    let num_samples = (sample_rate as f64 * params.duration_ms as f64 / 1000.0) as usize;

    info!("Generating test WAV: type={}, duration={}ms, samples={}",
          params.test_type, params.duration_ms, num_samples);

    let samples: Vec<i16> = match params.test_type.as_str() {
        "silence" => {
            vec![0i16; num_samples]
        }
        "sine_zero_end" => {
            // Sine wave that ends EXACTLY at zero
            // For sample indices 0 to N-1, we want phase(N-1) = 2*pi*k for some integer k
            // phase(i) = 2*pi*k*i/(N-1)
            // This means sample[N-1] = sin(2*pi*k) = 0
            let base_freq = 440.0;
            let desired_cycles = (base_freq * params.duration_ms as f64 / 1000.0).round() as usize;

            info!("sine_zero_end: {} complete cycles over {} samples", desired_cycles, num_samples);

            (0..num_samples)
                .map(|i| {
                    // Phase goes from 0 to 2*pi*desired_cycles as i goes from 0 to num_samples-1
                    let phase = 2.0 * std::f64::consts::PI * desired_cycles as f64 * i as f64 / (num_samples - 1) as f64;
                    let sample = phase.sin();
                    (sample * 16000.0) as i16
                })
                .collect()
        }
        "sine_fade_end" => {
            // Sine wave with a fade-out at the end (guaranteed no pop)
            let freq = 440.0;
            let fade_samples = (sample_rate as f64 * 0.005) as usize; // 5ms fade

            (0..num_samples)
                .map(|i| {
                    let t = i as f64 / sample_rate as f64;
                    let mut sample = (2.0 * std::f64::consts::PI * freq * t).sin();

                    // Apply fade at the end
                    let samples_from_end = num_samples - i;
                    if samples_from_end < fade_samples {
                        sample *= samples_from_end as f64 / fade_samples as f64;
                    }

                    (sample * 16000.0) as i16
                })
                .collect()
        }
        "dc_offset" => {
            vec![8000i16; num_samples]
        }
        "sine_nonzero_end" | _ => {
            // Sine wave that does NOT complete full cycles (ends at non-zero)
            let freq = 440.0;
            (0..num_samples)
                .map(|i| {
                    let t = i as f64 / sample_rate as f64;
                    let sample = (2.0 * std::f64::consts::PI * freq * t).sin();
                    (sample * 16000.0) as i16
                })
                .collect()
        }
    };

    // Log boundary values
    let first_sample = samples.first().copied().unwrap_or(0);
    let last_sample = samples.last().copied().unwrap_or(0);
    info!("WAV generated: first_sample={} ({:.6}), last_sample={} ({:.6})",
          first_sample, first_sample as f64 / 32767.0,
          last_sample, last_sample as f64 / 32767.0);

    let wav_data = generate_wav(&samples, sample_rate);

    // Return as downloadable WAV file
    (
        [
            (header::CONTENT_TYPE, "audio/wav"),
            (header::CONTENT_DISPOSITION, "attachment; filename=\"test-chunk.wav\""),
        ],
        wav_data,
    )
}

/// WebSocket upgrade handler
async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<RwLock<AppState>>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

/// Handle an individual WebSocket connection
async fn handle_socket(socket: WebSocket, state: Arc<RwLock<AppState>>) {
    let session_id = uuid::Uuid::new_v4().to_string();
    info!("New WebSocket connection: {}", session_id);

    let (mut sender, mut receiver) = socket.split();

    // Register session
    {
        let mut state_guard = state.write().await;
        state_guard.session_manager.add_session(&session_id);
    }

    // Text buffer for bidirectional streaming
    let text_buffer = Arc::new(RwLock::new(String::new()));
    let generation_active = Arc::new(RwLock::new(false));

    // Send welcome message
    let welcome = serde_json::json!({
        "type": "connected",
        "session_id": session_id,
        "message": "Connected to Dia2 Streaming Server"
    });
    if sender.send(Message::Text(welcome.to_string())).await.is_err() {
        return;
    }

    // Main message loop
    loop {
        tokio::select! {
            Some(msg) = receiver.next() => {
                match msg {
                    Ok(Message::Text(text)) => {
                        if let Err(e) = handle_text_message(
                            &text,
                            &mut sender,
                            &session_id,
                            &state,
                            &text_buffer,
                            &generation_active,
                        ).await {
                            error!("Error handling message: {}", e);
                            let error_msg = serde_json::json!({
                                "type": "error",
                                "error": e.to_string()
                            });
                            let _ = sender.send(Message::Text(error_msg.to_string())).await;
                        }
                    }
                    Ok(Message::Binary(data)) => {
                        // Binary messages could be used for audio input
                        warn!("Received binary message ({} bytes) - not implemented", data.len());
                    }
                    Ok(Message::Close(_)) => {
                        info!("Client {} requested close", session_id);
                        break;
                    }
                    Ok(Message::Ping(data)) => {
                        let _ = sender.send(Message::Pong(data)).await;
                    }
                    Ok(_) => {}
                    Err(e) => {
                        error!("WebSocket error: {}", e);
                        break;
                    }
                }
            }
            else => break,
        }
    }

    // Cleanup
    {
        let mut state_guard = state.write().await;
        state_guard.session_manager.remove_session(&session_id);
    }
    info!("WebSocket connection closed: {}", session_id);
}

/// Handle incoming text messages
async fn handle_text_message(
    text: &str,
    sender: &mut futures::stream::SplitSink<WebSocket, Message>,
    session_id: &str,
    state: &Arc<RwLock<AppState>>,
    text_buffer: &Arc<RwLock<String>>,
    generation_active: &Arc<RwLock<bool>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let msg: serde_json::Value = serde_json::from_str(text)?;

    match msg.get("type").and_then(|t| t.as_str()) {
        Some("text_chunk") => {
            // Bidirectional streaming: accumulate text chunks
            let chunk_text = msg.get("text").and_then(|t| t.as_str()).unwrap_or("");
            let is_final = msg.get("final").and_then(|f| f.as_bool()).unwrap_or(false);

            {
                let mut buffer = text_buffer.write().await;
                buffer.push_str(chunk_text);
            }

            // Acknowledge chunk
            let ack = serde_json::json!({
                "type": "chunk_ack",
                "chunk_index": msg.get("chunk_index"),
                "buffer_length": text_buffer.read().await.len()
            });
            sender.send(Message::Text(ack.to_string())).await?;

            // If final chunk, start generation
            if is_final {
                let full_text = {
                    let buffer = text_buffer.read().await;
                    buffer.clone()
                };

                // Clear buffer for next request
                {
                    let mut buffer = text_buffer.write().await;
                    buffer.clear();
                }

                // Start generation
                let request = TTSRequest {
                    text: full_text,
                    model_size: msg.get("model").and_then(|m| m.as_str()).unwrap_or("2b").to_string(),
                    config: msg.get("config").cloned(),
                };

                start_streaming_generation(
                    request,
                    sender,
                    session_id,
                    state,
                    generation_active,
                ).await?;
            }
        }
        Some("generate") => {
            info!("Received 'generate' message");
            // Direct generation request (non-streaming input)
            let text_input = msg.get("text")
                .or_else(|| msg.get("text_input"))
                .and_then(|t| t.as_str())
                .unwrap_or("")
                .to_string();

            if text_input.is_empty() {
                let error = serde_json::json!({
                    "type": "error",
                    "error": "Text input cannot be empty"
                });
                sender.send(Message::Text(error.to_string())).await?;
                return Ok(());
            }

            let request = TTSRequest {
                text: text_input,
                model_size: msg.get("model").and_then(|m| m.as_str()).unwrap_or("2b").to_string(),
                config: msg.get("config").cloned(),
            };

            start_streaming_generation(
                request,
                sender,
                session_id,
                state,
                generation_active,
            ).await?;
        }
        Some("cancel") => {
            // Cancel current generation
            let mut active = generation_active.write().await;
            *active = false;

            let cancelled = serde_json::json!({
                "type": "cancelled",
                "message": "Generation cancelled"
            });
            sender.send(Message::Text(cancelled.to_string())).await?;
        }
        Some("ping") => {
            let pong = serde_json::json!({
                "type": "pong",
                "timestamp": chrono_timestamp()
            });
            sender.send(Message::Text(pong.to_string())).await?;
        }
        Some("test_audio") => {
            // Test mode: send pre-generated WAV chunks to isolate browser vs generation issues
            let test_type = msg.get("test_type").and_then(|t| t.as_str()).unwrap_or("sine_zero_end");
            let num_chunks = msg.get("num_chunks").and_then(|n| n.as_u64()).unwrap_or(3) as usize;

            info!("Running audio test: type={}, chunks={}", test_type, num_chunks);

            send_test_audio_chunks(sender, test_type, num_chunks).await?;
        }
        _ => {
            let error = serde_json::json!({
                "type": "error",
                "error": format!("Unknown message type: {:?}", msg.get("type"))
            });
            sender.send(Message::Text(error.to_string())).await?;
        }
    }

    Ok(())
}

/// Start streaming generation via persistent Python subprocess
async fn start_streaming_generation(
    request: TTSRequest,
    sender: &mut futures::stream::SplitSink<WebSocket, Message>,
    session_id: &str,
    state: &Arc<RwLock<AppState>>,
    generation_active: &Arc<RwLock<bool>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Check if already generating
    {
        let active = generation_active.read().await;
        if *active {
            let error = serde_json::json!({
                "type": "error",
                "error": "Generation already in progress"
            });
            sender.send(Message::Text(error.to_string())).await?;
            return Ok(());
        }
    }

    // Set active flag
    {
        let mut active = generation_active.write().await;
        *active = true;
    }

    info!("Starting generation for session {} with text: {}...",
        session_id, &request.text.chars().take(50).collect::<String>());

    // Notify client
    let starting = serde_json::json!({
        "type": "status",
        "message": "Starting generation",
        "progress": 0.0
    });
    info!("Sending 'Starting generation' status to client");
    sender.send(Message::Text(starting.to_string())).await?;

    // Use the shared persistent TTS bridge
    let event_stream = {
        let state_guard = state.read().await;
        info!("Using persistent TTS bridge");
        state_guard.tts_bridge.generate_stream(request).await
    };

    match event_stream {
        Ok(mut event_stream) => {
            info!("Bridge started successfully, waiting for request_start synchronization");

            // Keep-alive interval (30 seconds) to prevent infrastructure timeouts
            let mut keepalive_interval = tokio::time::interval(std::time::Duration::from_secs(30));
            let mut keepalive_count = 0u32;
            let mut received_first_event = false;
            let mut synchronized = false;  // Wait for request_start before processing events

            loop {
                tokio::select! {
                    // Handle TTS events
                    event = event_stream.recv() => {
                        match event {
                            Some(event) => {
                                // Handle request synchronization first
                                if let TTSEvent::RequestStart { ref request_id } = event {
                                    info!("Synchronized with request: {}", request_id);
                                    synchronized = true;
                                    received_first_event = true;
                                    continue;  // Don't forward request_start to client
                                }

                                // Discard stale events until we see request_start
                                if !synchronized {
                                    warn!("Discarding stale event before request_start: {:?}",
                                          std::mem::discriminant(&event));
                                    continue;
                                }

                                info!("Received TTS event: {:?}", std::mem::discriminant(&event));
                                received_first_event = true;

                                // Check if cancelled
                                {
                                    let active = generation_active.read().await;
                                    if !*active {
                                        break;
                                    }
                                }

                                let (msg, is_terminal) = match event {
                                    TTSEvent::Status { message, progress } => {
                                        (serde_json::json!({
                                            "type": "status",
                                            "message": message,
                                            "progress": progress
                                        }), false)
                                    }
                                    TTSEvent::AudioChunk { data, chunk_index, timestamp_ms } => {
                                        (serde_json::json!({
                                            "type": "audio",
                                            "data": base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &data),
                                            "chunk_index": chunk_index,
                                            "timestamp_ms": timestamp_ms
                                        }), false)
                                    }
                                    TTSEvent::Complete { total_chunks, total_duration_ms } => {
                                        (serde_json::json!({
                                            "type": "complete",
                                            "total_chunks": total_chunks,
                                            "total_duration_ms": total_duration_ms
                                        }), true)
                                    }
                                    TTSEvent::Error { error } => {
                                        (serde_json::json!({
                                            "type": "error",
                                            "error": error
                                        }), true)
                                    }
                                    TTSEvent::Seed { seed } => {
                                        (serde_json::json!({
                                            "type": "seed",
                                            "seed": seed
                                        }), false)
                                    }
                                    TTSEvent::RequestStart { .. } => {
                                        // Already handled above, but match is exhaustive
                                        continue;
                                    }
                                };

                                if sender.send(Message::Text(msg.to_string())).await.is_err() {
                                    break;
                                }

                                if is_terminal {
                                    break;
                                }
                            }
                            None => break, // Stream ended
                        }
                    }

                    // Send keep-alive messages to prevent timeout
                    _ = keepalive_interval.tick() => {
                        keepalive_count += 1;
                        let status_msg = if !received_first_event {
                            // Model is still loading
                            format!("Loading model... ({}s)", keepalive_count * 30)
                        } else {
                            format!("Processing... ({}s)", keepalive_count * 30)
                        };

                        info!("Sending keep-alive #{}: {}", keepalive_count, status_msg);

                        let keepalive = serde_json::json!({
                            "type": "status",
                            "message": status_msg,
                            "progress": 0.05, // Small progress to indicate activity
                            "keepalive": true
                        });

                        if sender.send(Message::Text(keepalive.to_string())).await.is_err() {
                            warn!("Failed to send keep-alive, breaking loop");
                            break;
                        }
                    }
                }
            }
        }
        Err(e) => {
            let error = serde_json::json!({
                "type": "error",
                "error": e.to_string()
            });
            sender.send(Message::Text(error.to_string())).await?;
        }
    }

    // Reset active flag
    {
        let mut active = generation_active.write().await;
        *active = false;
    }

    Ok(())
}

fn chrono_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Generate a WAV file from PCM samples
fn generate_wav(samples: &[i16], sample_rate: u32) -> Vec<u8> {
    let num_samples = samples.len();
    let data_size = (num_samples * 2) as u32;
    let file_size = 36 + data_size;

    let mut wav = Vec::with_capacity(44 + num_samples * 2);

    // RIFF header
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&file_size.to_le_bytes());
    wav.extend_from_slice(b"WAVE");

    // fmt chunk
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    wav.extend_from_slice(&1u16.to_le_bytes());  // audio format (PCM)
    wav.extend_from_slice(&1u16.to_le_bytes());  // num channels
    wav.extend_from_slice(&sample_rate.to_le_bytes()); // sample rate
    wav.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    wav.extend_from_slice(&2u16.to_le_bytes());  // block align
    wav.extend_from_slice(&16u16.to_le_bytes()); // bits per sample

    // data chunk
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_size.to_le_bytes());
    for sample in samples {
        wav.extend_from_slice(&sample.to_le_bytes());
    }

    wav
}

/// Generate test audio chunks for debugging
///
/// Test types:
/// - "silence": Pure silence (all zeros) - should NOT pop
/// - "sine_zero_end": 440Hz sine wave that ends at zero crossing - should NOT pop
/// - "sine_nonzero_end": 440Hz sine wave that ends at non-zero - SHOULD pop
/// - "dc_offset": Constant DC offset (non-zero throughout) - SHOULD pop
async fn send_test_audio_chunks(
    sender: &mut futures::stream::SplitSink<WebSocket, Message>,
    test_type: &str,
    num_chunks: usize,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let sample_rate = 24000u32;
    let chunk_duration_ms = 160.0;
    let samples_per_chunk = (sample_rate as f64 * chunk_duration_ms / 1000.0) as usize;

    info!("Generating {} test chunks of type '{}' ({} samples each at {}Hz)",
          num_chunks, test_type, samples_per_chunk, sample_rate);

    // Notify start
    let start_msg = serde_json::json!({
        "type": "status",
        "message": format!("Starting audio test: {}", test_type),
        "progress": 0.0
    });
    sender.send(Message::Text(start_msg.to_string())).await?;

    let mut total_samples_generated = 0usize;

    for chunk_idx in 0..num_chunks {
        let samples: Vec<i16> = match test_type {
            "silence" => {
                // Pure silence - all zeros
                vec![0i16; samples_per_chunk]
            }
            "sine_zero_end" => {
                // Sine wave that completes full cycles (ends at zero)
                let freq = 440.0;
                let cycles_per_chunk = (freq * chunk_duration_ms / 1000.0).round();
                let adjusted_samples = (cycles_per_chunk * sample_rate as f64 / freq) as usize;

                (0..adjusted_samples)
                    .map(|i| {
                        let t = i as f64 / sample_rate as f64;
                        let sample = (2.0 * std::f64::consts::PI * freq * t).sin();
                        (sample * 16000.0) as i16
                    })
                    .collect()
            }
            "sine_nonzero_end" => {
                // Sine wave that does NOT complete full cycles (ends at non-zero)
                let freq = 440.0;
                // Use exact samples_per_chunk which won't align with sine period
                (0..samples_per_chunk)
                    .map(|i| {
                        let global_sample = total_samples_generated + i;
                        let t = global_sample as f64 / sample_rate as f64;
                        let sample = (2.0 * std::f64::consts::PI * freq * t).sin();
                        (sample * 16000.0) as i16
                    })
                    .collect()
            }
            "dc_offset" => {
                // Constant non-zero value (DC offset)
                vec![8000i16; samples_per_chunk]
            }
            "ramp_up" => {
                // Ramp from 0 to max - will pop at end
                (0..samples_per_chunk)
                    .map(|i| ((i as f64 / samples_per_chunk as f64) * 16000.0) as i16)
                    .collect()
            }
            "ramp_down" => {
                // Ramp from max to 0 - should NOT pop at end
                (0..samples_per_chunk)
                    .map(|i| ((1.0 - i as f64 / samples_per_chunk as f64) * 16000.0) as i16)
                    .collect()
            }
            _ => {
                // Default to silence
                vec![0i16; samples_per_chunk]
            }
        };

        total_samples_generated += samples.len();

        // Log the last few samples for debugging
        let last_5: Vec<i16> = samples.iter().rev().take(5).rev().cloned().collect();
        let last_sample_float = *samples.last().unwrap_or(&0) as f64 / 32767.0;
        info!("Chunk {}: {} samples, last 5 PCM: {:?}, last_float: {:.6}",
              chunk_idx, samples.len(), last_5, last_sample_float);

        let wav_data = generate_wav(&samples, sample_rate);

        let audio_msg = serde_json::json!({
            "type": "audio",
            "data": base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &wav_data),
            "chunk_index": chunk_idx,
            "timestamp_ms": chunk_idx as f64 * chunk_duration_ms,
            "test_info": {
                "test_type": test_type,
                "samples": samples.len(),
                "last_sample_pcm": samples.last().unwrap_or(&0),
                "last_sample_float": last_sample_float
            }
        });

        sender.send(Message::Text(audio_msg.to_string())).await?;

        // Small delay between chunks to simulate streaming
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }

    // Send complete
    let complete_msg = serde_json::json!({
        "type": "complete",
        "total_chunks": num_chunks,
        "total_duration_ms": num_chunks as f64 * chunk_duration_ms,
        "test_type": test_type
    });
    sender.send(Message::Text(complete_msg.to_string())).await?;

    info!("Test audio complete: {} chunks sent", num_chunks);

    Ok(())
}
