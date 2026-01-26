//! Dia2 WebSocket Streaming Server
//!
//! This server provides a WebSocket interface for streaming TTS generation.
//! It communicates with a Python subprocess that runs the actual model inference.

mod session;
mod tts_bridge;

use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    extract::State,
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
