//! TTS Bridge - Communicates with Python subprocess for model inference

use serde::{Deserialize, Serialize};
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;
use tokio::sync::mpsc;
use tracing::{error, info, warn};

/// Request to generate TTS
#[derive(Debug, Serialize)]
pub struct TTSRequest {
    pub text: String,
    pub model_size: String,
    pub config: Option<serde_json::Value>,
}

/// Events emitted during TTS generation
#[derive(Debug, Clone)]
pub enum TTSEvent {
    Status {
        message: String,
        progress: f64,
    },
    AudioChunk {
        data: Vec<u8>,
        chunk_index: u32,
        timestamp_ms: f64,
    },
    Complete {
        total_chunks: u32,
        total_duration_ms: f64,
    },
    Error {
        error: String,
    },
}

/// JSON structure from Python bridge
#[derive(Debug, Deserialize)]
struct PythonEvent {
    #[serde(rename = "type")]
    event_type: String,
    // Status fields
    message: Option<String>,
    progress: Option<f64>,
    // Audio chunk fields
    data: Option<String>, // base64 encoded
    chunk_index: Option<u32>,
    timestamp_ms: Option<f64>,
    // Complete fields
    total_chunks: Option<u32>,
    total_duration_ms: Option<f64>,
    // Error fields
    error: Option<String>,
}

pub struct TTSBridge {
    bridge_path: String,
}

impl TTSBridge {
    pub fn new(bridge_path: &str) -> Self {
        Self {
            bridge_path: bridge_path.to_string(),
        }
    }

    pub async fn generate_stream(
        &self,
        request: TTSRequest,
    ) -> Result<mpsc::Receiver<TTSEvent>, Box<dyn std::error::Error + Send + Sync>> {
        let (tx, rx) = mpsc::channel(100);

        let bridge_path = self.bridge_path.clone();
        let request_json = serde_json::to_string(&request)?;

        tokio::spawn(async move {
            if let Err(e) = run_python_bridge(&bridge_path, &request_json, tx.clone()).await {
                error!("Python bridge error: {}", e);
                let _ = tx.send(TTSEvent::Error {
                    error: e.to_string(),
                }).await;
            }
        });

        Ok(rx)
    }
}

async fn run_python_bridge(
    bridge_path: &str,
    request_json: &str,
    tx: mpsc::Sender<TTSEvent>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    info!("Starting Python bridge: {}", bridge_path);

    let mut child = Command::new("python")
        .arg(bridge_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let mut stdin = child.stdin.take().expect("Failed to get stdin");
    let stdout = child.stdout.take().expect("Failed to get stdout");
    let stderr = child.stderr.take().expect("Failed to get stderr");

    // Send request to Python
    stdin.write_all(request_json.as_bytes()).await?;
    stdin.write_all(b"\n").await?;
    stdin.flush().await?;
    drop(stdin); // Close stdin to signal end of input

    // Spawn stderr reader for debugging
    let tx_err = tx.clone();
    tokio::spawn(async move {
        let mut reader = BufReader::new(stderr);
        let mut line = String::new();
        while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
            warn!("Python stderr: {}", line.trim());
            line.clear();
        }
    });

    // Read events from stdout
    let mut reader = BufReader::new(stdout);
    let mut line = String::new();

    loop {
        line.clear();
        match reader.read_line(&mut line).await {
            Ok(0) => break, // EOF
            Ok(_) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }

                match serde_json::from_str::<PythonEvent>(trimmed) {
                    Ok(event) => {
                        let tts_event = parse_python_event(event);
                        let is_complete = matches!(tts_event, TTSEvent::Complete { .. } | TTSEvent::Error { .. });

                        if tx.send(tts_event).await.is_err() {
                            warn!("Receiver dropped, stopping bridge");
                            break;
                        }

                        if is_complete {
                            break;
                        }
                    }
                    Err(e) => {
                        warn!("Failed to parse Python event: {} - line: {}", e, trimmed);
                    }
                }
            }
            Err(e) => {
                error!("Error reading from Python: {}", e);
                break;
            }
        }
    }

    // Wait for child to finish
    let _ = child.wait().await;

    Ok(())
}

fn parse_python_event(event: PythonEvent) -> TTSEvent {
    match event.event_type.as_str() {
        "status" => TTSEvent::Status {
            message: event.message.unwrap_or_default(),
            progress: event.progress.unwrap_or(0.0),
        },
        "audio" | "audio_chunk" => {
            let data = event
                .data
                .and_then(|d| base64::Engine::decode(&base64::engine::general_purpose::STANDARD, &d).ok())
                .unwrap_or_default();

            TTSEvent::AudioChunk {
                data,
                chunk_index: event.chunk_index.unwrap_or(0),
                timestamp_ms: event.timestamp_ms.unwrap_or(0.0),
            }
        }
        "complete" => TTSEvent::Complete {
            total_chunks: event.total_chunks.unwrap_or(0),
            total_duration_ms: event.total_duration_ms.unwrap_or(0.0),
        },
        "error" => TTSEvent::Error {
            error: event.error.unwrap_or_else(|| "Unknown error".to_string()),
        },
        _ => TTSEvent::Error {
            error: format!("Unknown event type: {}", event.event_type),
        },
    }
}
