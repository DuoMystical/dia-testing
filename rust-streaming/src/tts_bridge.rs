//! TTS Bridge - Persistent Python subprocess for model inference
//!
//! This module maintains a persistent Python process that keeps the model
//! loaded between requests, significantly improving response time.

use serde::{Deserialize, Serialize};
use std::process::Stdio;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{mpsc, Mutex, RwLock};
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

/// Persistent Python bridge process
struct PersistentBridge {
    child: Child,
    stdin: tokio::process::ChildStdin,
    stdout_reader: BufReader<tokio::process::ChildStdout>,
}

impl PersistentBridge {
    async fn spawn(bridge_path: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        info!("Spawning persistent Python bridge: {}", bridge_path);

        let python_cmd = std::env::var("PYTHON_PATH").unwrap_or_else(|_| "python3".to_string());

        let mut child = Command::new(&python_cmd)
            .arg(bridge_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        let stdin = child.stdin.take().expect("Failed to get stdin");
        let stdout = child.stdout.take().expect("Failed to get stdout");
        let stderr = child.stderr.take().expect("Failed to get stderr");

        // Spawn stderr reader for debugging
        tokio::spawn(async move {
            let mut reader = BufReader::new(stderr);
            let mut line = String::new();
            while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
                warn!("Python stderr: {}", line.trim());
                line.clear();
            }
        });

        let stdout_reader = BufReader::new(stdout);

        Ok(Self {
            child,
            stdin,
            stdout_reader,
        })
    }

    async fn send_request(
        &mut self,
        request: &TTSRequest,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let json = serde_json::to_string(request)?;
        self.stdin.write_all(json.as_bytes()).await?;
        self.stdin.write_all(b"\n").await?;
        self.stdin.flush().await?;
        Ok(())
    }

    async fn read_event(&mut self) -> Result<Option<TTSEvent>, Box<dyn std::error::Error + Send + Sync>> {
        let mut line = String::new();
        match self.stdout_reader.read_line(&mut line).await {
            Ok(0) => Ok(None), // EOF
            Ok(_) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    return Ok(None);
                }
                let event: PythonEvent = serde_json::from_str(trimmed)?;
                Ok(Some(parse_python_event(event)))
            }
            Err(e) => Err(Box::new(e)),
        }
    }

    fn is_running(&mut self) -> bool {
        match self.child.try_wait() {
            Ok(None) => true,  // Still running
            Ok(Some(_)) => false,  // Exited
            Err(_) => false,  // Error checking
        }
    }
}

/// TTS Bridge with persistent process management
pub struct TTSBridge {
    bridge_path: String,
    process: Arc<RwLock<Option<PersistentBridge>>>,
    request_lock: Arc<Mutex<()>>,
}

impl TTSBridge {
    pub fn new(bridge_path: &str) -> Self {
        Self {
            bridge_path: bridge_path.to_string(),
            process: Arc::new(RwLock::new(None)),
            request_lock: Arc::new(Mutex::new(())),
        }
    }

    pub async fn generate_stream(
        &self,
        request: TTSRequest,
    ) -> Result<mpsc::Receiver<TTSEvent>, Box<dyn std::error::Error + Send + Sync>> {
        let (tx, rx) = mpsc::channel(100);

        // Clone Arc references for the spawned task
        let process = self.process.clone();
        let request_lock = self.request_lock.clone();
        let bridge_path = self.bridge_path.clone();

        tokio::spawn(async move {
            // Acquire request lock to serialize requests to the single Python process
            let _request_guard = request_lock.lock().await;

            // Ensure process is running
            {
                let mut process_guard = process.write().await;
                let needs_spawn = match process_guard.as_mut() {
                    None => true,
                    Some(bridge) => !bridge.is_running(),
                };

                if needs_spawn {
                    info!("Starting new Python bridge process");
                    match PersistentBridge::spawn(&bridge_path).await {
                        Ok(bridge) => *process_guard = Some(bridge),
                        Err(e) => {
                            error!("Failed to spawn bridge: {}", e);
                            let _ = tx.send(TTSEvent::Error {
                                error: e.to_string(),
                            }).await;
                            return;
                        }
                    }
                }
            }

            // Get mutable access to the process and send request
            {
                let mut process_guard = process.write().await;
                let bridge = match process_guard.as_mut() {
                    Some(b) => b,
                    None => {
                        let _ = tx.send(TTSEvent::Error {
                            error: "No bridge process".to_string(),
                        }).await;
                        return;
                    }
                };

                // Send request
                info!("Sending request to Python bridge");
                if let Err(e) = bridge.send_request(&request).await {
                    error!("Failed to send request: {}", e);
                    *process_guard = None;
                    let _ = tx.send(TTSEvent::Error {
                        error: e.to_string(),
                    }).await;
                    return;
                }

                // Read events until complete or error
                loop {
                    match bridge.read_event().await {
                        Ok(Some(event)) => {
                            let is_terminal = matches!(
                                event,
                                TTSEvent::Complete { .. } | TTSEvent::Error { .. }
                            );

                            if tx.send(event).await.is_err() {
                                warn!("Receiver dropped");
                                break;
                            }

                            if is_terminal {
                                break;
                            }
                        }
                        Ok(None) => {
                            // EOF or empty line, check if process died
                            if !bridge.is_running() {
                                let _ = tx.send(TTSEvent::Error {
                                    error: "Python bridge process died".to_string(),
                                }).await;
                                *process_guard = None;
                            }
                            break;
                        }
                        Err(e) => {
                            error!("Error reading from bridge: {}", e);
                            let _ = tx.send(TTSEvent::Error {
                                error: e.to_string(),
                            }).await;
                            *process_guard = None;
                            break;
                        }
                    }
                }
            }
        });

        Ok(rx)
    }
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
