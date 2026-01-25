//! Session management for WebSocket connections

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::info;

/// Tracks active WebSocket sessions
pub struct SessionManager {
    sessions: HashMap<String, SessionInfo>,
    total_connections: AtomicU64,
}

pub struct SessionInfo {
    pub created_at: std::time::Instant,
    pub messages_received: u64,
    pub audio_chunks_sent: u64,
}

impl SessionManager {
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
            total_connections: AtomicU64::new(0),
        }
    }

    pub fn add_session(&mut self, session_id: &str) {
        self.sessions.insert(
            session_id.to_string(),
            SessionInfo {
                created_at: std::time::Instant::now(),
                messages_received: 0,
                audio_chunks_sent: 0,
            },
        );
        self.total_connections.fetch_add(1, Ordering::SeqCst);
        info!("Session added: {} (total: {})", session_id, self.active_count());
    }

    pub fn remove_session(&mut self, session_id: &str) {
        if let Some(info) = self.sessions.remove(session_id) {
            let duration = info.created_at.elapsed();
            info!(
                "Session removed: {} (duration: {:.1}s, messages: {}, chunks: {})",
                session_id,
                duration.as_secs_f64(),
                info.messages_received,
                info.audio_chunks_sent
            );
        }
    }

    pub fn active_count(&self) -> usize {
        self.sessions.len()
    }

    pub fn total_connections(&self) -> u64 {
        self.total_connections.load(Ordering::SeqCst)
    }

    pub fn increment_messages(&mut self, session_id: &str) {
        if let Some(info) = self.sessions.get_mut(session_id) {
            info.messages_received += 1;
        }
    }

    pub fn increment_audio_chunks(&mut self, session_id: &str) {
        if let Some(info) = self.sessions.get_mut(session_id) {
            info.audio_chunks_sent += 1;
        }
    }
}

impl Default for SessionManager {
    fn default() -> Self {
        Self::new()
    }
}
