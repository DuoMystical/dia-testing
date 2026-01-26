/**
 * WebSocket Client for Dia2 Streaming TTS
 * Handles connection management, reconnection, and message handling
 */

class WebSocketClient {
    constructor(options = {}) {
        this.url = options.url || this._getDefaultUrl();
        this.reconnectInterval = options.reconnectInterval || 3000;
        this.maxReconnectAttempts = options.maxReconnectAttempts || 10;
        this.pingInterval = options.pingInterval || 30000; // 30 seconds
        this.reconnectAttempts = 0;
        this.socket = null;
        this.isConnected = false;
        this.sessionId = null;
        this.pingTimer = null;

        // Event handlers
        this.onConnect = options.onConnect || (() => {});
        this.onDisconnect = options.onDisconnect || (() => {});
        this.onMessage = options.onMessage || (() => {});
        this.onError = options.onError || (() => {});
        this.onAudioChunk = options.onAudioChunk || (() => {});
        this.onStatus = options.onStatus || (() => {});
        this.onComplete = options.onComplete || (() => {});

        // Bind methods
        this._handleOpen = this._handleOpen.bind(this);
        this._handleClose = this._handleClose.bind(this);
        this._handleMessage = this._handleMessage.bind(this);
        this._handleError = this._handleError.bind(this);
    }

    _getDefaultUrl() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.host;
        return `${protocol}//${host}/ws`;
    }

    connect() {
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            console.log('WebSocket already connected');
            return;
        }

        console.log(`Connecting to ${this.url}...`);

        try {
            this.socket = new WebSocket(this.url);
            this.socket.binaryType = 'arraybuffer';

            this.socket.addEventListener('open', this._handleOpen);
            this.socket.addEventListener('close', this._handleClose);
            this.socket.addEventListener('message', this._handleMessage);
            this.socket.addEventListener('error', this._handleError);
        } catch (error) {
            console.error('Failed to create WebSocket:', error);
            this._scheduleReconnect();
        }
    }

    disconnect() {
        this.reconnectAttempts = this.maxReconnectAttempts; // Prevent reconnection
        this._stopPing();
        if (this.socket) {
            this.socket.close();
            this.socket = null;
        }
        this.isConnected = false;
    }

    send(data) {
        if (!this.isConnected) {
            console.warn('Cannot send - not connected');
            return false;
        }

        try {
            const message = typeof data === 'string' ? data : JSON.stringify(data);
            this.socket.send(message);
            return true;
        } catch (error) {
            console.error('Failed to send message:', error);
            return false;
        }
    }

    /**
     * Send text in chunks for bidirectional streaming
     */
    sendTextChunks(text, chunkSize = 50) {
        const chunks = [];
        for (let i = 0; i < text.length; i += chunkSize) {
            chunks.push(text.slice(i, i + chunkSize));
        }

        chunks.forEach((chunk, index) => {
            this.send({
                type: 'text_chunk',
                text: chunk,
                chunk_index: index,
                final: index === chunks.length - 1
            });
        });

        return chunks.length;
    }

    /**
     * Send a generate request (non-streaming input)
     */
    generate(text, options = {}) {
        const config = {
            // Sampling parameters
            audio_temperature: options.audioTemperature || 0.8,
            audio_top_k: options.audioTopK || 50,
            text_temperature: options.textTemperature || 0.6,
            text_top_k: options.textTopK || 50,
            // CFG parameters
            cfg_scale: options.cfgScale || 2.0,
            cfg_filter_k: options.cfgFilterK || 50,
            // Streaming parameters
            chunk_size_frames: options.chunkSizeFrames || 32,
            min_chunk_frames: options.minChunkFrames || 16
        };

        // Add voice cloning audio if provided
        if (options.speaker1Audio) {
            config.speaker_1_audio = options.speaker1Audio;
        }
        if (options.speaker2Audio) {
            config.speaker_2_audio = options.speaker2Audio;
        }

        return this.send({
            type: 'generate',
            text: text,
            model: options.model || '2b',
            config: config
        });
    }

    /**
     * Cancel current generation
     */
    cancel() {
        return this.send({ type: 'cancel' });
    }

    _handleOpen(event) {
        console.log('WebSocket connected');
        this.isConnected = true;
        this.reconnectAttempts = 0;
        this._startPing();
        this.onConnect(event);
    }

    _startPing() {
        this._stopPing();
        this.pingTimer = setInterval(() => {
            if (this.isConnected) {
                this.send({ type: 'ping' });
            }
        }, this.pingInterval);
    }

    _stopPing() {
        if (this.pingTimer) {
            clearInterval(this.pingTimer);
            this.pingTimer = null;
        }
    }

    _handleClose(event) {
        console.log('WebSocket closed:', event.code, event.reason);
        this.isConnected = false;
        this.sessionId = null;
        this._stopPing();
        this.onDisconnect(event);

        if (event.code !== 1000) { // Not a normal close
            this._scheduleReconnect();
        }
    }

    _handleMessage(event) {
        try {
            const data = JSON.parse(event.data);
            this.onMessage(data);

            // Route to specific handlers
            switch (data.type) {
                case 'connected':
                    this.sessionId = data.session_id;
                    console.log('Session ID:', this.sessionId);
                    break;

                case 'audio':
                case 'audio_chunk':
                    // Decode base64 audio data
                    const audioBytes = this._base64ToArrayBuffer(data.data);
                    this.onAudioChunk({
                        data: audioBytes,
                        chunkIndex: data.chunk_index,
                        timestampMs: data.timestamp_ms
                    });
                    break;

                case 'status':
                    this.onStatus({
                        message: data.message,
                        progress: data.progress
                    });
                    break;

                case 'complete':
                    this.onComplete({
                        totalChunks: data.total_chunks,
                        totalDurationMs: data.total_duration_ms
                    });
                    break;

                case 'error':
                    this.onError(new Error(data.error));
                    break;

                case 'chunk_ack':
                    // Acknowledge text chunk received
                    break;

                case 'cancelled':
                    console.log('Generation cancelled');
                    break;

                case 'pong':
                    // Heartbeat response
                    break;
            }
        } catch (error) {
            console.error('Failed to parse message:', error);
        }
    }

    _handleError(event) {
        console.error('WebSocket error:', event);
        this.onError(new Error('WebSocket error'));
    }

    _scheduleReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.log('Max reconnection attempts reached');
            return;
        }

        this.reconnectAttempts++;
        console.log(`Reconnecting in ${this.reconnectInterval}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

        setTimeout(() => {
            this.connect();
        }, this.reconnectInterval);
    }

    _base64ToArrayBuffer(base64) {
        const binaryString = atob(base64);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }
        return bytes.buffer;
    }
}

// Export for use in other modules
window.WebSocketClient = WebSocketClient;
