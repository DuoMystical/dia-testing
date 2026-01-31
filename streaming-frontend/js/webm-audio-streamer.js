/**
 * WebM Audio Streamer using MediaSource Extensions
 *
 * Plays a continuous WebM/Opus stream using MSE for gapless playback.
 * Unlike chunk-based approaches, this appends bytes to a single continuous
 * stream, eliminating pops/clicks at chunk boundaries.
 */

class WebMAudioStreamer {
    constructor(options = {}) {
        this.mediaSource = null;
        this.sourceBuffer = null;
        this.audioElement = null;
        this.analyser = null;
        this.audioContext = null;
        this.sourceNode = null;
        this.gainNode = null;

        // State
        this.isInitialized = false;
        this.isPlaying = false;
        this.chunksReceived = 0;
        this.bytesReceived = 0;
        this.pendingBuffers = [];
        this.isUpdating = false;

        // Settings
        this.volume = options.volume || 0.8;

        // Callbacks
        this.onPlaybackStart = options.onPlaybackStart || (() => {});
        this.onPlaybackEnd = options.onPlaybackEnd || (() => {});
        this.onError = options.onError || (() => {});
        this.onVisualizerData = options.onVisualizerData || (() => {});
        this.onBufferUpdate = options.onBufferUpdate || (() => {});

        // Animation frame for visualization
        this.animationFrame = null;
    }

    async init() {
        if (this.isInitialized) return;

        // Check for MSE support
        if (!window.MediaSource) {
            throw new Error('MediaSource Extensions not supported');
        }

        // Check for WebM/Opus support
        const mimeType = 'audio/webm; codecs="opus"';
        if (!MediaSource.isTypeSupported(mimeType)) {
            throw new Error(`MIME type ${mimeType} not supported`);
        }

        // Create audio element
        this.audioElement = document.createElement('audio');
        this.audioElement.preload = 'auto';

        // Create MediaSource
        this.mediaSource = new MediaSource();
        this.audioElement.src = URL.createObjectURL(this.mediaSource);

        // Wait for sourceopen
        await new Promise((resolve, reject) => {
            this.mediaSource.addEventListener('sourceopen', resolve, { once: true });
            this.mediaSource.addEventListener('error', reject, { once: true });
            setTimeout(() => reject(new Error('MediaSource timeout')), 5000);
        });

        // Add source buffer
        this.sourceBuffer = this.mediaSource.addSourceBuffer(mimeType);
        this.sourceBuffer.mode = 'sequence';

        // Handle buffer updates
        this.sourceBuffer.addEventListener('updateend', () => {
            this.isUpdating = false;
            this._appendNextPending();
        });

        this.sourceBuffer.addEventListener('error', (e) => {
            console.error('[WebMStreamer] SourceBuffer error:', e);
            this.onError(new Error('SourceBuffer error'));
        });

        // Create Web Audio API nodes for visualization
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        this.sourceNode = this.audioContext.createMediaElementSource(this.audioElement);

        this.gainNode = this.audioContext.createGain();
        this.gainNode.gain.value = this.volume;

        this.analyser = this.audioContext.createAnalyser();
        this.analyser.fftSize = 256;
        this.analyser.smoothingTimeConstant = 0.8;

        // Connect: source -> gain -> analyser -> destination
        this.sourceNode.connect(this.gainNode);
        this.gainNode.connect(this.analyser);
        this.analyser.connect(this.audioContext.destination);

        // Audio element events
        this.audioElement.addEventListener('playing', () => {
            this.isPlaying = true;
            this.onPlaybackStart();
            this._startVisualizerLoop();
        });

        this.audioElement.addEventListener('ended', () => {
            this.isPlaying = false;
            this.onPlaybackEnd();
            this._stopVisualizerLoop();
        });

        this.audioElement.addEventListener('pause', () => {
            this.isPlaying = false;
            this._stopVisualizerLoop();
        });

        this.isInitialized = true;
        console.log('[WebMStreamer] Initialized with MSE');
    }

    /**
     * Add WebM data to the stream
     */
    async addChunk(webmData, chunkIndex) {
        if (!this.isInitialized) {
            await this.init();
        }

        // Resume AudioContext if suspended
        if (this.audioContext.state === 'suspended') {
            await this.audioContext.resume();
        }

        // Convert to ArrayBuffer if needed
        const buffer = webmData instanceof ArrayBuffer ? webmData : webmData.buffer;

        this.chunksReceived++;
        this.bytesReceived += buffer.byteLength;

        console.log(`[WebMStreamer] Received chunk ${chunkIndex}: ${buffer.byteLength} bytes, total: ${this.bytesReceived}`);

        // Queue for appending
        this.pendingBuffers.push(buffer);
        this._appendNextPending();

        // Auto-play when we have enough data
        if (this.chunksReceived >= 2 && !this.isPlaying && this.audioElement.paused) {
            try {
                await this.audioElement.play();
            } catch (e) {
                console.warn('[WebMStreamer] Auto-play blocked:', e.message);
            }
        }

        // Estimate duration based on typical bitrate
        const estimatedDuration = buffer.byteLength / 12000; // ~96kbps Opus
        return estimatedDuration;
    }

    _appendNextPending() {
        if (this.isUpdating || this.pendingBuffers.length === 0) {
            return;
        }

        if (!this.sourceBuffer || this.mediaSource.readyState !== 'open') {
            return;
        }

        this.isUpdating = true;
        const buffer = this.pendingBuffers.shift();

        try {
            this.sourceBuffer.appendBuffer(buffer);
            this.onBufferUpdate(this.bytesReceived, this.chunksReceived);
        } catch (e) {
            console.error('[WebMStreamer] Failed to append buffer:', e);
            this.isUpdating = false;
        }
    }

    /**
     * Signal that the stream is complete
     */
    endOfStream() {
        if (this.mediaSource && this.mediaSource.readyState === 'open') {
            // Wait for all pending buffers to be appended
            const waitForBuffers = () => {
                if (this.pendingBuffers.length > 0 || this.isUpdating) {
                    setTimeout(waitForBuffers, 50);
                    return;
                }
                try {
                    this.mediaSource.endOfStream();
                    console.log('[WebMStreamer] Stream ended');
                } catch (e) {
                    console.warn('[WebMStreamer] endOfStream error:', e);
                }
            };
            waitForBuffers();
        }
    }

    /**
     * Play the stream
     */
    async play() {
        if (!this.isInitialized) {
            await this.init();
        }

        if (this.audioContext.state === 'suspended') {
            await this.audioContext.resume();
        }

        try {
            await this.audioElement.play();
            this.isPlaying = true;
        } catch (e) {
            console.error('[WebMStreamer] Play failed:', e);
            throw e;
        }
    }

    /**
     * Pause playback
     */
    pause() {
        if (this.audioElement) {
            this.audioElement.pause();
            this.isPlaying = false;
        }
    }

    /**
     * Reset for a new stream
     */
    reset() {
        this._stopVisualizerLoop();

        if (this.audioElement) {
            this.audioElement.pause();
            this.audioElement.currentTime = 0;
        }

        // Clean up old MediaSource
        if (this.sourceBuffer && this.mediaSource?.readyState === 'open') {
            try {
                this.mediaSource.removeSourceBuffer(this.sourceBuffer);
            } catch (e) {}
        }

        if (this.audioElement?.src) {
            URL.revokeObjectURL(this.audioElement.src);
        }

        // Clean up Web Audio nodes - must close context since sourceNode can't be reused
        if (this.sourceNode) {
            try {
                this.sourceNode.disconnect();
            } catch (e) {}
            this.sourceNode = null;
        }
        if (this.gainNode) {
            try {
                this.gainNode.disconnect();
            } catch (e) {}
            this.gainNode = null;
        }
        if (this.analyser) {
            try {
                this.analyser.disconnect();
            } catch (e) {}
            this.analyser = null;
        }
        if (this.audioContext) {
            try {
                this.audioContext.close();
            } catch (e) {}
            this.audioContext = null;
        }

        // Reset state
        this.mediaSource = null;
        this.sourceBuffer = null;
        this.audioElement = null;
        this.pendingBuffers = [];
        this.isUpdating = false;
        this.isPlaying = false;
        this.chunksReceived = 0;
        this.bytesReceived = 0;
        this.isInitialized = false;

        console.log('[WebMStreamer] Reset');
    }

    /**
     * Set volume (0.0 to 1.0)
     */
    setVolume(volume) {
        this.volume = Math.max(0, Math.min(1, volume));
        if (this.gainNode) {
            this.gainNode.gain.value = this.volume;
        }
    }

    /**
     * Get current status
     */
    getStatus() {
        return {
            isInitialized: this.isInitialized,
            isPlaying: this.isPlaying,
            chunksReceived: this.chunksReceived,
            bytesReceived: this.bytesReceived,
            currentTime: this.audioElement?.currentTime || 0,
            duration: this.audioElement?.duration || 0,
            buffered: this.audioElement?.buffered?.length > 0
                ? this.audioElement.buffered.end(0)
                : 0,
            pendingBuffers: this.pendingBuffers.length,
            readyState: this.mediaSource?.readyState
        };
    }

    getChunkCount() {
        return this.chunksReceived;
    }

    getTotalDuration() {
        return this.audioElement?.duration || 0;
    }

    _startVisualizerLoop() {
        if (this.animationFrame) return;

        const update = () => {
            if (!this.analyser) return;

            const dataArray = new Uint8Array(this.analyser.frequencyBinCount);
            this.analyser.getByteFrequencyData(dataArray);
            this.onVisualizerData(dataArray);

            this.animationFrame = requestAnimationFrame(update);
        };

        update();
    }

    _stopVisualizerLoop() {
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
            this.animationFrame = null;
        }
    }

    /**
     * Clean up resources
     */
    dispose() {
        this.reset();

        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }

        if (this.audioElement) {
            this.audioElement.remove();
            this.audioElement = null;
        }

        console.log('[WebMStreamer] Disposed');
    }
}

// Export for use
if (typeof window !== 'undefined') {
    window.WebMAudioStreamer = WebMAudioStreamer;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = WebMAudioStreamer;
}
