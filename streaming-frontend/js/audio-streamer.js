/**
 * Audio Streamer for Dia2 TTS
 * Handles real-time audio playback of streaming WAV chunks
 */

class AudioStreamer {
    constructor(options = {}) {
        this.audioContext = null;
        this.gainNode = null;
        this.analyser = null;
        this.chunks = [];
        this.currentChunkIndex = 0;
        this.isPlaying = false;
        this.isPaused = false;
        this.currentSource = null;
        this.startTime = 0;
        this.pauseTime = 0;
        this.totalDuration = 0;

        // Playback settings
        this.volume = options.volume || 0.8;
        this.autoPlay = options.autoPlay !== false;

        // Callbacks
        this.onChunkPlay = options.onChunkPlay || (() => {});
        this.onPlaybackEnd = options.onPlaybackEnd || (() => {});
        this.onTimeUpdate = options.onTimeUpdate || (() => {});
        this.onVisualizerData = options.onVisualizerData || (() => {});

        // Animation frame for updates
        this.animationFrame = null;
    }

    async init() {
        if (this.audioContext) return;

        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();

            // Create gain node for volume control
            this.gainNode = this.audioContext.createGain();
            this.gainNode.gain.value = this.volume;

            // Create analyser for visualization
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 256;
            this.analyser.smoothingTimeConstant = 0.8;

            // Connect nodes
            this.gainNode.connect(this.analyser);
            this.analyser.connect(this.audioContext.destination);

            console.log('AudioStreamer initialized');
        } catch (error) {
            console.error('Failed to initialize AudioContext:', error);
            throw error;
        }
    }

    /**
     * Add a WAV chunk to the queue
     */
    async addChunk(wavData, chunkIndex) {
        if (!this.audioContext) {
            await this.init();
        }

        try {
            // Debug: log chunk info
            console.log(`[AudioStreamer] Adding chunk ${chunkIndex}, data size: ${wavData.byteLength} bytes`);

            // Validate WAV header
            const view = new DataView(wavData);
            const riff = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
            if (riff !== 'RIFF') {
                console.error(`[AudioStreamer] Invalid WAV header: expected RIFF, got ${riff}`);
                return 0;
            }

            // Decode the WAV data
            const audioBuffer = await this.audioContext.decodeAudioData(wavData.slice(0));

            console.log(`[AudioStreamer] Chunk ${chunkIndex} decoded: ${audioBuffer.duration.toFixed(3)}s, ${audioBuffer.sampleRate}Hz`);

            this.chunks.push({
                buffer: audioBuffer,
                index: chunkIndex,
                duration: audioBuffer.duration,
                played: false
            });

            this.totalDuration = this.chunks.reduce((sum, c) => sum + c.duration, 0);

            // Auto-play if enabled and this is the first chunk
            if (this.autoPlay && this.chunks.length === 1 && !this.isPlaying) {
                console.log('[AudioStreamer] Auto-playing first chunk');
                this.play();
            }

            return audioBuffer.duration;
        } catch (error) {
            console.error(`[AudioStreamer] Failed to decode chunk ${chunkIndex}:`, error);
            console.error('[AudioStreamer] Data size:', wavData.byteLength);
            // Log first few bytes for debugging
            if (wavData.byteLength > 0) {
                const bytes = new Uint8Array(wavData.slice(0, 16));
                console.error('[AudioStreamer] First 16 bytes:', Array.from(bytes).map(b => b.toString(16).padStart(2, '0')).join(' '));
            }
            return 0;
        }
    }

    /**
     * Start or resume playback
     */
    async play() {
        if (!this.audioContext) {
            await this.init();
        }

        console.log(`[AudioStreamer] play() called, audioContext.state: ${this.audioContext.state}, chunks: ${this.chunks.length}`);

        // Resume audio context if suspended
        if (this.audioContext.state === 'suspended') {
            console.log('[AudioStreamer] Resuming suspended AudioContext');
            await this.audioContext.resume();
        }

        if (this.isPaused) {
            // Resume from pause
            console.log('[AudioStreamer] Resuming from pause');
            this.isPaused = false;
            this.isPlaying = true;
            this._playFromIndex(this.currentChunkIndex, this.pauseTime);
        } else if (!this.isPlaying) {
            // Start fresh
            console.log('[AudioStreamer] Starting fresh playback');
            this.isPlaying = true;
            this.currentChunkIndex = 0;
            this._playFromIndex(0, 0);
        }

        this._startUpdateLoop();
    }

    /**
     * Pause playback
     */
    pause() {
        if (!this.isPlaying) return;

        this.isPaused = true;
        this.isPlaying = false;
        this.pauseTime = this.getCurrentTime();

        if (this.currentSource) {
            try {
                this.currentSource.stop();
            } catch (e) {
                // Ignore if already stopped
            }
            this.currentSource = null;
        }

        this._stopUpdateLoop();
    }

    /**
     * Stop playback and reset
     */
    stop() {
        this.isPlaying = false;
        this.isPaused = false;
        this.currentChunkIndex = 0;
        this.pauseTime = 0;

        if (this.currentSource) {
            try {
                this.currentSource.stop();
            } catch (e) {
                // Ignore if already stopped
            }
            this.currentSource = null;
        }

        this._stopUpdateLoop();
    }

    /**
     * Clear all chunks and reset state
     */
    reset() {
        this.stop();
        this.chunks = [];
        this.totalDuration = 0;
        this.startTime = 0;
    }

    /**
     * Set volume (0-1)
     */
    setVolume(value) {
        this.volume = Math.max(0, Math.min(1, value));
        if (this.gainNode) {
            this.gainNode.gain.value = this.volume;
        }
    }

    /**
     * Get current playback time in seconds
     */
    getCurrentTime() {
        if (!this.isPlaying || !this.audioContext) {
            return this.pauseTime;
        }

        // Calculate time based on chunks played
        let time = 0;
        for (let i = 0; i < this.currentChunkIndex; i++) {
            if (this.chunks[i]) {
                time += this.chunks[i].duration;
            }
        }

        // Add time within current chunk
        if (this.startTime > 0) {
            time += this.audioContext.currentTime - this.startTime;
        }

        return Math.min(time, this.totalDuration);
    }

    /**
     * Get total duration
     */
    getTotalDuration() {
        return this.totalDuration;
    }

    /**
     * Get chunk count
     */
    getChunkCount() {
        return this.chunks.length;
    }

    /**
     * Get visualizer data
     */
    getVisualizerData() {
        if (!this.analyser) return new Uint8Array(0);

        const bufferLength = this.analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        this.analyser.getByteFrequencyData(dataArray);
        return dataArray;
    }

    /**
     * Internal: Play from a specific chunk index
     */
    _playFromIndex(index, timeOffset = 0) {
        console.log(`[AudioStreamer] _playFromIndex(${index}), chunks available: ${this.chunks.length}`);

        if (index >= this.chunks.length) {
            console.log('[AudioStreamer] No more chunks to play, ending playback');
            this.isPlaying = false;
            this.onPlaybackEnd();
            this._stopUpdateLoop();
            return;
        }

        const chunk = this.chunks[index];
        this.currentChunkIndex = index;

        console.log(`[AudioStreamer] Playing chunk ${index}, duration: ${chunk.duration.toFixed(3)}s`);

        // Create source for this chunk
        this.currentSource = this.audioContext.createBufferSource();
        this.currentSource.buffer = chunk.buffer;
        this.currentSource.connect(this.gainNode);

        // Handle chunk end
        this.currentSource.onended = () => {
            console.log(`[AudioStreamer] Chunk ${index} ended`);
            if (this.isPlaying && !this.isPaused) {
                chunk.played = true;
                this._playFromIndex(index + 1, 0);
            }
        };

        // Start playback
        this.startTime = this.audioContext.currentTime;

        if (timeOffset > 0 && timeOffset < chunk.duration) {
            this.currentSource.start(0, timeOffset);
        } else {
            this.currentSource.start(0);
        }

        this.onChunkPlay(index);
    }

    /**
     * Internal: Start update loop for time and visualizer
     */
    _startUpdateLoop() {
        const update = () => {
            if (!this.isPlaying) return;

            const currentTime = this.getCurrentTime();
            this.onTimeUpdate(currentTime, this.totalDuration);

            const visualizerData = this.getVisualizerData();
            this.onVisualizerData(visualizerData);

            this.animationFrame = requestAnimationFrame(update);
        };

        this.animationFrame = requestAnimationFrame(update);
    }

    /**
     * Internal: Stop update loop
     */
    _stopUpdateLoop() {
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
            this.animationFrame = null;
        }
    }
}

// Export for use in other modules
window.AudioStreamer = AudioStreamer;
