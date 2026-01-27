/**
 * Audio Streamer for Dia2 TTS
 * Handles real-time gapless audio playback of streaming WAV chunks
 */

class AudioStreamer {
    constructor(options = {}) {
        this.audioContext = null;
        this.gainNode = null;
        this.analyser = null;

        // Chunk storage - indexed by chunk number for proper ordering
        this.chunkBuffers = new Map(); // chunkIndex -> AudioBuffer
        this.nextExpectedChunk = 0;
        this.highestChunkReceived = -1;

        // Playback state
        this.isPlaying = false;
        this.isPaused = false;
        this.scheduledSources = []; // Track scheduled sources for cleanup
        this.nextPlayTime = 0; // When the next chunk should start
        this.playbackStartTime = 0; // When playback began
        this.totalScheduledDuration = 0;
        this.lastScheduledChunk = -1;

        // Scheduling settings
        this.scheduleAheadTime = 0.1; // Schedule 100ms ahead
        this.schedulerInterval = null;

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

            console.log('[AudioStreamer] Initialized');
        } catch (error) {
            console.error('[AudioStreamer] Failed to initialize AudioContext:', error);
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

            // Store by index for proper ordering
            this.chunkBuffers.set(chunkIndex, audioBuffer);

            // Track if this is our first chunk (regardless of index)
            const isFirstChunk = this.highestChunkReceived === -1;
            this.highestChunkReceived = Math.max(this.highestChunkReceived, chunkIndex);

            // Auto-play if enabled and this is the first chunk we've received (not necessarily index 0)
            if (this.autoPlay && isFirstChunk && !this.isPlaying) {
                console.log(`[AudioStreamer] Auto-playing first chunk (index=${chunkIndex})`);
                await this.play();
            }

            // If already playing, try to schedule newly arrived chunks
            if (this.isPlaying && !this.isPaused) {
                this._scheduleAvailableChunks();
            }

            return audioBuffer.duration;
        } catch (error) {
            console.error(`[AudioStreamer] Failed to decode chunk ${chunkIndex}:`, error);
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

        console.log(`[AudioStreamer] play() called, state: ${this.audioContext.state}, chunks: ${this.chunkBuffers.size}`);

        // Resume audio context if suspended
        if (this.audioContext.state === 'suspended') {
            console.log('[AudioStreamer] Resuming suspended AudioContext');
            await this.audioContext.resume();
        }

        if (!this.isPlaying) {
            this.isPlaying = true;
            this.isPaused = false;

            // Initialize playback timing
            this.playbackStartTime = this.audioContext.currentTime;
            this.nextPlayTime = this.audioContext.currentTime;
            this.totalScheduledDuration = 0;

            // Find the minimum chunk index we have, and set lastScheduledChunk to one less
            const chunkIndices = Array.from(this.chunkBuffers.keys()).sort((a, b) => a - b);
            if (chunkIndices.length > 0) {
                this.lastScheduledChunk = chunkIndices[0] - 1;
                console.log(`[AudioStreamer] Starting playback: available chunks=[${chunkIndices.join(',')}], lastScheduledChunk=${this.lastScheduledChunk}`);
            } else {
                this.lastScheduledChunk = -1;
                console.log('[AudioStreamer] Starting playback with no chunks yet');
            }

            // Start the scheduler
            this._startScheduler();
            this._startUpdateLoop();

            // Schedule initial chunks
            this._scheduleAvailableChunks();
        }
    }

    /**
     * Schedule all available chunks in order
     */
    _scheduleAvailableChunks() {
        // Debug: log current state
        const chunkIndices = Array.from(this.chunkBuffers.keys()).sort((a, b) => a - b);

        // First, check if we have ANY unscheduled chunks
        const unscheduledChunks = chunkIndices.filter(idx => idx > this.lastScheduledChunk);

        if (unscheduledChunks.length > 0) {
            console.log(`[AudioStreamer] Scheduling: lastScheduled=${this.lastScheduledChunk}, available=[${chunkIndices.join(',')}], unscheduled=[${unscheduledChunks.join(',')}]`);
        }

        // Schedule chunks in order starting from the next expected
        // But also handle gaps - if we're missing chunk N, skip to the next available
        while (true) {
            const nextExpected = this.lastScheduledChunk + 1;

            if (this.chunkBuffers.has(nextExpected)) {
                // Normal case: next chunk is available
                const buffer = this.chunkBuffers.get(nextExpected);
                this._scheduleChunk(buffer, nextExpected);
                this.lastScheduledChunk = nextExpected;
            } else {
                // Check if there are any chunks with higher indices we should schedule
                // This handles cases where chunk indices skip (e.g., 0, 2, 4 instead of 0, 1, 2)
                const nextAvailable = chunkIndices.find(idx => idx > this.lastScheduledChunk);

                if (nextAvailable !== undefined && nextAvailable !== nextExpected) {
                    console.warn(`[AudioStreamer] Gap detected! Expected chunk ${nextExpected}, scheduling ${nextAvailable} instead`);
                    const buffer = this.chunkBuffers.get(nextAvailable);
                    this._scheduleChunk(buffer, nextAvailable);
                    this.lastScheduledChunk = nextAvailable;
                } else {
                    // No more chunks to schedule
                    break;
                }
            }
        }
    }

    /**
     * Schedule a single chunk for playback at the correct time
     */
    _scheduleChunk(buffer, chunkIndex) {
        const source = this.audioContext.createBufferSource();
        source.buffer = buffer;
        source.connect(this.gainNode);

        // Schedule to play at the exact right time
        const startTime = this.nextPlayTime;
        source.start(startTime);

        console.log(`[AudioStreamer] Scheduled chunk ${chunkIndex} at ${startTime.toFixed(3)}s, duration: ${buffer.duration.toFixed(3)}s`);

        // Track this source
        this.scheduledSources.push({
            source,
            chunkIndex,
            startTime,
            endTime: startTime + buffer.duration
        });

        // Update timing for next chunk
        this.nextPlayTime = startTime + buffer.duration;
        this.totalScheduledDuration += buffer.duration;

        // Notify chunk play when it actually starts
        const timeUntilStart = (startTime - this.audioContext.currentTime) * 1000;
        if (timeUntilStart > 0) {
            setTimeout(() => this.onChunkPlay(chunkIndex), timeUntilStart);
        } else {
            this.onChunkPlay(chunkIndex);
        }

        // Handle source end
        source.onended = () => {
            // Remove from tracked sources
            this.scheduledSources = this.scheduledSources.filter(s => s.source !== source);

            // Check if playback is complete
            if (this.scheduledSources.length === 0 && this.lastScheduledChunk === this.highestChunkReceived) {
                // Wait a moment to see if more chunks arrive
                setTimeout(() => {
                    if (this.scheduledSources.length === 0 && this.lastScheduledChunk === this.highestChunkReceived) {
                        console.log('[AudioStreamer] Playback complete');
                        this.isPlaying = false;
                        this.onPlaybackEnd();
                        this._stopScheduler();
                        this._stopUpdateLoop();
                    }
                }, 200);
            }
        };
    }

    /**
     * Start the scheduler that checks for new chunks to schedule
     */
    _startScheduler() {
        if (this.schedulerInterval) return;

        this.schedulerInterval = setInterval(() => {
            if (this.isPlaying && !this.isPaused) {
                this._scheduleAvailableChunks();
            }
        }, 50); // Check every 50ms
    }

    /**
     * Stop the scheduler
     */
    _stopScheduler() {
        if (this.schedulerInterval) {
            clearInterval(this.schedulerInterval);
            this.schedulerInterval = null;
        }
    }

    /**
     * Pause playback
     */
    pause() {
        if (!this.isPlaying) return;

        this.isPaused = true;
        this.isPlaying = false;

        // Stop all scheduled sources
        for (const scheduled of this.scheduledSources) {
            try {
                scheduled.source.stop();
            } catch (e) {
                // Ignore if already stopped
            }
        }
        this.scheduledSources = [];

        this._stopScheduler();
        this._stopUpdateLoop();
    }

    /**
     * Stop playback and reset
     */
    stop() {
        this.isPlaying = false;
        this.isPaused = false;

        // Stop all scheduled sources
        for (const scheduled of this.scheduledSources) {
            try {
                scheduled.source.stop();
            } catch (e) {
                // Ignore if already stopped
            }
        }
        this.scheduledSources = [];

        this._stopScheduler();
        this._stopUpdateLoop();
    }

    /**
     * Clear all chunks and reset state
     */
    reset() {
        this.stop();
        this.chunkBuffers.clear();
        this.nextExpectedChunk = 0;
        this.highestChunkReceived = -1;
        this.lastScheduledChunk = -1;
        this.nextPlayTime = 0;
        this.playbackStartTime = 0;
        this.totalScheduledDuration = 0;
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
            return 0;
        }

        const elapsed = this.audioContext.currentTime - this.playbackStartTime;
        return Math.min(elapsed, this.totalScheduledDuration);
    }

    /**
     * Get total duration of all received chunks
     */
    getTotalDuration() {
        let total = 0;
        for (const buffer of this.chunkBuffers.values()) {
            total += buffer.duration;
        }
        return total;
    }

    /**
     * Get chunk count
     */
    getChunkCount() {
        return this.chunkBuffers.size;
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
     * Play a single chunk by index (for debugging/testing)
     * Stops any current playback and plays just the specified chunk
     */
    async playSingleChunk(chunkIndex) {
        if (!this.audioContext) {
            await this.init();
        }

        // Resume audio context if suspended
        if (this.audioContext.state === 'suspended') {
            await this.audioContext.resume();
        }

        // Check if chunk exists
        if (!this.chunkBuffers.has(chunkIndex)) {
            console.warn(`[AudioStreamer] Chunk ${chunkIndex} not found`);
            return false;
        }

        // Stop any current playback
        this.stop();

        const buffer = this.chunkBuffers.get(chunkIndex);
        const source = this.audioContext.createBufferSource();
        source.buffer = buffer;
        source.connect(this.gainNode);

        console.log(`[AudioStreamer] Playing single chunk ${chunkIndex}: ${buffer.duration.toFixed(3)}s`);

        // Track this source for cleanup
        this.scheduledSources.push({
            source,
            chunkIndex,
            startTime: this.audioContext.currentTime,
            endTime: this.audioContext.currentTime + buffer.duration
        });

        source.start();
        this.isPlaying = true;

        // Notify chunk play
        this.onChunkPlay(chunkIndex);

        // Start update loop for visualizer
        this._startUpdateLoop();

        source.onended = () => {
            this.scheduledSources = this.scheduledSources.filter(s => s.source !== source);
            this.isPlaying = false;
            this._stopUpdateLoop();
            this.onPlaybackEnd();
        };

        return true;
    }

    /**
     * Get a specific chunk's buffer (for inspection)
     */
    getChunkBuffer(chunkIndex) {
        return this.chunkBuffers.get(chunkIndex) || null;
    }

    /**
     * Internal: Start update loop for time and visualizer
     */
    _startUpdateLoop() {
        const update = () => {
            if (!this.isPlaying) return;

            const currentTime = this.getCurrentTime();
            this.onTimeUpdate(currentTime, this.getTotalDuration());

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
