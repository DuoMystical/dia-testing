/**
 * Seamless Audio Streamer using AudioWorklet
 *
 * Unlike the chunk-based approach, this streams samples continuously
 * through a worklet processor, eliminating gaps and pops between chunks.
 */

class SeamlessAudioStreamer {
    constructor(options = {}) {
        this.audioContext = null;
        this.workletNode = null;
        this.gainNode = null;
        this.analyser = null;

        // State
        this.isInitialized = false;
        this.isPlaying = false;
        this.samplesBuffered = 0;
        this.chunksReceived = 0;
        this.totalSamplesDecoded = 0;

        // Settings
        this.volume = options.volume || 0.8;
        this.minBufferBeforePlay = options.minBufferBeforePlay || 4800; // 100ms at 48kHz

        // Callbacks
        this.onBufferLevel = options.onBufferLevel || (() => {});
        this.onPlaybackStart = options.onPlaybackStart || (() => {});
        this.onUnderrun = options.onUnderrun || (() => {});
        this.onError = options.onError || (() => {});

        // Animation frame for visualization
        this.animationFrame = null;
        this.onVisualizerData = options.onVisualizerData || (() => {});
    }

    async init() {
        if (this.isInitialized) return;

        try {
            // Create AudioContext
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();

            // Load the worklet processor
            const workletUrl = new URL('./audio-worklet-processor.js', import.meta.url).href;
            await this.audioContext.audioWorklet.addModule(workletUrl);

            // Create the worklet node
            this.workletNode = new AudioWorkletNode(this.audioContext, 'streaming-audio-processor');

            // Handle messages from worklet
            this.workletNode.port.onmessage = (event) => {
                const { type, samplesAvailable, bufferSeconds, underrunCount } = event.data;

                if (type === 'bufferLevel') {
                    this.samplesBuffered = samplesAvailable;
                    this.onBufferLevel(samplesAvailable, bufferSeconds);

                    // Auto-start playback when we have enough buffer
                    if (!this.isPlaying && samplesAvailable >= this.minBufferBeforePlay) {
                        this._startPlayback();
                    }
                } else if (type === 'status') {
                    console.log('[SeamlessStreamer] Status:', event.data);
                }
            };

            // Create gain node for volume control
            this.gainNode = this.audioContext.createGain();
            this.gainNode.gain.value = this.volume;

            // Create analyser for visualization
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 256;
            this.analyser.smoothingTimeConstant = 0.8;

            // Connect: worklet -> gain -> analyser -> destination
            this.workletNode.connect(this.gainNode);
            this.gainNode.connect(this.analyser);
            this.analyser.connect(this.audioContext.destination);

            this.isInitialized = true;
            console.log('[SeamlessStreamer] Initialized with AudioWorklet');

        } catch (error) {
            console.error('[SeamlessStreamer] Failed to initialize:', error);

            // Fallback: try loading worklet from different path
            if (error.message.includes('module')) {
                try {
                    await this.audioContext.audioWorklet.addModule('/js/audio-worklet-processor.js');
                    console.log('[SeamlessStreamer] Loaded worklet from fallback path');
                } catch (fallbackError) {
                    console.error('[SeamlessStreamer] Fallback also failed:', fallbackError);
                    this.onError(fallbackError);
                    throw fallbackError;
                }
            } else {
                this.onError(error);
                throw error;
            }
        }
    }

    /**
     * Add a WAV chunk to the stream
     * The WAV is decoded and samples are sent to the worklet
     */
    async addChunk(wavData, chunkIndex) {
        if (!this.isInitialized) {
            await this.init();
        }

        // Resume context if suspended (browser autoplay policy)
        if (this.audioContext.state === 'suspended') {
            await this.audioContext.resume();
        }

        try {
            // Decode the WAV data
            const audioBuffer = await this.audioContext.decodeAudioData(wavData.slice(0));
            const samples = audioBuffer.getChannelData(0);

            // Send samples to worklet
            // Note: We need to copy the data because it might be detached
            const samplesCopy = new Float32Array(samples);
            this.workletNode.port.postMessage({
                type: 'samples',
                samples: samplesCopy
            }, [samplesCopy.buffer]); // Transfer ownership for performance

            this.chunksReceived++;
            this.totalSamplesDecoded += samples.length;

            console.log(`[SeamlessStreamer] Added chunk ${chunkIndex}: ${samples.length} samples (${(samples.length / this.audioContext.sampleRate * 1000).toFixed(1)}ms)`);

            return audioBuffer.duration;

        } catch (error) {
            console.error(`[SeamlessStreamer] Failed to decode chunk ${chunkIndex}:`, error);
            return 0;
        }
    }

    /**
     * Start playback (called automatically when buffer is ready)
     */
    _startPlayback() {
        if (this.isPlaying) return;

        this.workletNode.port.postMessage({ type: 'command', command: 'start' });
        this.isPlaying = true;
        this.onPlaybackStart();
        this._startVisualizerLoop();

        console.log('[SeamlessStreamer] Playback started');
    }

    /**
     * Manually start playback (if auto-start is disabled)
     */
    async play() {
        if (!this.isInitialized) {
            await this.init();
        }

        if (this.audioContext.state === 'suspended') {
            await this.audioContext.resume();
        }

        this._startPlayback();
    }

    /**
     * Stop playback
     */
    stop() {
        if (!this.isInitialized) return;

        this.workletNode.port.postMessage({ type: 'command', command: 'stop' });
        this.isPlaying = false;
        this._stopVisualizerLoop();

        console.log('[SeamlessStreamer] Playback stopped');
    }

    /**
     * Reset the streamer (clear buffer, stop playback)
     */
    reset() {
        if (!this.isInitialized) return;

        this.workletNode.port.postMessage({ type: 'command', command: 'reset' });
        this.isPlaying = false;
        this.samplesBuffered = 0;
        this.chunksReceived = 0;
        this.totalSamplesDecoded = 0;
        this._stopVisualizerLoop();

        console.log('[SeamlessStreamer] Reset');
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
            samplesBuffered: this.samplesBuffered,
            chunksReceived: this.chunksReceived,
            totalSamplesDecoded: this.totalSamplesDecoded,
            contextState: this.audioContext?.state,
            sampleRate: this.audioContext?.sampleRate
        };
    }

    /**
     * Visualizer loop for waveform display
     */
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
        this.stop();
        this._stopVisualizerLoop();

        if (this.workletNode) {
            this.workletNode.disconnect();
            this.workletNode = null;
        }
        if (this.gainNode) {
            this.gainNode.disconnect();
            this.gainNode = null;
        }
        if (this.analyser) {
            this.analyser.disconnect();
            this.analyser = null;
        }
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }

        this.isInitialized = false;
        console.log('[SeamlessStreamer] Disposed');
    }
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SeamlessAudioStreamer;
}
