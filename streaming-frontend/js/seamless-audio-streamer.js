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

        // Store decoded chunks for debugging/single-chunk playback
        this.storedChunks = new Map();
        this.singleChunkSource = null;
    }

    async init() {
        if (this.isInitialized) return;

        try {
            // Create AudioContext
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();

            // Load the worklet processor - try multiple paths
            const workletPaths = [
                '/js/audio-worklet-processor.js',
                './js/audio-worklet-processor.js',
                'js/audio-worklet-processor.js'
            ];

            let loaded = false;
            for (const path of workletPaths) {
                try {
                    await this.audioContext.audioWorklet.addModule(path);
                    console.log(`[SeamlessStreamer] Loaded worklet from: ${path}`);
                    loaded = true;
                    break;
                } catch (e) {
                    console.log(`[SeamlessStreamer] Failed to load from ${path}, trying next...`);
                }
            }

            if (!loaded) {
                throw new Error('Could not load audio worklet from any path');
            }

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
            this.onError(error);
            throw error;
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

            console.log(`[SeamlessStreamer] Decoded chunk ${chunkIndex}: ${samples.length} samples, first=${samples[0]?.toFixed(4)}, last=${samples[samples.length-1]?.toFixed(4)}`);

            // Store the AudioBuffer for single-chunk playback debugging
            this.storedChunks.set(chunkIndex, {
                buffer: audioBuffer,
                samples: samples,
                wavData: wavData.slice(0)  // Store copy of original WAV
            });

            // Send samples to worklet - use regular array to avoid transfer issues
            // Convert Float32Array to regular array for reliable cross-thread transfer
            const samplesArray = Array.from(samples);
            this.workletNode.port.postMessage({
                type: 'samples',
                samples: samplesArray
            });

            this.chunksReceived++;
            this.totalSamplesDecoded += samples.length;

            // Verify samples were sent correctly
            console.log(`[SeamlessStreamer] Sent chunk ${chunkIndex} to worklet (${samplesArray.length} samples), first 5: [${samplesArray.slice(0,5).map(s => s.toFixed(4)).join(', ')}]`);

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
        this.storedChunks.clear();
        this._stopVisualizerLoop();

        // Stop any single-chunk playback
        if (this.singleChunkSource) {
            try {
                this.singleChunkSource.stop();
            } catch (e) {}
            this.singleChunkSource = null;
        }

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
     * Play a single chunk directly (for debugging - bypasses the worklet)
     * This helps isolate whether pops come from the source audio or the streaming
     */
    async playSingleChunk(chunkIndex) {
        const chunk = this.storedChunks.get(chunkIndex);
        if (!chunk) {
            console.error(`[SeamlessStreamer] Chunk ${chunkIndex} not found. Available: ${Array.from(this.storedChunks.keys()).join(', ')}`);
            return false;
        }

        if (!this.isInitialized) {
            await this.init();
        }

        if (this.audioContext.state === 'suspended') {
            await this.audioContext.resume();
        }

        // Stop any previous single-chunk playback
        if (this.singleChunkSource) {
            try {
                this.singleChunkSource.stop();
            } catch (e) {}
            this.singleChunkSource = null;
        }

        // Create a new source node for this chunk
        const source = this.audioContext.createBufferSource();
        source.buffer = chunk.buffer;

        // Connect directly to gain node (bypasses worklet)
        source.connect(this.gainNode);

        source.onended = () => {
            console.log(`[SeamlessStreamer] Single chunk ${chunkIndex} finished`);
            this.singleChunkSource = null;
        };

        this.singleChunkSource = source;
        source.start(0);

        console.log(`[SeamlessStreamer] Playing single chunk ${chunkIndex} (${chunk.buffer.duration.toFixed(3)}s, ${chunk.samples.length} samples)`);
        console.log(`[SeamlessStreamer] Chunk ${chunkIndex} boundaries: first=${chunk.samples[0]?.toFixed(6)}, last=${chunk.samples[chunk.samples.length-1]?.toFixed(6)}`);

        return true;
    }

    /**
     * Get chunk count (for compatibility with chunked streamer)
     */
    getChunkCount() {
        return this.storedChunks.size;
    }

    /**
     * Get total duration (for compatibility with chunked streamer)
     */
    getTotalDuration() {
        let total = 0;
        for (const chunk of this.storedChunks.values()) {
            total += chunk.buffer.duration;
        }
        return total;
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

    /**
     * Test the worklet by sending a known-good sine wave
     * Use this to verify the worklet is working correctly
     */
    async testWithSineWave(durationSeconds = 1.0, frequency = 440) {
        if (!this.isInitialized) {
            await this.init();
        }

        if (this.audioContext.state === 'suspended') {
            await this.audioContext.resume();
        }

        const sampleRate = this.audioContext.sampleRate;
        const numSamples = Math.floor(sampleRate * durationSeconds);
        const samples = new Array(numSamples);

        // Generate a sine wave that ends at zero (full period)
        const periodsToGenerate = Math.floor(frequency * durationSeconds);
        const actualDuration = periodsToGenerate / frequency;
        const actualSamples = Math.floor(sampleRate * actualDuration);

        for (let i = 0; i < actualSamples; i++) {
            samples[i] = 0.3 * Math.sin(2 * Math.PI * frequency * i / sampleRate);
        }

        console.log(`[SeamlessStreamer] TEST: Sending ${actualSamples} sine wave samples (${actualDuration.toFixed(3)}s, ${frequency}Hz)`);
        console.log(`[SeamlessStreamer] TEST: First sample=${samples[0].toFixed(6)}, last=${samples[actualSamples-1].toFixed(6)}`);

        this.workletNode.port.postMessage({
            type: 'samples',
            samples: samples.slice(0, actualSamples)
        });

        // Auto-start if not playing
        if (!this.isPlaying) {
            setTimeout(() => this._startPlayback(), 100);
        }

        return actualDuration;
    }

    /**
     * Test by adding raw samples directly (for debugging sample boundaries)
     */
    async testAddRawSamples(samples) {
        if (!this.isInitialized) {
            await this.init();
        }

        if (this.audioContext.state === 'suspended') {
            await this.audioContext.resume();
        }

        const samplesArray = Array.isArray(samples) ? samples : Array.from(samples);

        console.log(`[SeamlessStreamer] TEST: Adding ${samplesArray.length} raw samples`);
        console.log(`[SeamlessStreamer] First 5: [${samplesArray.slice(0,5).map(s => s.toFixed(6)).join(', ')}]`);
        console.log(`[SeamlessStreamer] Last 5: [${samplesArray.slice(-5).map(s => s.toFixed(6)).join(', ')}]`);

        this.workletNode.port.postMessage({
            type: 'samples',
            samples: samplesArray
        });

        if (!this.isPlaying) {
            setTimeout(() => this._startPlayback(), 100);
        }
    }
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SeamlessAudioStreamer;
}
