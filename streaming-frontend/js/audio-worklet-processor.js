/**
 * AudioWorklet Processor for seamless audio streaming
 *
 * This processor maintains a circular buffer of samples and outputs them
 * continuously, eliminating gaps between chunks that cause pops.
 */

class StreamingAudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();

        // Circular buffer for samples (5 seconds at 48kHz)
        this.bufferSize = 48000 * 5;
        this.buffer = new Float32Array(this.bufferSize);
        this.writePos = 0;
        this.readPos = 0;
        this.samplesAvailable = 0;

        // State
        this.isPlaying = false;
        this.underrunCount = 0;

        // Handle messages from main thread
        this.port.onmessage = (event) => {
            const { type, samples, command } = event.data;

            if (type === 'samples') {
                this._addSamples(samples);
            } else if (type === 'command') {
                if (command === 'start') {
                    this.isPlaying = true;
                } else if (command === 'stop') {
                    this.isPlaying = false;
                } else if (command === 'reset') {
                    this._reset();
                }
            } else if (type === 'getStatus') {
                this.port.postMessage({
                    type: 'status',
                    samplesAvailable: this.samplesAvailable,
                    bufferSize: this.bufferSize,
                    isPlaying: this.isPlaying,
                    underrunCount: this.underrunCount
                });
            }
        };
    }

    _addSamples(newSamples) {
        const samplesToAdd = newSamples.length;

        // Check for buffer overflow
        if (this.samplesAvailable + samplesToAdd > this.bufferSize) {
            // Buffer overflow - drop oldest samples
            const overflow = (this.samplesAvailable + samplesToAdd) - this.bufferSize;
            this.readPos = (this.readPos + overflow) % this.bufferSize;
            this.samplesAvailable -= overflow;
            console.warn(`[AudioWorklet] Buffer overflow, dropped ${overflow} samples`);
        }

        // Copy samples into circular buffer
        for (let i = 0; i < samplesToAdd; i++) {
            this.buffer[this.writePos] = newSamples[i];
            this.writePos = (this.writePos + 1) % this.bufferSize;
        }
        this.samplesAvailable += samplesToAdd;

        // Notify main thread of buffer level
        this.port.postMessage({
            type: 'bufferLevel',
            samplesAvailable: this.samplesAvailable,
            bufferSeconds: this.samplesAvailable / sampleRate
        });
    }

    _reset() {
        this.writePos = 0;
        this.readPos = 0;
        this.samplesAvailable = 0;
        this.isPlaying = false;
        this.underrunCount = 0;
    }

    process(inputs, outputs, parameters) {
        const output = outputs[0];
        const channel = output[0];

        if (!channel) return true;

        const framesToProcess = channel.length; // Usually 128 samples

        if (!this.isPlaying) {
            // Output silence when not playing
            channel.fill(0);
            return true;
        }

        if (this.samplesAvailable >= framesToProcess) {
            // Normal playback - read from buffer
            for (let i = 0; i < framesToProcess; i++) {
                channel[i] = this.buffer[this.readPos];
                this.readPos = (this.readPos + 1) % this.bufferSize;
            }
            this.samplesAvailable -= framesToProcess;
        } else if (this.samplesAvailable > 0) {
            // Partial buffer - play what we have, then silence
            let i = 0;
            while (this.samplesAvailable > 0 && i < framesToProcess) {
                channel[i] = this.buffer[this.readPos];
                this.readPos = (this.readPos + 1) % this.bufferSize;
                this.samplesAvailable--;
                i++;
            }
            // Fill rest with silence (fade to zero to avoid pop)
            const lastSample = i > 0 ? channel[i - 1] : 0;
            const fadeLength = Math.min(framesToProcess - i, 64);
            for (let j = 0; j < fadeLength && i < framesToProcess; j++, i++) {
                channel[i] = lastSample * (1 - j / fadeLength);
            }
            while (i < framesToProcess) {
                channel[i++] = 0;
            }
            this.underrunCount++;
        } else {
            // Buffer underrun - output silence
            channel.fill(0);
            this.underrunCount++;
        }

        return true; // Keep processor alive
    }
}

registerProcessor('streaming-audio-processor', StreamingAudioProcessor);
