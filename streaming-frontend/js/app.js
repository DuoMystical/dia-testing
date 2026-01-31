/**
 * Dia2 Streaming TTS - Main Application
 *
 * Uses WebM/Opus streaming via MediaSource Extensions for gapless audio playback.
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const textInput = document.getElementById('text-input');
    const charCount = document.getElementById('char-count');
    const modelSelect = document.getElementById('model');
    const streamInputCheckbox = document.getElementById('stream-input');
    const generateBtn = document.getElementById('generate-btn');
    const cancelBtn = document.getElementById('cancel-btn');

    const connectionStatus = document.getElementById('connection-status');
    const statusText = connectionStatus.querySelector('.status-text');

    const progressSection = document.getElementById('progress-section');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    const chunksCount = document.getElementById('chunks-count');
    const audioDuration = document.getElementById('audio-duration');

    const visualizerCanvas = document.getElementById('visualizer');
    const visualizerCtx = visualizerCanvas.getContext('2d');
    const playPauseBtn = document.getElementById('play-pause-btn');
    const playIcon = document.getElementById('play-icon');
    const currentTimeEl = document.getElementById('current-time');
    const totalTimeEl = document.getElementById('total-time');
    const volumeSlider = document.getElementById('volume');
    const chunkList = document.getElementById('chunk-list');
    const chunkTotal = document.getElementById('chunk-total');

    // Advanced settings - Sampling
    const audioTempInput = document.getElementById('audio-temp');
    const audioTopKInput = document.getElementById('audio-top-k');
    const textTempInput = document.getElementById('text-temp');
    const textTopKInput = document.getElementById('text-top-k');

    // Advanced settings - CFG
    const cfgScaleInput = document.getElementById('cfg-scale');
    const cfgFilterKInput = document.getElementById('cfg-filter-k');

    // Advanced settings - Streaming
    const chunkSizeInput = document.getElementById('chunk-size');
    const minChunkSizeInput = document.getElementById('min-chunk-size');

    // Debug options
    const debugIncludeWarmupCheckbox = document.getElementById('debug-include-warmup');

    // Seed
    const seedInput = document.getElementById('seed-input');
    const seedDisplay = document.getElementById('seed-display');

    // Voice cloning
    const speaker1AudioInput = document.getElementById('speaker-1-audio');
    const speaker1Status = document.getElementById('speaker-1-status');
    const speaker2AudioInput = document.getElementById('speaker-2-audio');
    const speaker2Status = document.getElementById('speaker-2-status');

    // Timer elements
    const generationTimer = document.getElementById('generation-timer');
    const firstAudioTime = document.getElementById('first-audio-time');
    const chunkLatency = document.getElementById('chunk-latency');
    const chunkTimingLog = document.getElementById('chunk-timing-log');
    const timingEntries = document.getElementById('timing-entries');

    // State
    let isGenerating = false;
    let wsClient = null;
    let audioStreamer = null; // WebM streamer (primary) or legacy fallback

    // Timer state
    let generationStartTime = null;
    let timerInterval = null;
    let firstChunkTime = null;

    // Voice cloning state (base64 encoded audio)
    let speaker1AudioData = null;
    let speaker2AudioData = null;

    // Chunk timing state
    let lastChunkReceiveTime = null;
    let chunkTimings = [];

    // Initialize WebSocket client
    function initWebSocket() {
        wsClient = new WebSocketClient({
            onConnect: () => {
                connectionStatus.classList.remove('disconnected');
                connectionStatus.classList.add('connected');
                statusText.textContent = 'Connected';
                generateBtn.disabled = false;
            },
            onDisconnect: () => {
                connectionStatus.classList.remove('connected');
                connectionStatus.classList.add('disconnected');
                statusText.textContent = 'Disconnected';
                generateBtn.disabled = true;
                setGenerating(false);
            },
            onAudioChunk: handleAudioChunk,
            onStatus: handleStatus,
            onComplete: handleComplete,
            onError: handleError,
            onSeed: handleSeed
        });

        wsClient.connect();

        // Expose for debugging
        window.wsClient = wsClient;
        window.audioStreamer = audioStreamer;
    }

    // Handle seed received from backend
    function handleSeed(seed) {
        seedDisplay.textContent = `Seed: ${seed}`;
        seedDisplay.hidden = false;
        seedDisplay.style.cursor = 'pointer';
        seedDisplay.onclick = () => {
            navigator.clipboard.writeText(seed.toString()).then(() => {
                const original = seedDisplay.textContent;
                seedDisplay.textContent = 'Copied!';
                setTimeout(() => {
                    seedDisplay.textContent = original;
                }, 1000);
            });
        };
    }

    // Initialize Audio Streamer
    function initAudioStreamer() {
        // Use WebM/MSE streaming (gapless playback)
        if (typeof WebMAudioStreamer !== 'undefined') {
            audioStreamer = new WebMAudioStreamer({
                volume: volumeSlider.value / 100,
                onPlaybackStart: () => {
                    playIcon.textContent = '\u23F8'; // Pause icon
                },
                onPlaybackEnd: () => {
                    playIcon.textContent = '\u25B6'; // Play icon
                },
                onVisualizerData: (data) => {
                    drawVisualizer(data);
                },
                onBufferUpdate: (bytes, chunks) => {
                    // Update time display
                    if (audioStreamer) {
                        const current = audioStreamer.audioElement?.currentTime || 0;
                        const total = audioStreamer.getTotalDuration();
                        currentTimeEl.textContent = formatTime(current);
                        totalTimeEl.textContent = formatTime(total);
                    }
                },
                onError: (error) => {
                    console.error('[WebMStreamer] Error:', error);
                }
            });
            console.log('[App] WebM streaming enabled (MSE-based gapless playback)');
        } else {
            // Fallback to legacy chunk-based streamer
            audioStreamer = new AudioStreamer({
                volume: volumeSlider.value / 100,
                autoPlay: true,
                onChunkPlay: (index) => {
                    updateChunkHighlight(index);
                },
                onPlaybackEnd: () => {
                    playIcon.textContent = '\u25B6'; // Play icon
                },
                onTimeUpdate: (current, total) => {
                    currentTimeEl.textContent = formatTime(current);
                    totalTimeEl.textContent = formatTime(total);
                },
                onVisualizerData: (data) => {
                    drawVisualizer(data);
                }
            });
            console.warn('[App] WebMAudioStreamer not available, using legacy chunked playback');
        }

        window.audioStreamer = audioStreamer;
    }

    // Timer functions
    function startTimer() {
        generationStartTime = performance.now();
        firstChunkTime = null;
        generationTimer.textContent = '0.0s';
        firstAudioTime.hidden = true;

        timerInterval = setInterval(() => {
            if (generationStartTime) {
                const elapsed = (performance.now() - generationStartTime) / 1000;
                generationTimer.textContent = `${elapsed.toFixed(1)}s`;
            }
        }, 100);
    }

    function stopTimer() {
        if (timerInterval) {
            clearInterval(timerInterval);
            timerInterval = null;
        }
        if (generationStartTime) {
            const elapsed = (performance.now() - generationStartTime) / 1000;
            generationTimer.textContent = `${elapsed.toFixed(1)}s`;
        }
    }

    function recordFirstChunk() {
        if (!firstChunkTime && generationStartTime) {
            firstChunkTime = performance.now();
            const timeToFirst = (firstChunkTime - generationStartTime) / 1000;
            firstAudioTime.textContent = `First audio: ${timeToFirst.toFixed(1)}s`;
            firstAudioTime.hidden = false;
        }
    }

    // Handle audio chunk received
    async function handleAudioChunk(chunk) {
        const receiveTime = performance.now();

        // Record time to first chunk
        recordFirstChunk();

        // Calculate inter-chunk latency
        let interChunkLatency = 0;
        if (lastChunkReceiveTime !== null) {
            interChunkLatency = receiveTime - lastChunkReceiveTime;
        }
        lastChunkReceiveTime = receiveTime;

        // Add chunk to streamer
        await audioStreamer.addChunk(chunk.data, chunk.chunkIndex);

        // Use actual duration from server (in ms)
        const chunkAudioMs = chunk.duration_ms || 0;

        // Track timing
        const timingData = {
            index: chunk.chunkIndex,
            receiveTime: receiveTime - generationStartTime,
            interChunkLatency: interChunkLatency,
            audioDurationMs: chunkAudioMs,
            bytesReceived: chunk.data.byteLength
        };
        chunkTimings.push(timingData);

        // Add timing entry to log
        addTimingEntry(timingData);

        // Update latency display
        if (interChunkLatency > 0) {
            chunkLatency.textContent = `Chunk gap: ${interChunkLatency.toFixed(0)}ms`;
            chunkLatency.hidden = false;
        }

        // Update UI
        chunksCount.textContent = `${audioStreamer.getChunkCount()} chunks`;
        audioDuration.textContent = `${audioStreamer.getTotalDuration().toFixed(1)}s audio`;

        // Add chunk to list
        addChunkToList(chunk.chunkIndex, chunkAudioMs);

        playPauseBtn.disabled = false;
    }

    // Add timing entry to the log
    function addTimingEntry(timing) {
        chunkTimingLog.hidden = false;
        const entry = document.createElement('div');
        entry.className = 'timing-entry';
        entry.innerHTML = `
            <span class="chunk-num">#${timing.index}</span>
            @ ${(timing.receiveTime/1000).toFixed(2)}s |
            <span class="duration">${timing.audioDurationMs.toFixed(0)}ms audio</span> |
            <span class="latency">gap: ${timing.interChunkLatency.toFixed(0)}ms</span> |
            ${(timing.bytesReceived/1024).toFixed(1)}KB
        `;
        timingEntries.appendChild(entry);
        timingEntries.scrollTop = timingEntries.scrollHeight;
    }

    // Handle status update
    function handleStatus(status) {
        progressText.textContent = status.message;
        progressBar.style.width = `${(status.progress || 0) * 100}%`;
    }

    // Handle generation complete
    function handleComplete(result) {
        stopTimer();
        const genTime = generationStartTime ? ((performance.now() - generationStartTime) / 1000).toFixed(1) : '?';
        progressText.textContent = `Complete! ${result.totalChunks} chunks, ${(result.totalDurationMs / 1000).toFixed(1)}s audio in ${genTime}s`;
        progressBar.style.width = '100%';

        // Signal end of stream (WebM streamer needs this)
        if (audioStreamer.endOfStream) {
            audioStreamer.endOfStream();
        }

        // Keep progress section visible
        setGenerating(false, true);
    }

    // Handle errors
    function handleError(error) {
        stopTimer();
        console.error('Error:', error);
        progressText.textContent = `Error: ${error.message}`;
        setGenerating(false);
    }

    // Set generating state
    function setGenerating(generating, keepProgressVisible = false) {
        isGenerating = generating;
        generateBtn.disabled = generating || !wsClient?.isConnected;
        cancelBtn.disabled = !generating;
        if (!keepProgressVisible) {
            progressSection.hidden = !generating;
        }
    }

    // Add chunk indicator to list
    function addChunkToList(index, durationMs) {
        const item = document.createElement('div');
        item.className = 'chunk-item new';
        item.textContent = index + 1;
        item.title = `Chunk ${index + 1}: ${durationMs.toFixed(0)}ms`;
        item.dataset.index = index;

        chunkList.appendChild(item);
        chunkTotal.textContent = `(${audioStreamer.getChunkCount()})`;

        // Remove 'new' animation class after animation
        setTimeout(() => {
            item.classList.remove('new');
        }, 300);

        // Auto-scroll to show new chunk
        chunkList.scrollTop = chunkList.scrollHeight;
    }

    // Update chunk highlight during playback
    function updateChunkHighlight(currentIndex) {
        const items = chunkList.querySelectorAll('.chunk-item');
        items.forEach((item, i) => {
            item.classList.toggle('playing', i === currentIndex);
        });
    }

    // Draw audio visualizer
    function drawVisualizer(data) {
        const width = visualizerCanvas.width;
        const height = visualizerCanvas.height;
        const barCount = data.length;
        const barWidth = width / barCount;

        visualizerCtx.fillStyle = '#0f172a';
        visualizerCtx.fillRect(0, 0, width, height);

        for (let i = 0; i < barCount; i++) {
            const value = data[i] / 255;
            const barHeight = value * height;

            const hue = 240 + value * 120;
            visualizerCtx.fillStyle = `hsl(${hue}, 70%, ${50 + value * 30}%)`;

            visualizerCtx.fillRect(
                i * barWidth,
                height - barHeight,
                barWidth - 1,
                barHeight
            );
        }
    }

    // Draw idle visualizer
    function drawIdleVisualizer() {
        const width = visualizerCanvas.width;
        const height = visualizerCanvas.height;

        visualizerCtx.fillStyle = '#0f172a';
        visualizerCtx.fillRect(0, 0, width, height);

        visualizerCtx.strokeStyle = '#334155';
        visualizerCtx.lineWidth = 2;
        visualizerCtx.beginPath();
        visualizerCtx.moveTo(0, height / 2);
        visualizerCtx.lineTo(width, height / 2);
        visualizerCtx.stroke();
    }

    // Format time as M:SS
    function formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    // Generate button click
    function handleGenerate() {
        const text = textInput.value.trim();
        if (!text) {
            alert('Please enter some text');
            return;
        }

        // Reset audio streamer
        audioStreamer.reset();
        chunkList.innerHTML = '';

        // Reset timing state
        lastChunkReceiveTime = null;
        chunkTimings = [];
        timingEntries.innerHTML = '';
        chunkTimingLog.hidden = true;
        chunkLatency.hidden = true;
        seedDisplay.hidden = true;

        // Reset progress
        progressBar.style.width = '0%';
        progressText.textContent = 'Starting...';
        chunksCount.textContent = '0 chunks';
        audioDuration.textContent = '0.0s audio';

        // Start the generation timer
        startTimer();

        setGenerating(true);

        const options = {
            model: modelSelect.value,
            audioTemperature: parseFloat(audioTempInput.value) || 0.8,
            audioTopK: parseInt(audioTopKInput.value) || 50,
            textTemperature: parseFloat(textTempInput.value) || 0.6,
            textTopK: parseInt(textTopKInput.value) || 50,
            cfgScale: parseFloat(cfgScaleInput.value) || 2.0,
            cfgFilterK: parseInt(cfgFilterKInput.value) || 50,
            chunkSizeFrames: parseInt(chunkSizeInput.value) || 1,
            minChunkFrames: parseInt(minChunkSizeInput.value) || 1,
            seed: seedInput.value.trim(),
            speaker1Audio: speaker1AudioData,
            speaker2Audio: speaker2AudioData,
            debugIncludeWarmup: debugIncludeWarmupCheckbox.checked
        };

        if (streamInputCheckbox.checked) {
            wsClient.sendTextChunks(text, 50, options);
        } else {
            wsClient.generate(text, options);
        }
    }

    // Cancel button click
    function handleCancel() {
        stopTimer();
        wsClient.cancel();
        setGenerating(false);
    }

    // Play/Pause button click
    async function handlePlayPause() {
        try {
            if (audioStreamer.isPlaying) {
                audioStreamer.pause();
                playIcon.textContent = '\u25B6';
            } else {
                await audioStreamer.play();
                playIcon.textContent = '\u23F8';
            }
        } catch (error) {
            console.error('Playback error:', error);
        }
    }

    // Volume change
    function handleVolumeChange() {
        audioStreamer.setVolume(volumeSlider.value / 100);
    }

    // Character count update
    function updateCharCount() {
        charCount.textContent = textInput.value.length;
    }

    // Voice cloning file upload handlers
    function handleSpeaker1Upload(event) {
        const file = event.target.files[0];
        if (!file) {
            speaker1AudioData = null;
            speaker1Status.textContent = '';
            speaker1Status.className = 'upload-status';
            return;
        }

        if (!file.type.includes('wav') && !file.name.endsWith('.wav')) {
            speaker1Status.textContent = 'Error: Only WAV files are supported';
            speaker1Status.className = 'upload-status error';
            speaker1AudioData = null;
            return;
        }

        const reader = new FileReader();
        reader.onload = function(e) {
            const base64 = e.target.result.split(',')[1];
            speaker1AudioData = base64;
            speaker1Status.textContent = `Loaded: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
            speaker1Status.className = 'upload-status loaded';
        };
        reader.onerror = function() {
            speaker1Status.textContent = 'Error reading file';
            speaker1Status.className = 'upload-status error';
            speaker1AudioData = null;
        };
        reader.readAsDataURL(file);
    }

    function handleSpeaker2Upload(event) {
        const file = event.target.files[0];
        if (!file) {
            speaker2AudioData = null;
            speaker2Status.textContent = '';
            speaker2Status.className = 'upload-status';
            return;
        }

        if (!file.type.includes('wav') && !file.name.endsWith('.wav')) {
            speaker2Status.textContent = 'Error: Only WAV files are supported';
            speaker2Status.className = 'upload-status error';
            speaker2AudioData = null;
            return;
        }

        const reader = new FileReader();
        reader.onload = function(e) {
            const base64 = e.target.result.split(',')[1];
            speaker2AudioData = base64;
            speaker2Status.textContent = `Loaded: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
            speaker2Status.className = 'upload-status loaded';
        };
        reader.onerror = function() {
            speaker2Status.textContent = 'Error reading file';
            speaker2Status.className = 'upload-status error';
            speaker2AudioData = null;
        };
        reader.readAsDataURL(file);
    }

    // Event Listeners
    generateBtn.addEventListener('click', handleGenerate);
    cancelBtn.addEventListener('click', handleCancel);
    playPauseBtn.addEventListener('click', handlePlayPause);
    volumeSlider.addEventListener('input', handleVolumeChange);
    textInput.addEventListener('input', updateCharCount);
    speaker1AudioInput.addEventListener('change', handleSpeaker1Upload);
    speaker2AudioInput.addEventListener('change', handleSpeaker2Upload);

    // Initialize
    initWebSocket();
    initAudioStreamer();
    drawIdleVisualizer();
    updateCharCount();

    // Set default text
    if (!textInput.value) {
        textInput.value = '[S1] Hello! This is a streaming TTS demo. [S2] It generates audio in real-time as the model processes the text.';
        updateCharCount();
    }

    // Resize visualizer canvas to match container
    function resizeVisualizer() {
        const container = visualizerCanvas.parentElement;
        visualizerCanvas.width = container.clientWidth;
        drawIdleVisualizer();
    }

    window.addEventListener('resize', resizeVisualizer);
    resizeVisualizer();
});
