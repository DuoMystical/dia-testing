<script lang="ts">
	let textInput = $state('');
	let selectedModel = $state<'1b' | '2b'>('2b');

	// Audio sampling (affects voice quality/variation)
	let audioTemperature = $state(0.8);
	let audioTopK = $state(50);

	// Text sampling (affects pronunciation/pacing)
	let textTemperature = $state(0.6);
	let textTopK = $state(50);

	// Generation config
	let cfgScale = $state(2.0);
	let cfgFilterK = $state(50);
	let useCudaGraph = $state(true);
	let useTorchCompile = $state(false);

	let isGenerating = $state(false);
	let audioUrl = $state<string | null>(null);
	let error = $state<string | null>(null);
	let statusMessage = $state<string | null>(null);
	let generationTime = $state<number | null>(null);
	let modelUsed = $state<string | null>(null);

	// Backend URL - Dia2 backend with dual model support
	const BACKEND_URL = 'https://occasional-angel-puddle-6a36e6b8.koyeb.app';

	async function generateAudio() {
		if (!textInput.trim()) {
			error = 'Please enter some text';
			return;
		}

		isGenerating = true;
		error = null;
		audioUrl = null;
		generationTime = null;
		modelUsed = null;
		statusMessage = `Connecting to GPU server with Dia2-${selectedModel.toUpperCase()}...`;

		// Use AbortController with 5 minute timeout for slow GPU processing
		const controller = new AbortController();
		const timeoutId = setTimeout(() => controller.abort(), 300000); // 5 minutes

		try {
			const response = await fetch(`${BACKEND_URL}/api/generate`, {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify({
					text_input: textInput,
					model: selectedModel,
					audio_temperature: audioTemperature,
					audio_top_k: audioTopK,
					text_temperature: textTemperature,
					text_top_k: textTopK,
					cfg_scale: cfgScale,
					cfg_filter_k: cfgFilterK,
					use_cuda_graph: useCudaGraph,
					use_torch_compile: useTorchCompile
				}),
				signal: controller.signal
			});

			clearTimeout(timeoutId);
			statusMessage = 'Processing audio...';

			if (!response.ok) {
				let errorMsg = 'Failed to generate audio';
				try {
					const errorData = await response.json();
					errorMsg = errorData.detail || errorMsg;
				} catch {
					if (response.status === 503) {
						errorMsg = 'Server is starting up. Please wait 30-60 seconds and try again.';
					}
				}
				throw new Error(errorMsg);
			}

			// Extract headers for generation info
			const genTime = response.headers.get('X-Generation-Time');
			const usedModel = response.headers.get('X-Model-Used');
			if (genTime) generationTime = parseFloat(genTime);
			if (usedModel) modelUsed = usedModel;

			const blob = await response.blob();
			if (blob.size < 100) {
				throw new Error('Generated audio is empty. Try again.');
			}
			audioUrl = URL.createObjectURL(blob);
			statusMessage = null;
		} catch (e) {
			if (e instanceof Error && e.name === 'AbortError') {
				error = 'Request timed out. The GPU may be busy. Please try again.';
			} else {
				error = e instanceof Error ? e.message : 'An error occurred';
			}
			statusMessage = null;
		} finally {
			isGenerating = false;
			clearTimeout(timeoutId);
		}
	}

	function downloadAudio() {
		if (audioUrl) {
			const a = document.createElement('a');
			a.href = audioUrl;
			a.download = `dia2-${selectedModel}-${Date.now()}.wav`;
			a.click();
		}
	}
</script>

<svelte:head>
	<title>Dia2 Text-to-Speech</title>
</svelte:head>

<main>
	<h1>Dia2 Text-to-Speech</h1>
	<p class="subtitle">Generate realistic streaming dialogue with Dia2 models</p>

	<div class="container">
		<!-- Model Selector -->
		<div class="model-selector">
			<label><strong>Model</strong></label>
			<div class="model-options">
				<button
					class="model-btn"
					class:active={selectedModel === '1b'}
					onclick={() => selectedModel = '1b'}
				>
					<span class="model-name">Dia2-1B</span>
					<span class="model-desc">Faster, lighter</span>
				</button>
				<button
					class="model-btn"
					class:active={selectedModel === '2b'}
					onclick={() => selectedModel = '2b'}
				>
					<span class="model-name">Dia2-2B</span>
					<span class="model-desc">Better quality</span>
				</button>
			</div>
		</div>

		<div class="input-section">
			<label for="text-input">
				<strong>Text Input</strong>
				<span class="hint">Use [S1] and [S2] for different speakers</span>
			</label>
			<textarea
				id="text-input"
				bind:value={textInput}
				placeholder="[S1] Hello, how are you today? [S2] I'm doing great, thanks for asking!"
				rows="6"
			></textarea>
		</div>

		<details class="settings">
			<summary>Generation Settings</summary>
			<div class="settings-sections">
				<!-- Audio Sampling -->
				<div class="settings-section">
					<h4>Audio Sampling</h4>
					<p class="section-hint">Affects voice quality and variation</p>
					<div class="settings-grid">
						<div class="setting">
							<label for="audio-temp">Temperature: {audioTemperature.toFixed(2)}</label>
							<input type="range" id="audio-temp" bind:value={audioTemperature} min="0.1" max="1.5" step="0.05" />
						</div>
						<div class="setting">
							<label for="audio-topk">Top K: {audioTopK}</label>
							<input type="range" id="audio-topk" bind:value={audioTopK} min="10" max="100" step="5" />
						</div>
					</div>
				</div>

				<!-- Text Sampling -->
				<div class="settings-section">
					<h4>Text Sampling</h4>
					<p class="section-hint">Affects pronunciation and pacing</p>
					<div class="settings-grid">
						<div class="setting">
							<label for="text-temp">Temperature: {textTemperature.toFixed(2)}</label>
							<input type="range" id="text-temp" bind:value={textTemperature} min="0.1" max="1.5" step="0.05" />
						</div>
						<div class="setting">
							<label for="text-topk">Top K: {textTopK}</label>
							<input type="range" id="text-topk" bind:value={textTopK} min="10" max="100" step="5" />
						</div>
					</div>
				</div>

				<!-- Generation Config -->
				<div class="settings-section">
					<h4>Generation Config</h4>
					<p class="section-hint">Classifier-free guidance settings</p>
					<div class="settings-grid">
						<div class="setting">
							<label for="cfg-scale">CFG Scale: {cfgScale.toFixed(1)}</label>
							<input type="range" id="cfg-scale" bind:value={cfgScale} min="1" max="10" step="0.5" />
						</div>
						<div class="setting">
							<label for="cfg-filter-k">CFG Filter K: {cfgFilterK}</label>
							<input type="range" id="cfg-filter-k" bind:value={cfgFilterK} min="10" max="100" step="5" />
						</div>
					</div>
				</div>

				<!-- Optimizations -->
				<div class="settings-section">
					<h4>Optimizations</h4>
					<div class="settings-grid">
						<div class="setting checkbox-setting">
							<label>
								<input type="checkbox" bind:checked={useCudaGraph} />
								CUDA Graph
								<span class="opt-hint">Faster inference</span>
							</label>
						</div>
						<div class="setting checkbox-setting">
							<label>
								<input type="checkbox" bind:checked={useTorchCompile} />
								torch.compile
								<span class="opt-hint">Slower first run, faster after</span>
							</label>
						</div>
					</div>
				</div>
			</div>
		</details>

		<button onclick={generateAudio} disabled={isGenerating}>
			{isGenerating ? 'Generating...' : `Generate with Dia2-${selectedModel.toUpperCase()}`}
		</button>

		{#if statusMessage}
			<div class="status">{statusMessage}</div>
		{/if}

		{#if error}
			<div class="error">{error}</div>
		{/if}

		{#if audioUrl}
			<div class="audio-output">
				<h3>Generated Audio</h3>
				{#if generationTime && modelUsed}
					<p class="generation-info">
						Generated in {generationTime.toFixed(2)}s using {modelUsed}
					</p>
				{/if}
				<audio controls src={audioUrl}></audio>
				<button onclick={downloadAudio} class="download-btn">Download Audio</button>
			</div>
		{/if}
	</div>
</main>

<style>
	:global(body) {
		font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
		background: #0a0a0a;
		color: #e0e0e0;
		margin: 0;
		padding: 20px;
	}

	main {
		max-width: 800px;
		margin: 0 auto;
	}

	h1 {
		text-align: center;
		color: #fff;
		margin-bottom: 0.5rem;
	}

	.subtitle {
		text-align: center;
		color: #888;
		margin-bottom: 2rem;
	}

	.container {
		background: #1a1a1a;
		border-radius: 12px;
		padding: 24px;
		box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
	}

	/* Model Selector Styles */
	.model-selector {
		margin-bottom: 1.5rem;
	}

	.model-selector > label {
		display: block;
		margin-bottom: 0.5rem;
		color: #ccc;
	}

	.model-options {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 12px;
	}

	.model-btn {
		padding: 16px;
		background: #0a0a0a;
		border: 2px solid #333;
		border-radius: 8px;
		cursor: pointer;
		transition: all 0.2s;
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 4px;
	}

	.model-btn:hover {
		border-color: #555;
		background: #151515;
	}

	.model-btn.active {
		border-color: #4a9eff;
		background: #0a1a2a;
	}

	.model-name {
		font-size: 1.1rem;
		font-weight: 600;
		color: #fff;
	}

	.model-desc {
		font-size: 0.85rem;
		color: #888;
	}

	.model-btn.active .model-name {
		color: #4a9eff;
	}

	.input-section {
		margin-bottom: 1.5rem;
	}

	label {
		display: block;
		margin-bottom: 0.5rem;
		color: #ccc;
	}

	.hint {
		font-size: 0.85rem;
		color: #666;
		font-weight: normal;
		margin-left: 0.5rem;
	}

	textarea {
		width: 100%;
		padding: 12px;
		border: 1px solid #333;
		border-radius: 8px;
		background: #0a0a0a;
		color: #e0e0e0;
		font-size: 1rem;
		resize: vertical;
		box-sizing: border-box;
	}

	textarea:focus {
		outline: none;
		border-color: #4a9eff;
	}

	.settings {
		margin-bottom: 1.5rem;
		border: 1px solid #333;
		border-radius: 8px;
		padding: 12px;
	}

	.settings summary {
		cursor: pointer;
		color: #888;
		font-weight: 500;
	}

	.settings-sections {
		margin-top: 1rem;
		display: flex;
		flex-direction: column;
		gap: 1.5rem;
	}

	.settings-section {
		padding-bottom: 1rem;
		border-bottom: 1px solid #2a2a2a;
	}

	.settings-section:last-child {
		border-bottom: none;
		padding-bottom: 0;
	}

	.settings-section h4 {
		margin: 0 0 0.25rem 0;
		color: #ccc;
		font-size: 0.95rem;
	}

	.section-hint {
		margin: 0 0 0.75rem 0;
		font-size: 0.8rem;
		color: #666;
	}

	.settings-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
		gap: 1rem;
	}

	.setting label {
		font-size: 0.9rem;
		margin-bottom: 0.25rem;
	}

	.setting input[type="range"] {
		width: 100%;
		accent-color: #4a9eff;
	}

	.checkbox-setting label {
		display: flex;
		align-items: center;
		gap: 8px;
		cursor: pointer;
		flex-wrap: wrap;
	}

	.checkbox-setting input[type="checkbox"] {
		width: 18px;
		height: 18px;
		accent-color: #4a9eff;
	}

	.opt-hint {
		font-size: 0.75rem;
		color: #666;
		margin-left: auto;
	}

	button {
		width: 100%;
		padding: 14px;
		background: linear-gradient(135deg, #4a9eff, #0066cc);
		color: white;
		border: none;
		border-radius: 8px;
		font-size: 1.1rem;
		font-weight: 600;
		cursor: pointer;
		transition: transform 0.1s, opacity 0.2s;
	}

	button:hover:not(:disabled) {
		transform: translateY(-1px);
	}

	button:disabled {
		opacity: 0.6;
		cursor: not-allowed;
	}

	.status {
		margin-top: 1rem;
		padding: 12px;
		background: #1a2a1a;
		border: 1px solid #2a4a2a;
		border-radius: 8px;
		color: #88cc88;
		text-align: center;
		animation: pulse 2s infinite;
	}

	@keyframes pulse {
		0%, 100% { opacity: 1; }
		50% { opacity: 0.6; }
	}

	.error {
		margin-top: 1rem;
		padding: 12px;
		background: #331111;
		border: 1px solid #662222;
		border-radius: 8px;
		color: #ff6666;
	}

	.audio-output {
		margin-top: 1.5rem;
		padding: 16px;
		background: #0a0a0a;
		border-radius: 8px;
		text-align: center;
	}

	.audio-output h3 {
		margin: 0 0 0.5rem 0;
		color: #fff;
	}

	.generation-info {
		margin: 0 0 1rem 0;
		font-size: 0.9rem;
		color: #88cc88;
	}

	audio {
		width: 100%;
		margin-bottom: 1rem;
	}

	.download-btn {
		width: auto;
		padding: 10px 20px;
		font-size: 0.9rem;
		background: #333;
	}

	.download-btn:hover {
		background: #444;
	}
</style>
