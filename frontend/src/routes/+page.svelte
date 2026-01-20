<script lang="ts">
	let textInput = $state('');
	let maxNewTokens = $state(1024);
	let cfgScale = $state(3.0);
	let temperature = $state(1.3);
	let topP = $state(0.95);
	let cfgFilterTopK = $state(35);
	let speedFactor = $state(0.94);

	let isGenerating = $state(false);
	let audioUrl = $state<string | null>(null);
	let error = $state<string | null>(null);

	// Backend URL - premade Koyeb Dia backend
	const BACKEND_URL = 'https://distinct-rosalie-covora-2ccf0de5.koyeb.app';

	async function generateAudio() {
		if (!textInput.trim()) {
			error = 'Please enter some text';
			return;
		}

		isGenerating = true;
		error = null;
		audioUrl = null;

		try {
			const response = await fetch(`${BACKEND_URL}/api/generate`, {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify({
					text_input: textInput,
					max_new_tokens: maxNewTokens,
					cfg_scale: cfgScale,
					temperature: temperature,
					top_p: topP,
					cfg_filter_top_k: cfgFilterTopK,
					speed_factor: speedFactor
				})
			});

			if (!response.ok) {
				const errorData = await response.json();
				throw new Error(errorData.detail || 'Failed to generate audio');
			}

			const blob = await response.blob();
			audioUrl = URL.createObjectURL(blob);
		} catch (e) {
			error = e instanceof Error ? e.message : 'An error occurred';
		} finally {
			isGenerating = false;
		}
	}

	function downloadAudio() {
		if (audioUrl) {
			const a = document.createElement('a');
			a.href = audioUrl;
			a.download = `dia-audio-${Date.now()}.wav`;
			a.click();
		}
	}
</script>

<svelte:head>
	<title>Dia Text-to-Speech</title>
</svelte:head>

<main>
	<h1>Dia 1.6B Text-to-Speech</h1>
	<p class="subtitle">Generate realistic dialogue audio using the Dia model</p>

	<div class="container">
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
			<div class="settings-grid">
				<div class="setting">
					<label for="max-tokens">Max New Tokens: {maxNewTokens}</label>
					<input type="range" id="max-tokens" bind:value={maxNewTokens} min="256" max="4096" step="64" />
				</div>
				<div class="setting">
					<label for="cfg-scale">CFG Scale: {cfgScale.toFixed(1)}</label>
					<input type="range" id="cfg-scale" bind:value={cfgScale} min="1" max="5" step="0.1" />
				</div>
				<div class="setting">
					<label for="temperature">Temperature: {temperature.toFixed(2)}</label>
					<input type="range" id="temperature" bind:value={temperature} min="0.5" max="2" step="0.05" />
				</div>
				<div class="setting">
					<label for="top-p">Top P: {topP.toFixed(2)}</label>
					<input type="range" id="top-p" bind:value={topP} min="0.5" max="1" step="0.01" />
				</div>
				<div class="setting">
					<label for="top-k">Filter Top K: {cfgFilterTopK}</label>
					<input type="range" id="top-k" bind:value={cfgFilterTopK} min="10" max="100" step="5" />
				</div>
				<div class="setting">
					<label for="speed">Speed Factor: {speedFactor.toFixed(2)}</label>
					<input type="range" id="speed" bind:value={speedFactor} min="0.5" max="1.5" step="0.01" />
				</div>
			</div>
		</details>

		<button onclick={generateAudio} disabled={isGenerating}>
			{isGenerating ? 'Generating...' : 'Generate Audio'}
		</button>

		{#if error}
			<div class="error">{error}</div>
		{/if}

		{#if audioUrl}
			<div class="audio-output">
				<h3>Generated Audio</h3>
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

	.settings-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
		gap: 1rem;
		margin-top: 1rem;
	}

	.setting label {
		font-size: 0.9rem;
		margin-bottom: 0.25rem;
	}

	.setting input[type="range"] {
		width: 100%;
		accent-color: #4a9eff;
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
		margin: 0 0 1rem 0;
		color: #fff;
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
