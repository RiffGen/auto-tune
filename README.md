# Auto-Tune Model

This repository contains an auto-tune model for pitch correction using Python, Cog, and Replicate.

## Quickstart

### 1. Install Cog

If you haven't already, install [Cog](https://github.com/replicate/cog):

```bash
pip install cog
```

### 2. Log in to Replicate

Authenticate your local Cog installation with Replicate:

```bash
cog login
```

Follow the instructions to paste your Replicate token.

### 3. Push the Model to Replicate

Push your model to Replicate (replace `riffgen/auto-tune` with your own repo if needed):

```bash
cog push r8.im/riffgen/auto-tune
```

### 4. Run the Model Locally

You can test the model locally by running:

```bash
python predict.py
```

Edit the `__main__` block in `predict.py` to change the input parameters, for example:

```python
if __name__ == "__main__":
    p = Predictor()
    p.setup()
    out = p.predict(
        audio_file="https://your-audio-url.m4a",
        autotune_style="balanced",
        scale="D:maj",
        correction_strength=0.8,
        adaptive_strength=True,
        smooth_transitions=True,
        plot=False,
        output_format="wav",
    )
    print("Output file:", out)
```

### 5. Run the Model on Replicate

After pushing, you can run the model on Replicate via the web UI or API. The UI will let you:

- Upload or link to an audio file
- Choose a musical scale (key)
- Choose an auto-tune style (intensity)
- Adjust correction strength, adaptive, and smoothing
- Download the processed audio

For API usage, see the [Replicate Python client docs](https://replicate.com/docs/reference/python/).

---

**Questions?**

- See [Cog documentation](https://github.com/replicate/cog)
- See [Replicate documentation](https://replicate.com/docs)
- Or open an issue in this repo!
