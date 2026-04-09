# TRIBE v2 First Try

## Run

```bash
cd /Users/macos/Documents/Filter/test-meta-ai
./.venv/bin/python first_try.py
```

## What this does

- Loads pretrained `facebook/tribev2`.
- Creates a short synthetic audio clip (`first_try_input.wav`).
- Runs CPU inference and saves predictions to `first_try_preds.npy`.

## Expected success signal

Terminal prints:

- `OK: predictions shape = (6, 20484)`
- `Saved predictions to: .../first_try_preds.npy`

## Local UI (upload your own media)

```bash
cd /Users/macos/Documents/Filter/test-meta-ai
./.venv/bin/streamlit run app.py
```

Then open the local URL shown by Streamlit (usually `http://localhost:8501`), upload your `.wav/.mp3/.mp4/.mov`, and click **Run TRIBE v2**.

## How to read analysis output

1. **Verify quality first**: confirm shape is 2D (`segments x vertices`) and output is non-empty.
2. **Locate high windows**: use z-score tiers (`z>=1.5` high, `z>=2.0` very high) to find stronger-response segments.
3. **Compare runs**: use baseline vs target in the app's compare section; check average response, variability, and count of high windows.

Important limitations:
- TRIBE output is a predicted neural-response pattern, not a direct emotion detector.
- Do not treat this as clinical/psychological diagnosis.
- Prefer relative interpretation (within-run and between-run comparisons), not fixed absolute thresholds.
