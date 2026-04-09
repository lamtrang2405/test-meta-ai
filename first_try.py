from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tribev2 import TribeModel
from tribev2.demo_utils import get_audio_and_text_events


def main() -> None:
    base = Path(__file__).resolve().parent
    cache_dir = base / "cache"
    input_wav = base / "first_try_input.wav"
    output_npy = base / "first_try_preds.npy"

    # Generate a short test tone so first try has no external file dependency.
    sr = 16_000
    seconds = 6
    t = np.linspace(0, seconds, sr * seconds, endpoint=False)
    x = 0.1 * np.sin(2 * np.pi * 220.0 * t)
    sf.write(input_wav, x, sr)

    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=str(cache_dir))
    model.data.audio_feature.device = "cpu"
    events = pd.DataFrame(
        [
            {
                "type": "Audio",
                "filepath": str(input_wav),
                "start": 0,
                "timeline": "default",
                "subject": "default",
            }
        ]
    )
    events = get_audio_and_text_events(events, audio_only=True)
    preds, segments = model.predict(events=events)

    # Save only predictions for easy inspection.
    np.save(output_npy, preds)
    print(f"OK: predictions shape = {preds.shape}")
    print(f"Saved predictions to: {output_npy}")
    print(f"Number of segments: {len(segments)}")


if __name__ == "__main__":
    main()
