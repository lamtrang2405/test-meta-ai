from pathlib import Path

import app


def main() -> None:
    media = Path("/Users/macos/Documents/Filter/tribev2/first_try_input.wav")
    model = app.load_model()
    preds, segments = app.run_inference(model, media)
    print("SELF-CHECK OK")
    print("shape:", preds.shape)
    print("segments:", len(segments))


if __name__ == "__main__":
    main()
