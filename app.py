from __future__ import annotations

import os
from datetime import datetime
import json
from pathlib import Path

# Disable CUDA before importing torch/tribev2 stacks.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pandas as pd
import streamlit as st
import torch
from moviepy import VideoFileClip
from tribev2 import TribeModel
from tribev2.demo_utils import get_audio_and_text_events


APP_DIR = Path(__file__).resolve().parent
CACHE_DIR = APP_DIR / "cache"
UPLOAD_DIR = APP_DIR / "uploads"
OUTPUT_DIR = APP_DIR / "outputs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def contiguous_ranges(indices: np.ndarray) -> list[tuple[int, int]]:
    if len(indices) == 0:
        return []
    ranges: list[tuple[int, int]] = []
    start = int(indices[0])
    prev = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
            continue
        ranges.append((start, prev))
        start = idx
        prev = idx
    ranges.append((start, prev))
    return ranges


def format_ranges(indices: np.ndarray) -> str:
    ranges = contiguous_ranges(indices)
    if not ranges:
        return "none"
    parts = [f"{a}" if a == b else f"{a}-{b}" for a, b in ranges]
    return ", ".join(parts)


def summarize_signal(mean_signal: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray], str]:
    mu = float(mean_signal.mean())
    sigma = float(mean_signal.std())
    z = np.zeros_like(mean_signal) if sigma == 0 else (mean_signal - mu) / sigma
    tiers = {
        "very_high": np.where(z >= 2.0)[0],
        "high": np.where((z >= 1.5) & (z < 2.0))[0],
        "medium": np.where((z >= 0.5) & (z < 1.5))[0],
        "low": np.where(z < 0.5)[0],
    }
    trend = "stable" if sigma < 0.02 else "dynamic"
    summary = (
        f"Global response is {trend} (mean={mu:.4f}, std={sigma:.4f}). "
        f"High-response windows (z>=1.5): {format_ranges(np.where(z >= 1.5)[0])}."
    )
    return z, tiers, summary


def compare_runs(a: np.ndarray, b: np.ndarray) -> str:
    a_m = a.mean(axis=1)
    b_m = b.mean(axis=1)
    a_mu, b_mu = float(a_m.mean()), float(b_m.mean())
    a_sd, b_sd = float(a_m.std()), float(b_m.std())
    a_z = np.zeros_like(a_m) if a_sd == 0 else (a_m - a_mu) / a_sd
    b_z = np.zeros_like(b_m) if b_sd == 0 else (b_m - b_mu) / b_sd
    a_hi = int((a_z >= 1.5).sum())
    b_hi = int((b_z >= 1.5).sum())
    mean_delta = b_mu - a_mu
    var_delta = b_sd - a_sd
    mean_word = "higher" if mean_delta > 0 else "lower"
    var_word = "more variable" if var_delta > 0 else "more stable"
    high_word = "more" if b_hi > a_hi else "fewer"
    return (
        f"Run B is {mean_word} in average global response ({b_mu:.4f} vs {a_mu:.4f}), "
        f"is {var_word} ({b_sd:.4f} vs {a_sd:.4f}), and has {high_word} high-response "
        f"segments (z>=1.5: {b_hi} vs {a_hi})."
    )


def _as_float_or_none(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def infer_segment_times(segments: list, n_segments: int) -> list[tuple[float | None, float | None]]:
    """
    Best-effort extraction of (start, end) time from model segments.
    Falls back to (None, None) when unavailable.
    """
    out: list[tuple[float | None, float | None]] = []
    for i in range(n_segments):
        if i >= len(segments):
            out.append((None, None))
            continue
        seg = segments[i]
        start = end = None
        if isinstance(seg, dict):
            for k in ("start", "onset", "t_start", "segment_start"):
                start = _as_float_or_none(seg.get(k))
                if start is not None:
                    break
            for k in ("end", "offset", "t_end", "segment_end"):
                end = _as_float_or_none(seg.get(k))
                if end is not None:
                    break
        elif isinstance(seg, (tuple, list)) and len(seg) >= 2:
            start = _as_float_or_none(seg[0])
            end = _as_float_or_none(seg[1])
        out.append((start, end))
    return out


def format_time_range(start: float | None, end: float | None) -> str:
    def _clock(seconds: float) -> str:
        total = max(0.0, float(seconds))
        h = int(total // 3600)
        m = int((total % 3600) // 60)
        s = total % 60
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:04.1f}"
        return f"{m:02d}:{s:04.1f}"

    if start is None and end is None:
        return "unknown"
    if start is not None and end is not None:
        return f"{_clock(start)}-{_clock(end)}"
    if start is not None:
        return f"{_clock(start)}-?"
    return f"?-{_clock(end)}"


def top_impressive_segments(
    mean_over_time: np.ndarray,
    z: np.ndarray,
    seg_times: list[tuple[float | None, float | None]],
    top_k: int = 5,
) -> pd.DataFrame:
    n = len(mean_over_time)
    idx = np.argsort(z)[::-1][: min(top_k, n)]
    rows = []
    for rank, i in enumerate(idx, start=1):
        s, e = seg_times[int(i)] if i < len(seg_times) else (None, None)
        rows.append(
            {
                "rank": rank,
                "time_range": format_time_range(s, e),
                "mean_activation": float(mean_over_time[int(i)]),
                "z_score": float(z[int(i)]),
                "segment": int(i),
            }
        )
    return pd.DataFrame(rows)


def top_drop_transitions(
    mean_over_time: np.ndarray,
    seg_times: list[tuple[float | None, float | None]],
    top_k: int = 5,
) -> pd.DataFrame:
    if len(mean_over_time) < 2:
        return pd.DataFrame(columns=["from_time", "to_time", "delta", "from_segment", "to_segment"])
    diff = mean_over_time[1:] - mean_over_time[:-1]
    drop_idx = np.argsort(diff)[: min(top_k, len(diff))]
    rows = []
    for i in drop_idx:
        s0, e0 = seg_times[int(i)] if i < len(seg_times) else (None, None)
        s1, e1 = seg_times[int(i + 1)] if i + 1 < len(seg_times) else (None, None)
        rows.append(
            {
                "from_time": format_time_range(s0, e0),
                "to_time": format_time_range(s1, e1),
                "delta": float(diff[int(i)]),
                "from_segment": int(i),
                "to_segment": int(i + 1),
            }
        )
    return pd.DataFrame(rows)


def force_cpu_features(model: TribeModel) -> None:
    for attr in ("audio_feature", "text_feature", "video_feature"):
        try:
            feature = getattr(model.data, attr, None)
            if feature is not None and hasattr(feature, "device"):
                feature.device = "cpu"
        except Exception:
            pass


def load_model() -> TribeModel:
    # Hard guard against any CUDA selection in downstream libs.
    torch.cuda.is_available = lambda: False
    model = TribeModel.from_pretrained(
        "facebook/tribev2",
        cache_folder=str(CACHE_DIR),
        device="cpu",
    )
    # Force CPU on Macs without CUDA.
    force_cpu_features(model)
    return model


def run_inference(model: TribeModel, media_path: Path) -> tuple[np.ndarray, list]:
    suffix = media_path.suffix.lower()
    if suffix in {".wav", ".mp3", ".flac", ".ogg"}:
        events = pd.DataFrame(
            [
                {
                    "type": "Audio",
                    "filepath": str(media_path),
                    "start": 0,
                    "timeline": "default",
                    "subject": "default",
                }
            ]
        )
        events = get_audio_and_text_events(events, audio_only=True)
    elif suffix in {".mp4", ".avi", ".mkv", ".mov", ".webm"}:
        # Convert video to wav and run the audio-only path to avoid video extractor/CUDA paths.
        extracted_wav = UPLOAD_DIR / f"{media_path.stem}_audio.wav"
        with VideoFileClip(str(media_path)) as video:
            if video.audio is None:
                raise ValueError("Video has no audio track.")
            video.audio.write_audiofile(
                str(extracted_wav),
                fps=16000,
                bitrate="128k",
                logger=None,
            )
        events = pd.DataFrame(
            [
                {
                    "type": "Audio",
                    "filepath": str(extracted_wav),
                    "start": 0,
                    "timeline": "default",
                    "subject": "default",
                }
            ]
        )
        events = get_audio_and_text_events(events, audio_only=True)
    else:
        raise ValueError("Unsupported file type.")
    force_cpu_features(model)
    return model.predict(events=events)


st.set_page_config(page_title="TRIBE v2 Local Runner", layout="centered")
st.title("TRIBE v2 Local Runner")
st.write("Upload audio/video, run local inference, and save predictions to `.npy`.")
st.caption("CPU mode can take a few minutes per file.")

uploaded = st.file_uploader(
    "Choose media file",
    type=["wav", "mp3", "flac", "ogg", "mp4", "avi", "mkv", "mov", "webm"],
)

if uploaded is not None:
    media_path = UPLOAD_DIR / uploaded.name
    media_path.write_bytes(uploaded.getbuffer())
    st.success(f"Saved upload: {media_path}")

    if st.button("Run TRIBE v2", type="primary"):
        with st.spinner("Loading model and running inference..."):
            try:
                model = load_model()
                preds, segments = run_inference(model, media_path)

                out_name = f"{media_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npy"
                out_path = OUTPUT_DIR / out_name
                np.save(out_path, preds)
                sidecar_path = out_path.with_suffix(".segments.json")
                seg_times = infer_segment_times(segments, int(preds.shape[0]))
                sidecar_payload = {
                    "source_media": str(media_path.name),
                    "created_at": datetime.now().isoformat(),
                    "n_segments": int(preds.shape[0]),
                    "segment_times": [
                        {"segment": int(i), "start": t[0], "end": t[1]} for i, t in enumerate(seg_times)
                    ],
                }
                sidecar_path.write_text(json.dumps(sidecar_payload, indent=2), encoding="utf-8")

                st.success("Inference complete.")
                st.write(f"Prediction shape: `{preds.shape}`")
                st.write(f"Segments kept: `{len(segments)}`")
                st.write(f"Saved output: `{out_path}`")
                st.write(f"Saved segment metadata: `{sidecar_path}`")

                st.download_button(
                    "Download predictions (.npy)",
                    data=out_path.read_bytes(),
                    file_name=out_name,
                    mime="application/octet-stream",
                )
            except Exception as e:
                st.error(f"Run failed: {e}")

st.divider()
st.subheader("Analyze Existing Results (.npy)")
st.caption(
    "Interpretation guide: rows are kept time segments; columns are cortical vertices "
    "(brain locations on the fsaverage5 surface). Values are model-predicted activity."
)
result_files = sorted(OUTPUT_DIR.glob("*.npy"), key=lambda p: p.stat().st_mtime, reverse=True)
if not result_files:
    st.info("No result files found in `outputs/` yet.")
else:
    selected = st.selectbox(
        "Choose result file",
        options=result_files,
        format_func=lambda p: p.name,
    )
    if st.button("Analyze Result", type="secondary"):
        try:
            arr = np.load(selected)
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D array, got shape {arr.shape}")
            n_segments, n_vertices = arr.shape

            st.success("Analysis ready.")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Segments", f"{n_segments}")
            c2.metric("Vertices", f"{n_vertices}")
            c3.metric("Mean", f"{float(arr.mean()):.4f}")
            c4.metric("Std", f"{float(arr.std()):.4f}")
            st.markdown(
                "- **Segments**: number of time windows retained for prediction.\n"
                "- **Vertices**: number of brain surface points predicted per segment.\n"
                "- **Mean**: average predicted activity across all values.\n"
                "- **Std**: spread/variability of predicted activity."
            )
            st.caption(
                f"min={float(arr.min()):.4f}, max={float(arr.max()):.4f}, file=`{selected.name}`"
            )
            st.caption(
                "Higher absolute values indicate stronger predicted response at a vertex/segment. "
                "Min/max show extremes; compare runs by shape and overall range/variance."
            )

            mean_over_time = arr.mean(axis=1)
            z, tiers, summary = summarize_signal(mean_over_time)
            sidecar = selected.with_suffix(".segments.json")
            seg_times = [(None, None)] * n_segments
            if sidecar.exists():
                try:
                    meta = json.loads(sidecar.read_text(encoding="utf-8"))
                    raw_times = meta.get("segment_times", [])
                    seg_times = [
                        (
                            _as_float_or_none(item.get("start")),
                            _as_float_or_none(item.get("end")),
                        )
                        for item in raw_times[:n_segments]
                    ]
                    if len(seg_times) < n_segments:
                        seg_times.extend([(None, None)] * (n_segments - len(seg_times)))
                    st.caption(f"Loaded timing metadata from `{sidecar.name}`.")
                except Exception:
                    st.caption("Timing metadata found but unreadable; using segment indices only.")
            else:
                st.caption("No timing metadata sidecar found; showing segment indices only.")
                st.caption("Tip: rerun inference with this app version to generate exact time ranges.")

            st.markdown("### Interpretation Summary")
            st.write(summary)
            st.caption(
                "These are relative response-strength tiers only, not direct emotion labels."
            )
            st.markdown(
                "- **Very high (z>=2.0)**: "
                f"{format_ranges(tiers['very_high'])}\n"
                "- **High (1.5<=z<2.0)**: "
                f"{format_ranges(tiers['high'])}\n"
                "- **Medium (0.5<=z<1.5)**: "
                f"{format_ranges(tiers['medium'])}\n"
                "- **Low (z<0.5)**: "
                f"{format_ranges(tiers['low'])}"
            )
            st.markdown("### What This Might Indicate")
            high_idx = np.where(z >= 1.5)[0]
            if len(high_idx) == 0:
                st.write(
                    "No strong high-response windows in this run; response appears broadly steady."
                )
            else:
                consistency = "sustained" if len(contiguous_ranges(high_idx)) < len(high_idx) else "spiky"
                st.write(
                    f"High windows suggest salience/novelty or attention-load changes. Pattern appears {consistency}."
                )
            st.markdown("### What We Cannot Claim")
            st.warning(
                "This output is a predicted neural response pattern. It is not a validated emotion "
                "classifier and should not be used for clinical or psychological diagnosis."
            )

            df_mean = pd.DataFrame(
                {"segment": np.arange(n_segments), "mean_activation": mean_over_time}
            ).set_index("segment")
            st.write("Mean activation over segments")
            st.line_chart(df_mean)
            st.caption(
                "This trend summarizes global predicted brain response over time. "
                "Peaks suggest segments with stronger overall activation."
            )
            st.write("High-response time-frame table (z-score)")
            hi_df = pd.DataFrame(
                {
                    "time_range": [format_time_range(*seg_times[i]) for i in range(n_segments)],
                    "segment": np.arange(n_segments),
                    "z_score": z,
                    "tier": np.where(
                        z >= 2.0,
                        "very_high",
                        np.where(z >= 1.5, "high", np.where(z >= 0.5, "medium", "low")),
                    ),
                }
            )
            st.dataframe(hi_df, use_container_width=True, height=220, hide_index=True)
            st.write("Most impressive windows (top z-score)")
            st.dataframe(
                top_impressive_segments(mean_over_time, z, seg_times, top_k=5),
                use_container_width=True,
                height=220,
                hide_index=True,
            )
            st.caption(
                "These are the strongest global-response windows in this file. "
                "Use `time_range` to locate moments in your video/audio."
            )
            st.write("Strongest drop transitions")
            st.dataframe(
                top_drop_transitions(mean_over_time, seg_times, top_k=5),
                use_container_width=True,
                height=220,
                hide_index=True,
            )
            st.caption(
                "Negative `delta` values mean response dropped from one segment to the next."
            )

            top_k = min(8, n_vertices)
            var_idx = np.argsort(arr.var(axis=0))[-top_k:]
            df_top = pd.DataFrame(
                arr[:, var_idx],
                columns=[f"v{int(i)}" for i in var_idx],
                index=np.arange(n_segments),
            )
            st.write("Top-variance vertices")
            st.line_chart(df_top)
            st.caption(
                "These are the most dynamic vertices (largest variance). "
                "Use this to spot where response changes most across segments."
            )

            heat_cols = min(120, n_vertices)
            sample_idx = np.linspace(0, n_vertices - 1, num=heat_cols, dtype=int)
            heat = pd.DataFrame(arr[:, sample_idx])
            st.write("Compact heatmap (sampled vertices)")
            st.dataframe(
                heat.style.background_gradient(cmap="viridis"),
                use_container_width=True,
                height=260,
            )
            st.caption(
                "Heatmap colors: darker/lighter cells indicate lower/higher predicted activity "
                "for sampled vertices across time segments."
            )
        except Exception as e:
            st.error(f"Analysis failed: {e}")

    st.markdown("### Compare Two Runs")
    if len(result_files) >= 2:
        c_a, c_b = st.columns(2)
        file_a = c_a.selectbox(
            "Run A (baseline)",
            options=result_files,
            index=1 if len(result_files) > 1 else 0,
            format_func=lambda p: p.name,
            key="cmp_a",
        )
        file_b = c_b.selectbox(
            "Run B (target)",
            options=result_files,
            index=0,
            format_func=lambda p: p.name,
            key="cmp_b",
        )
        if st.button("Compare Runs"):
            try:
                arr_a = np.load(file_a)
                arr_b = np.load(file_b)
                if arr_a.ndim != 2 or arr_b.ndim != 2:
                    raise ValueError("Both files must be 2D arrays.")
                st.write(compare_runs(arr_a, arr_b))
            except Exception as e:
                st.error(f"Comparison failed: {e}")
    else:
        st.caption("Need at least two `.npy` files in `outputs/` to compare runs.")
