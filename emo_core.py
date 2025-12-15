# emo_core.py
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import altair as alt
import joblib
import pandas as pd
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer
from training.test_inference import LinearReLUDropoutLinearNet

# =========================================================
# Config / constants
# =========================================================

# Fixed order of emotions in the chart
EMOTION_ORDER = ["Anger", "Fear", "Joy", "Love", "Sadness"]

# Color palette for each emotion
COLOR_MAP = {
    "Anger": "#E63946",
    "Fear": "#F4A261",
    "Joy": "#2A9D8F",
    "Love": "#E76F51",
    "Sadness": "#457B9D",
}

# Session state keys
SESSION_KEY_SCORES = "current_emotion_scores"
SESSION_KEY_LYRICS = "current_lyrics"
SESSION_KEY_VERSIONS = "saved_versions_df"

# Paths (relative to project root)
BASE_DIR = Path(__file__).parent
CSS_FILE = BASE_DIR / "interface/styles.css"
TEMPLATES_FILE = BASE_DIR / "interface/templates.html"
MODEL_PATH = BASE_DIR / "training" / "models" / "emotion_classifier_v2.pt"

# Templates cache
TEMPLATES_CACHE: str | None = None  # loaded lazily

# Model cache (loaded once per session)
MODEL_CACHE: Optional[Tuple[torch.nn.Module, SentenceTransformer, Optional[object]]] = None

MOCK_MODEL = False


# =========================================================
# Session state init
# =========================================================

def init_session_state() -> None:
    """Ensure all required session_state keys exist."""
    if SESSION_KEY_SCORES not in st.session_state:
        st.session_state[SESSION_KEY_SCORES] = {
            emotion: 0.0 for emotion in EMOTION_ORDER
        }

    if SESSION_KEY_LYRICS not in st.session_state:
        st.session_state[SESSION_KEY_LYRICS] = ""

    if SESSION_KEY_VERSIONS not in st.session_state:
        cols = ["version_id", "title", "lyrics"] + EMOTION_ORDER
        st.session_state[SESSION_KEY_VERSIONS] = pd.DataFrame(columns=cols)


# =========================================================
# CSS & HTML templates
# =========================================================

def load_css() -> None:
    """Load the global CSS stylesheet."""
    if CSS_FILE.exists():
        css = CSS_FILE.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    else:
        st.warning(
            "Could not find 'styles.css'. "
            "Make sure it is in 'interface/styles.css'."
        )


def _load_templates_cache() -> None:
    """Load templates.html into memory (if not already loaded)."""
    global TEMPLATES_CACHE
    if TEMPLATES_CACHE is not None:
        return

    if TEMPLATES_FILE.exists():
        TEMPLATES_CACHE = TEMPLATES_FILE.read_text(encoding="utf-8")
    else:
        TEMPLATES_CACHE = ""
        st.error(
            "Could not find 'templates.html'. "
            "Make sure it exists in 'interface/templates.html'."
        )


def get_template_block(block_name: str) -> str:
    """
    Return a template block from templates.html.

    Blocks are delimited using HTML comments, e.g.:

    <!-- HEADER_TEMPLATE_START --> ... <!-- HEADER_TEMPLATE_END -->
    """
    _load_templates_cache()
    if not TEMPLATES_CACHE:
        return ""

    start = f"<!-- {block_name}_START -->"
    end = f"<!-- {block_name}_END -->"

    try:
        return TEMPLATES_CACHE.split(start)[1].split(end)[0].strip()
    except IndexError:
        st.error(f"Template block '{block_name}' not found in templates.html")
        return ""


def render_header() -> None:
    """Render the top header from the HEADER_TEMPLATE block."""
    header_html = get_template_block("HEADER_TEMPLATE")
    if header_html:
        st.markdown(header_html, unsafe_allow_html=True)


def render_result_card(emotion: str, score: float) -> None:
    """Fill and render the RESULT_CARD_TEMPLATE block."""
    template = get_template_block("RESULT_CARD_TEMPLATE")
    if not template:
        return

    html = (
        template.replace("{{EMOTION_CLASS}}", emotion)
        .replace("{{EMOTION_NAME}}", emotion.upper())
        .replace("{{SCORE}}", f"{score:.1%}")
    )
    st.markdown(html, unsafe_allow_html=True)


# =========================================================
# Model / scores
# =========================================================

def _load_model() -> Tuple[torch.nn.Module, SentenceTransformer, Optional[object]]:
    """
    Load the emotion classification model, SentenceTransformer, and scaler.
    Uses caching to avoid reloading on every call.
    """
    global MODEL_CACHE
    
    if MODEL_CACHE is not None:
        return MODEL_CACHE
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {MODEL_PATH}. "
            "Please train the model first or check the path."
        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint dictionary
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    
    # Extract model parameters from checkpoint
    input_dim = checkpoint["input_dim"]
    num_classes = checkpoint["n_classes"]
    
    # Create and load model
    model = LinearReLUDropoutLinearNet(input_dim=input_dim, num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # Set to evaluation mode
    
    # Load SentenceTransformer
    sentence_transformer_name = checkpoint.get("sentence_transformer_name", "all-MiniLM-L6-v2")
    sentence_transformer = SentenceTransformer(sentence_transformer_name)
    
    # Load scaler if needed
    scaler = None
    if checkpoint.get("scale_embeddings", False):
        scaler_path = MODEL_PATH.parent / "scaler.joblib"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
        else:
            st.warning(f"Scaler file not found at {scaler_path}, proceeding without scaling")
    
    # Cache the loaded model
    MODEL_CACHE = (model, sentence_transformer, scaler)
    return MODEL_CACHE


def generate_emotion_scores(lyrics: str) -> Dict[str, float]:
    """
    Generate emotion scores for a given lyrics string.

    Returns a dict {emotion: score} where scores sum to 1.
    """
    if MOCK_MODEL:
        raw_values = {emotion: random.random() for emotion in EMOTION_ORDER}
        total = sum(raw_values.values()) or 1.0
        return {k: v / total for k, v in raw_values.items()}
    
    try:
        model, sentence_transformer, scaler = _load_model()
        device = next(model.parameters()).device
        
        # Generate embedding
        embedding = sentence_transformer.encode(lyrics, convert_to_numpy=True)
        
        # Apply scaler if available
        if scaler is not None:
            embedding = scaler.transform(embedding.reshape(1, -1))[0]
        
        # Convert to tensor and predict
        x = torch.from_numpy(embedding).float().unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        
        return {emotion: float(score) for emotion, score in zip(EMOTION_ORDER, probs)}
    
    except Exception as e:
        st.error(f"Error generating emotion scores: {str(e)}")
        return {emotion: 0.0 for emotion in EMOTION_ORDER}

# =========================================================
# Versions storage (session only, no disk)
# =========================================================

def load_versions() -> pd.DataFrame:
    """Load saved versions from session state."""
    return st.session_state[SESSION_KEY_VERSIONS]


def save_versions(df: pd.DataFrame) -> None:
    """Persist versions DataFrame in session state."""
    st.session_state[SESSION_KEY_VERSIONS] = df


# =========================================================
# Charts
# =========================================================

def render_emotion_chart(
    scores: Dict[str, float],
    placeholder: st.delta_generator.DeltaGenerator,
) -> None:
    """Render the emotion bar chart given a dict of scores."""
    df = pd.DataFrame(list(scores.items()), columns=["Emotion", "Score"])

    max_val = df["Score"].max()
    if max_val == 0:
        max_val = 0.1
    y_limit = min(max_val * 1.2, 1.0)

    chart = (
        alt.Chart(df)
        .mark_bar(
            cornerRadiusTopLeft=10,
            cornerRadiusTopRight=10,
        )
        .encode(
            x=alt.X(
                "Emotion",
                sort=EMOTION_ORDER,
                axis=alt.Axis(
                    labelAngle=0,
                    title=None,
                    labelColor="#6b7280",
                    ticks=False,
                    domain=False,
                ),
            ),
            y=alt.Y(
                "Score",
                scale=alt.Scale(domain=[0, y_limit]),
                axis=alt.Axis(
                    format="%",
                    title=None,
                    tickCount=5,
                    grid=True,
                    gridDash=[2, 4],
                    gridColor="#e5e7eb",
                    domain=False,
                ),
            ),
            color=alt.Color(
                "Emotion",
                scale=alt.Scale(
                    domain=list(COLOR_MAP.keys()),
                    range=list(COLOR_MAP.values()),
                ),
                legend=None,
            ),
            tooltip=["Emotion", alt.Tooltip("Score", format=".1%")],
        )
        .properties(
            height=330,
            padding={"left": 10, "top": 5, "right": 10, "bottom": 30},
        )
        .configure_view(strokeWidth=0)
        .configure_axis(labelFont="Inter", labelFontSize=12)
    )

    placeholder.altair_chart(chart, use_container_width=True)


def render_compare_scatter(versions_df: pd.DataFrame) -> None:
    """
    Render a scatter+line plot comparing emotion scores across versions.

    x = Emotion, y = Score, color = Version title.
    """
    long_df = versions_df.melt(
        id_vars=["title"],
        value_vars=EMOTION_ORDER,
        var_name="Emotion",
        value_name="Score",
    )

    base = alt.Chart(long_df).encode(
        x=alt.X(
            "Emotion:N",
            sort=EMOTION_ORDER,
            axis=alt.Axis(
                labelAngle=0,
                title=None,
                labelColor="#6b7280",
                ticks=False,
                domain=False,
            ),
        ),
        y=alt.Y(
            "Score:Q",
            axis=alt.Axis(
                format="%",
                title=None,
                tickCount=5,
                grid=True,
                gridDash=[2, 4],
                gridColor="#e5e7eb",
                domain=False,
            ),
        ),
        color=alt.Color("title:N", title="Version"),
        tooltip=["title", "Emotion", alt.Tooltip("Score", format=".1%")],
    )

    line = base.mark_line()
    points = base.mark_point(size=120, filled=True)

    chart = (
        (line + points)
        .properties(
            height=330,
            padding={"left": 10, "top": 5, "right": 10, "bottom": 30},
        )
        .configure_view(strokeWidth=0)
        .configure_axis(labelFont="Inter", labelFontSize=12)
    )

    st.altair_chart(chart, use_container_width=True)
