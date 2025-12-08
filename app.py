import random
import time
from pathlib import Path
from typing import Dict

import altair as alt
import pandas as pd
import streamlit as st


# =========================================================
# Configuration
# =========================================================

st.set_page_config(
    page_title="EmoLyrics AI",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Fixed order of emotions in the chart
EMOTION_ORDER = ["Anger", "Fear", "Joy", "Love", "Sadness", "Surprise"]

# Color palette for each emotion
COLOR_MAP = {
    "Anger": "#E63946",
    "Fear": "#F4A261",
    "Joy": "#2A9D8F",
    "Love": "#E76F51",
    "Sadness": "#457B9D",
    "Surprise": "#8D99AE",
}

# Session state key for current scores
SESSION_KEY_SCORES = "current_emotion_scores"

# Paths
BASE_DIR = Path(__file__).parent
CSS_FILE = BASE_DIR / "styles.css"
TEMPLATES_FILE = BASE_DIR / "templates.html"
TEMPLATES_CACHE: str | None = None  # loaded lazily


# Initialize state
if SESSION_KEY_SCORES not in st.session_state:
    st.session_state[SESSION_KEY_SCORES] = {
        emotion: 0.0 for emotion in EMOTION_ORDER
    }


# =========================================================
# Helpers: CSS and HTML templates
# =========================================================

def load_css() -> None:
    """Load the global CSS stylesheet."""
    if CSS_FILE.exists():
        css = CSS_FILE.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    else:
        st.warning(
            "Could not find 'styles.css'. "
            "Make sure it is in the same folder as app.py."
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
            "Make sure it exists and contains the template blocks."
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


# Load global CSS as soon as possible
load_css()


# =========================================================
# Helpers: emotion scores and chart
# =========================================================

def generate_random_scores() -> Dict[str, float]:
    """
    Generate a random, normalized score for each emotion.

    This is a placeholder for the real model inference.
    """
    raw_values = {emotion: random.random() for emotion in EMOTION_ORDER}
    total = sum(raw_values.values())
    return {k: v / total for k, v in raw_values.items()}


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


# =========================================================
# Main UI
# =========================================================

def main() -> None:
    """Main entrypoint for the Streamlit app."""
    # Header
    render_header()

    # Layout: left (lyrics input) / right (results)
    col_left, col_right = st.columns([1, 1])

    # ----- Left column: lyrics input -----
    with col_left:
        st.markdown(
            '<div class="section-title">'
            '<span class="icon">📝</span><span>Lyrics</span></div>',
            unsafe_allow_html=True,
        )

        input_mode = st.radio(
            "How do you want to provide the lyrics?",
            ["Type manually", "Upload file (.txt)"],
            horizontal=True,
        )

        lyrics: str = ""

        if input_mode == "Type manually":
            lyrics = st.text_area(
                "Lyrics input",
                height=320,
                placeholder="Type your song lyrics here...",
                label_visibility="collapsed",
                key="lyrics_manual",
            )
        else:
            uploaded_file = st.file_uploader(
                "Upload a plain-text (.txt) file with the lyrics",
                type=["txt"],
                key="lyrics_file_uploader",
            )

            if uploaded_file is not None:
                try:
                    file_text = uploaded_file.read().decode(
                        "utf-8",
                        errors="ignore",
                    )
                except Exception:
                    file_text = ""

                lyrics = st.text_area(
                    "Loaded lyrics",
                    value=file_text,
                    height=320,
                    label_visibility="collapsed",
                    key="lyrics_file",
                )
            else:
                st.info("Upload a file to view and edit the lyrics here.")
                lyrics = ""

        st.write("")
        run_button = st.button("✨ Analyze emotional profile")

    # ----- Right column: chart + result card -----
    with col_right:
        st.markdown(
            '<div class="section-title">'
            '<span class="icon">📊</span><span>Emotional profile</span></div>',
            unsafe_allow_html=True,
        )

        chart_placeholder = st.empty()
        result_placeholder = st.empty()

        # Initial render (previous scores or zeros)
        current_scores = st.session_state[SESSION_KEY_SCORES]
        render_emotion_chart(current_scores, chart_placeholder)

        # Analysis logic
        if run_button:
            if lyrics.strip():
                with st.spinner("Analyzing lyrics..."):
                    old_scores = st.session_state[SESSION_KEY_SCORES]
                    new_scores = generate_random_scores()

                    # Smooth animation between old and new values
                    steps = 25
                    for i in range(steps + 1):
                        t = i / steps
                        t_smooth = 1 - (1 - t) ** 3  # ease-out cubic

                        interpolated = {}
                        for emo in EMOTION_ORDER:
                            start = old_scores[emo]
                            end = new_scores[emo]
                            interpolated[emo] = start + (end - start) * t_smooth

                        render_emotion_chart(interpolated, chart_placeholder)
                        time.sleep(0.01)

                    st.session_state[SESSION_KEY_SCORES] = new_scores

                # Show result card
                top_emotion = max(new_scores, key=new_scores.get)
                top_score = new_scores[top_emotion]
                with result_placeholder:
                    render_result_card(top_emotion, top_score)
            else:
                st.warning(
                    "Please enter or upload some lyrics before running the analysis."
                )
                # If there are previous scores, keep showing the last result card
                if sum(current_scores.values()) > 0:
                    top_emotion = max(current_scores, key=current_scores.get)
                    with result_placeholder:
                        render_result_card(top_emotion, current_scores[top_emotion])
        else:
            # No new analysis: show last result, if any
            if sum(current_scores.values()) > 0:
                top_emotion = max(current_scores, key=current_scores.get)
                with result_placeholder:
                    render_result_card(top_emotion, current_scores[top_emotion])


if __name__ == "__main__":
    main()
