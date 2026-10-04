"""Streamlit UI。表示と入力のみを担当し、処理は DecisionFlow に委譲する。"""

from __future__ import annotations

from dataclasses import asdict, replace

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from jev_decision_router.config import Settings
from jev_decision_router.factory import (
    ConfigurationError,
    build_chat_client,
    build_decision_client,
)
from jev_decision_router.flow import DecisionFlow
from jev_decision_router.interfaces import ChatError, DecisionError
from jev_decision_router.router import DecisionRouter, RouteDecision
from jev_decision_router.routes import DEFAULT_ROUTES


def render_sidebar(defaults: Settings) -> tuple[Settings, bool, bool]:
    with st.sidebar:
        st.header("Settings")
        jev_base_url = st.text_input("Jev API URL", defaults.jev_base_url)
        jev_model = st.text_input("Jev model", defaults.jev_model)
        jev_api_key = st.text_input(
            "Jev API key",
            value=defaults.typesafe_api_key,
            type="password",
            help="You can also set TYPESAFE_API_KEY in .env",
        )
        jev_mock_mode = st.toggle(
            "Jev mock mode",
            value=defaults.jev_mock_mode,
            help="UI verification only. Turn this off to use the real Jev API.",
        )

        st.divider()
        vllm_base_url = st.text_input("vLLM OpenAI-compatible URL", defaults.vllm_base_url)
        vllm_model = st.text_input(
            "vLLM model (optional)",
            defaults.vllm_model,
            help="If empty, the app uses the first model returned by /v1/models.",
        )
        vllm_api_key = st.text_input("vLLM API key", defaults.vllm_api_key, type="password")

        st.divider()
        confidence_threshold = st.slider(
            "Auto-execution confidence threshold",
            min_value=0.0,
            max_value=1.0,
            value=defaults.confidence_threshold,
            step=0.05,
        )
        stop_on_human_review = st.toggle(
            "Stop before vLLM on human_review",
            value=True,
            help="Demonstrates a governance gate in the decision layer.",
        )
        stream_response = st.toggle(
            "Stream vLLM response",
            value=True,
            help="Show the vLLM answer token by token.",
        )

    settings = replace(
        defaults,
        typesafe_api_key=jev_api_key,
        jev_base_url=jev_base_url,
        jev_model=jev_model,
        jev_mock_mode=jev_mock_mode,
        confidence_threshold=confidence_threshold,
        vllm_base_url=vllm_base_url,
        vllm_model=vllm_model,
        vllm_api_key=vllm_api_key,
    )
    return settings, stop_on_human_review, stream_response


def render_decision(decision: RouteDecision, latency_ms: float, will_stop: bool) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Effective route", decision.route)
    c2.metric("Jev confidence", f"{decision.confidence:.1%}")
    c3.metric("Jev latency", f"{latency_ms:.0f} ms")

    st.markdown("### Choice probabilities")
    probabilities = pd.DataFrame(
        {
            "choice": list(decision.probabilities.keys()),
            "probability": list(decision.probabilities.values()),
        }
    ).set_index("choice")
    st.bar_chart(probabilities)

    st.markdown("### Decision path")
    gate = "Confidence gate → human_review" if decision.gated else "Confidence gate → pass"
    destination = "STOP / Human review" if will_stop else "local vLLM"
    st.code(
        "Human request\n"
        "   ↓\n"
        f"Jev Choice = {decision.raw_choice} (confidence={decision.confidence:.3f})\n"
        "   ↓\n"
        f"{gate}\n"
        "   ↓\n"
        f"{destination}",
        language="text",
    )

    with st.expander("Raw Jev result"):
        st.json(asdict(decision))


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="Jev Decision Layer + local vLLM", page_icon="🧭", layout="wide")
    st.title("🧭 Jev Decision Layer + local vLLM")
    st.caption(
        "Human request → Jev / Choice → confidence gate → local vLLM. Jev decides; vLLM generates."
    )

    settings, stop_on_human_review, stream_response = render_sidebar(Settings.from_env())

    request_text = st.text_area(
        "Human request",
        height=170,
        placeholder=(
            "Example: FastAPIでJWT認証つきAPIを実装したい。設計方針とサンプルコードを書いて。\n\n"
            "Example: 明日の取締役会に提出する売上悪化の原因を整理し、打ち手を比較して。"
        ),
    )

    st.markdown("### Jev Choice policy")
    st.dataframe(
        pd.DataFrame(
            [{"choice": r.name, "meaning": r.description} for r in DEFAULT_ROUTES.values()]
        ),
        use_container_width=True,
        hide_index=True,
    )

    if not st.button("Run decision flow", type="primary", use_container_width=True):
        return
    if not request_text.strip():
        st.warning("Please enter a request.")
        return

    try:
        decision_client = build_decision_client(settings)
    except ConfigurationError as exc:
        st.error(str(exc))
        return

    flow = DecisionFlow(
        router=DecisionRouter(decision_client, settings.confidence_threshold),
        chat_client=build_chat_client(settings),
        stop_on_human_review=stop_on_human_review,
    )

    with st.status("Jev is making a decision…", expanded=True) as status:
        try:
            timed = flow.decide(request_text)
        except DecisionError as exc:
            status.update(label="Jev request failed", state="error")
            st.exception(exc)
            return
        decision = timed.decision
        st.write(f"Raw Jev choice: `{decision.raw_choice}`")
        st.write(f"Confidence: `{decision.confidence:.3f}`")
        if decision.gated:
            st.warning(
                f"Confidence < {settings.confidence_threshold:.2f}; "
                "route was gated to `human_review`."
            )
        status.update(label="Decision complete", state="complete")

    will_stop = flow.should_stop(decision)
    render_decision(decision, timed.latency_ms, will_stop)

    if will_stop:
        st.warning(
            "The decision layer stopped automatic generation. "
            "A human should review the request or add missing context before execution."
        )
        return

    st.markdown("### local vLLM execution")
    st.caption(f"Route policy: {DEFAULT_ROUTES[decision.route].description}")
    if stream_response:
        render_streamed_response(flow, decision, request_text)
    else:
        render_response(flow, decision, request_text)


def render_response(flow: DecisionFlow, decision: RouteDecision, request_text: str) -> None:
    try:
        with st.spinner("Calling local vLLM…"):
            generated = flow.generate(decision, request_text)
    except ChatError as exc:
        st.exception(exc)
        return

    m1, m2 = st.columns(2)
    m1.metric("vLLM model", generated.response.model)
    m2.metric("vLLM latency", f"{generated.latency_ms:.0f} ms")

    st.markdown("### Response")
    st.write(generated.response.content)


def render_streamed_response(
    flow: DecisionFlow, decision: RouteDecision, request_text: str
) -> None:
    metrics = st.empty()
    st.markdown("### Response")
    try:
        stream = flow.stream(decision, request_text)
        st.write_stream(stream)
    except ChatError as exc:
        st.exception(exc)
        return

    with metrics.container():
        m1, m2, m3 = st.columns(3)
        m1.metric("vLLM model", stream.model)
        m2.metric("Time to first token", _format_ms(stream.first_chunk_ms))
        m3.metric("vLLM latency", _format_ms(stream.total_ms))


def _format_ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f} ms"


main()
