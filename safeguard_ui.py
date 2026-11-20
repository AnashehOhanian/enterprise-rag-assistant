"""Streamlit estimator for the Safeguard 4U material catalog.

On this computer:

    python app.py
    streamlit run safeguard_ui.py

From a phone on the same Wi-Fi, run run_phone.ps1. That listens on every
network interface at port 8502.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import html
import io
import json
import os
import uuid
from pathlib import Path
from typing import Any
from urllib import error, request

import safeguard_i18n as i18n
import streamlit as st
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent


def _project_path(*parts: str) -> Path:
    """Return one file inside this project, from the folder that contains this script."""
    return _ROOT.joinpath(*parts)


load_dotenv(_project_path(".env"))

REQUEST_TIMEOUT_SECONDS = 60
PHOTO_TIMEOUT_SECONDS = 90
MOCKUP_TIMEOUT_SECONDS = 120

_FAMILIES = (
    ("wide_flange", "Wide flange (W)"),
    ("american_standard", "American standard (S)"),
    ("channel", "Channel (C)"),
    ("angle", "Angle (L)"),
)
_TUBE_SHAPES = (
    ("square", "Square HSS"),
    ("rectangular", "Rectangular HSS"),
    ("pipe", "Pipe"),
)
_TARGET_PROJECTS = (
    ("Patio & Outdoor Structures", "patio"),
    ("Custom Gates", "gate"),
    ("Stairs & Railings", "stair_railing"),
    ("Railings & Balustrades", "railing"),
    ("Structural Steel", "structural"),
)
_PREVIEW_FAMILIES = {"patio", "gate", "stair", "stair_railing", "railing"}
_PORTFOLIO_ROOT = _project_path("portfolio")
_PORTFOLIO_FOLDERS = {
    "patio": "patio",
    "gate": "gate",
    "stair": "stairs",
    "stair_railing": "stairs",
    "railing": "railing",
}
_PORTFOLIO_KINDS = {
    "patio": "Patio cover",
    "gate": "Custom gate",
    "stair": "Steel stair",
    "stair_railing": "Steel stair",
    "railing": "Steel railing",
}
_PORTFOLIO_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
_EXAMPLE_ROOT = _project_path("examples")
_LOGO_PATH = _project_path("assets", "safeguard4u-logo.png")
_STYLE_EXAMPLES = (
    ("modern_horizontal_gate", "Modern Horizontal Gate", "modern-horizontal-gate"),
    ("ornamental_iron_gate", "Ornamental Iron Gate", "ornamental-iron-gate"),
    ("stair_railing", "Stair Railing", "stair-railing"),
    ("outdoor_patio_frame", "Outdoor Patio Frame", "outdoor-patio-frame"),
)
_PREVIEW_SUGGESTIONS = {
    "patio": "Suggested preview: a charcoal steel cover of posts, beams, and rafters installed on the outdoor area in this photo.",
    "gate": "Suggested preview: one charcoal steel gate of square tube and flat bar installed at the entrance in this photo.",
    "stair": "Suggested preview: a charcoal steel stair flight installed on the visible run in this photo.",
    "stair_railing": "Suggested preview: a charcoal steel stair flight and guard installed on the visible run in this photo.",
    "railing": "Suggested preview: a charcoal steel railing of posts, rails, and infill installed along the visible edge in this photo.",
    "structural": (
        "Suggested preview: the proposed steel frame on the visible structure. "
        "A drawn preview is offered for a patio, gate, stairs, or railing."
    ),
}
_MATERIAL_TYPES = (
    "Rolled profile",
    "Tube or pipe",
    "Plate",
    "Gate",
    "Stairs",
    "Railings & Balustrades",
    "Patio & Outdoor Structures",
    "Known weight",
)
_RAILING_INFILL = {
    "vertical_picket": "Vertical 1/2 in square bar",
    "horizontal_picket": "Horizontal 1/2 in square bar",
    "cable": "Horizontal cable",
    "ornamental": "Ornamental square bar",
}
_STAIR_LAYOUTS = {
    "straight": "Straight",
    "spiral": "Spiral",
}
_PATIO_KINDS = {
    "pergola": "Posts, beams, rafters, and lattice",
    "cover": "Posts, beams, rafters, and purlins",
    "framing": "Posts and beams",
    "railing": "Posts and three rails",
}


class ApiError(Exception):
    """A response the Flask API rejected or could not complete."""


_HOST_SECRET_NAMES = (
    "OPENAI_API_KEY",
    "GROQ_API_KEY",
    "LLM_API_KEY",
    "LLM_API_BASE",
    "API_BASE_URL",
    "LLM_MODEL",
    "VISION_MODEL",
    "IMAGE_MODEL",
    "APP_API_KEY",
    "SHOP_OWNER_NAME",
    "SHOP_PHONE",
    "SHOP_EMAIL",
    "SHOP_PASSCODE",
    "SHOP_NOTIFY_LANGUAGE",
    "TWILIO_ACCOUNT_SID",
    "TWILIO_AUTH_TOKEN",
    "TWILIO_FROM_NUMBER",
    "SHOP_SMS_EMAIL",
    "SMTP_HOST",
    "SMTP_PORT",
    "SMTP_USER",
    "SMTP_PASSWORD",
    "SMTP_FROM",
)


def _apply_host_secrets() -> None:
    """Fill empty settings from Streamlit secrets without printing the values."""
    try:
        secrets = st.secrets
    except Exception:
        return
    for name in _HOST_SECRET_NAMES:
        if os.getenv(name, "").strip():
            continue
        try:
            value = secrets[name]
        except Exception:
            continue
        text = str(value).strip()
        if text:
            os.environ[name] = text


def _embed_api() -> None:
    """Start Flask only when SAFEGUARD_EMBED_API is explicitly turned on.

    Streamlit Community Cloud leaves that flag unset and runs the estimator
    in this process instead.
    """
    flag = os.getenv("SAFEGUARD_EMBED_API", "").strip().lower()
    if flag not in {"1", "true", "yes"}:
        return
    _apply_host_secrets()
    import app as api

    api.serve_embedded()


def _apply_theme() -> None:
    """Paint buttons, tabs, chat, and estimate notes in the logo colors.

    The base palette lives in ``.streamlit/config.toml``. This layer only
    adjusts the pieces that theme file cannot reach.
    """
    st.markdown(
        """
        <style>
        header[data-testid="stHeader"] {
            background: #1F262E;
        }
        header[data-testid="stHeader"] button,
        header[data-testid="stHeader"] svg {
            color: #F4F6F8;
            fill: #F4F6F8;
        }
        [data-testid="stChatMessage"] {
            background: #F4F1EB;
            border: 1px solid #D6DCE2;
            border-radius: 0.6rem;
        }
        [data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) {
            background-color: #F4F1EB;
            border: 1px solid #E4D3B5;
        }
        [data-testid="stAlertContentInfo"] p {
            color: #1E242B;
        }
        div[data-testid="stFormSubmitButton"] button,
        button[data-testid="stBaseButton-primary"] {
            background-color: #B88746;
            color: #1E242B;
            border: 1px solid #A6783C;
        }
        div[data-testid="stFormSubmitButton"] button:hover,
        button[data-testid="stBaseButton-primary"]:hover {
            background-color: #C59B5F;
            color: #1E242B;
            border-color: #B88746;
        }
        button[data-baseweb="tab"][aria-selected="true"],
        button[data-baseweb="tab"][aria-selected="true"] p {
            color: #8C6239;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Render the takeoff form, the price bracket, and the recommendations."""
    st.set_page_config(
        page_title="Safeguard 4U AI Estimator & Structural Assistant",
        page_icon=":material/straighten:",
        layout="wide",
    )
    _apply_theme()
    _apply_host_secrets()
    _embed_api()
    api_base, api_key, vision_key = _sidebar()
    catalog, load_error = _load_catalog(api_base, api_key)

    st.title("Safeguard 4U AI Estimator & Structural Assistant")
    st.caption(
        "Safeguard 4U planning brackets for structural steel, tube, plate, gates, stairs, railings, and outdoor frames. "
        "Carry the low and the high together."
    )

    if load_error:
        st.error(load_error)
        return
    if catalog is None:
        return

    names = ["Estimate", "Ask", "Catalog"]
    if _shop_unlocked():
        names = ["Estimate", "Ask", "Shop", "Catalog"]
    tabs = st.tabs(names)
    with tabs[0]:
        _practice_notes()
        payload = _takeoff_form(catalog)
        signature = json.dumps(payload, sort_keys=True)
        if st.button("Build price range", type="primary"):
            _submit(api_base, api_key, payload, signature)
        _show_saved_result(signature)
        _show_addons(catalog, payload, signature, api_base, api_key, vision_key)
    with tabs[1]:
        _client_chat(api_base, api_key)
    if _shop_unlocked():
        with tabs[2]:
            _shop_inbox(api_base, api_key)
        with tabs[3]:
            _show_catalog(catalog)
    else:
        with tabs[2]:
            _show_catalog(catalog)


def _default_api_base() -> str:
    """Return API_BASE_URL when one is configured.

    A blank value runs catalog, chat, photos, and the shop inbox in this process.
    """
    return os.getenv("API_BASE_URL", "").strip().rstrip("/")


def _sidebar() -> tuple[str, str, str]:
    """Collect the API address, the app key, and an optional vision key."""
    with st.sidebar:
        if _LOGO_PATH.is_file():
            st.image(str(_LOGO_PATH), width=200)
        st.markdown("**Safeguard 4U**")
        st.caption("AI Estimator & Structural Assistant")
        st.header("Connection")
        if "estimator_api_base" not in st.session_state:
            st.session_state.estimator_api_base = _default_api_base()
        api_base = st.text_input(
            "API base URL",
            key="estimator_api_base",
            help="Leave blank to run in this app. Set API_BASE_URL only for a separate API.",
        )
        api_key = st.text_input(
            "API key",
            type="password",
            help="Required only when APP_API_KEY is set on the API.",
        )
        vision_key = st.text_input(
            "Vision model key",
            type="password",
            key="vision_model_key",
            help=(
                "Optional OpenAI key for this photo. Leave blank to use OPENAI_API_KEY from the API. "
                "Patio, gate, stair, and railing photos use it for the sales note and the generated preview. "
                "The key is sent only with a photo and is not saved."
            ),
        )
        if st.button("Reload catalog", use_container_width=True):
            st.session_state.pop("catalog", None)
            st.session_state.pop("catalog_token", None)
        _shop_access()
    return api_base, api_key, vision_key


def _shop_unlocked() -> bool:
    """Return whether this browser session entered the shop passcode."""
    return bool(st.session_state.get("shop_unlocked"))


def _shop_access() -> None:
    """Ask for the shop passcode without showing any stored leads."""
    st.divider()
    if _shop_unlocked():
        st.caption("Shop access is open in this browser.")
        return
    with st.form("shop_access"):
        code = st.text_input("Shop access", type="password")
        submitted = st.form_submit_button("Open shop")
    if not submitted:
        return
    expected = os.getenv("SHOP_PASSCODE", "").strip()
    entered = code.strip()
    if expected and entered and hmac.compare_digest(entered, expected):
        st.session_state.shop_unlocked = True
        st.rerun()
    st.caption("Shop access stayed closed.")


def _load_catalog(api_base: str, api_key: str) -> tuple[dict[str, Any] | None, str | None]:
    """Return the catalog, using the last copy when the connection has not changed."""
    token = (api_base.strip(), api_key.strip())
    if st.session_state.get("catalog_token") == token and isinstance(st.session_state.get("catalog"), dict):
        _sidebar_status(st.session_state.catalog, None)
        return st.session_state.catalog, None
    try:
        _status, body = _request("GET", api_base, "/catalog", api_key=api_key)
    except ApiError as exc:
        _sidebar_status(None, str(exc))
        return None, str(exc)
    st.session_state.catalog = body
    st.session_state.catalog_token = token
    _sidebar_status(body, None)
    return body, None


def _sidebar_status(catalog: dict[str, Any] | None, error: str | None) -> None:
    """Show the catalog connection in the sidebar."""
    with st.sidebar:
        if error:
            st.error(error)
            return
        profiles = len(catalog.get("profiles", [])) if catalog else 0
        tubes = len(catalog.get("tubes", [])) if catalog else 0
        stairs = len(catalog.get("stairs", [])) if catalog else 0
        railings = len(catalog.get("railings", [])) if catalog else 0
        patios = len(catalog.get("patios", [])) if catalog else 0
        st.success(
            f"Catalog loaded · {profiles} profiles · {tubes} tubes · {stairs} stair styles · "
            f"{railings} railing styles · {patios} outdoor frames"
        )


def _show_catalog(catalog: dict[str, Any]) -> None:
    """Show the stock lists and planning rates returned by the API."""
    profiles = [row for row in catalog.get("profiles", []) if isinstance(row, dict)]
    tubes = [row for row in catalog.get("tubes", []) if isinstance(row, dict)]
    stairs = [row for row in catalog.get("stairs", []) if isinstance(row, dict)]
    railings = [row for row in catalog.get("railings", []) if isinstance(row, dict)]
    mounts = [row for row in catalog.get("railing_mounts", []) if isinstance(row, dict)]
    patios = [row for row in catalog.get("patios", []) if isinstance(row, dict)]
    rates = catalog.get("rates") if isinstance(catalog.get("rates"), dict) else {}
    st.caption("Stock sizes and default planning rates loaded from the Safeguard 4U catalog.")
    _show_parameters(rates)
    _show_profile_table(profiles)
    _show_tube_table(tubes)
    _show_stair_table(stairs)
    _show_railing_table(railings)
    _show_railing_mounts(mounts)
    _show_patio_table(patios)
    _show_addon_table([row for row in catalog.get("addons", []) if isinstance(row, dict)])


def _show_parameters(rates: dict[str, Any]) -> None:
    """Show the estimator's default rates as a short review table."""
    st.subheader("Default parameters")
    st.caption("These rates are what the estimator applies when it builds a bracket.")
    labels = {
        "material_low_usd_per_lb": ("Material low rate", "Low edge of a steel bracket", _dollars_per_lb),
        "material_high_usd_per_lb": ("Material high rate", "High edge before the allowance", _dollars_per_lb),
        "high_side_allowance": ("High-side allowance", "Added only on the high edge", _percent),
        "plate_lb_per_sq_ft_per_inch": (
            "Plate weight factor",
            "Pounds per square foot for each inch of thickness",
            _plate_factor,
        ),
        "stair_default_rise_in": (
            "Default stair rise",
            "Used when a stair takeoff does not set a rise",
            _inches,
        ),
        "stair_default_run_in": (
            "Default stair run",
            "Used for a straight stair when a run is not set",
            _inches,
        ),
        "stair_spiral_degrees_per_step": (
            "Spiral tread sector",
            "Degrees of turn planned for each spiral step",
            _degrees,
        ),
        "patio_post_spacing_ft": (
            "Patio post spacing",
            "Planning distance between posts along the width",
            _feet,
        ),
        "patio_post_height_ft": (
            "Patio post height",
            "Planning height of a pergola, cover, or frame post",
            _feet,
        ),
        "patio_railing_post_spacing_ft": (
            "Railing post spacing",
            "Planning distance between perimeter railing posts",
            _feet,
        ),
        "patio_railing_height_ft": (
            "Perimeter railing height",
            "Planning height of a patio perimeter post",
            _feet,
        ),
        "railing_picket_spacing_in": (
            "Vertical picket spacing",
            "Center spacing of vertical 1/2 in square-bar pickets",
            _inches,
        ),
        "railing_horizontal_spacing_in": (
            "Horizontal bar spacing",
            "Vertical spacing of horizontal picket bars",
            _inches,
        ),
        "railing_ornamental_spacing_in": (
            "Ornamental bar spacing",
            "Center spacing of ornamental square bars",
            _inches,
        ),
        "railing_cable_spacing_in": (
            "Cable spacing",
            "Vertical spacing of horizontal cable runs",
            _inches,
        ),
    }
    rows: list[dict[str, str]] = []
    for key, (name, role, formatter) in labels.items():
        if key not in rates:
            continue
        rows.append({"Parameter": name, "Value": formatter(rates[key]), "Role": role})
    for key, value in rates.items():
        if key in labels or not isinstance(key, str):
            continue
        rows.append({"Parameter": key, "Value": str(value), "Role": "Returned by the catalog"})
    if not rows:
        st.info("The catalog response did not include default parameters.")
        return
    with st.container(border=True):
        st.dataframe(
            rows,
            hide_index=True,
            width="stretch",
            column_config={
                "Parameter": st.column_config.TextColumn("Parameter", width="medium"),
                "Value": st.column_config.TextColumn("Value", width="small"),
                "Role": st.column_config.TextColumn("Role", width="large"),
            },
        )


def _show_profile_table(profiles: list[dict[str, Any]]) -> None:
    """Show every rolled shape, with a family and designation filter."""
    st.subheader(f"Steel profiles · {len(profiles)}")
    st.caption("Nominal weight per foot for each rolled shape.")
    family_col, search_col = st.columns([1, 2])
    with family_col:
        family = st.selectbox(
            "Profile family",
            ["all", *[key for key, _label in _FAMILIES]],
            format_func=lambda key: "All families" if key == "all" else dict(_FAMILIES).get(key, key),
            key="catalog_profile_family",
        )
    with search_col:
        query = st.text_input("Find a profile", placeholder="W12x26", key="catalog_profile_query")
    needle = query.strip().lower()
    rows = []
    for item in profiles:
        designation = str(item.get("designation", ""))
        item_family = str(item.get("family", ""))
        if family != "all" and item_family != family:
            continue
        if needle and needle not in designation.lower():
            continue
        weight = item.get("weight_lb_per_ft")
        rows.append(
            {
                "Designation": designation,
                "Family": dict(_FAMILIES).get(item_family, item_family),
                "Weight (lb/ft)": weight if isinstance(weight, (int, float)) else None,
            }
        )
    _stock_table(
        rows,
        len(profiles),
        "No profiles match that filter.",
        {
            "Designation": st.column_config.TextColumn("Designation", width="medium"),
            "Family": st.column_config.TextColumn("Family", width="medium"),
            "Weight (lb/ft)": st.column_config.NumberColumn("Weight (lb/ft)", format="%.2f"),
        },
    )


def _show_tube_table(tubes: list[dict[str, Any]]) -> None:
    """Show every HSS and pipe size, with a shape and designation filter."""
    st.subheader(f"Tube sizes · {len(tubes)}")
    st.caption("Outside size, wall, and nominal weight per foot for HSS and pipe.")
    shape_col, search_col = st.columns([1, 2])
    with shape_col:
        shape = st.selectbox(
            "Tube shape",
            ["all", *[key for key, _label in _TUBE_SHAPES]],
            format_func=lambda key: "All shapes" if key == "all" else dict(_TUBE_SHAPES).get(key, key),
            key="catalog_tube_shape",
        )
    with search_col:
        query = st.text_input("Find a tube", placeholder="HSS4x4x1/4", key="catalog_tube_query")
    needle = query.strip().lower()
    rows = []
    for item in tubes:
        designation = str(item.get("designation", ""))
        item_shape = str(item.get("shape", ""))
        if shape != "all" and item_shape != shape:
            continue
        if needle and needle not in designation.lower():
            continue
        rows.append(
            {
                "Designation": designation,
                "Shape": dict(_TUBE_SHAPES).get(item_shape, item_shape),
                "Width (in)": _as_float(item.get("outside_width_in")),
                "Height (in)": _as_float(item.get("outside_height_in")),
                "Wall (in)": _as_float(item.get("wall_in")),
                "Schedule": _blank_text(item.get("schedule")),
                "Nominal size": _blank_text(item.get("nominal_size")),
                "Weight (lb/ft)": _as_float(item.get("weight_lb_per_ft")),
            }
        )
    _stock_table(
        rows,
        len(tubes),
        "No tube sizes match that filter.",
        {
            "Designation": st.column_config.TextColumn("Designation", width="medium"),
            "Shape": st.column_config.TextColumn("Shape", width="small"),
            "Width (in)": st.column_config.NumberColumn("Width (in)", format="%.3f"),
            "Height (in)": st.column_config.NumberColumn("Height (in)", format="%.3f"),
            "Wall (in)": st.column_config.NumberColumn("Wall (in)", format="%.3f"),
            "Schedule": st.column_config.TextColumn("Schedule", width="small"),
            "Nominal size": st.column_config.TextColumn("Nominal size", width="small"),
            "Weight (lb/ft)": st.column_config.NumberColumn("Weight (lb/ft)", format="%.2f"),
        },
    )


def _show_stair_table(stairs: list[dict[str, Any]]) -> None:
    """Show every stair style and the weight factors used to price it."""
    st.subheader(f"Stair styles · {len(stairs)}")
    st.caption(
        "Planning weight and shop band for a primed flight. "
        "Straight treads use a 12 in depth. A spiral width is the overall diameter."
    )
    rows = []
    for item in stairs:
        layout = str(item.get("layout", ""))
        extra = _as_float(item.get("stringer_lb_per_ft_per_in")) or 0.0
        base = _as_float(item.get("stringer_lb_per_ft"))
        if base is None:
            stringer = ""
        elif extra:
            stringer = f"{base:.2f} lb/ft + {extra:.2f} lb/ft per in of width"
        else:
            stringer = f"{base:.2f} lb/ft"
        riser = _as_float(item.get("riser_lb_per_sq_ft")) or 0.0
        low = _as_float(item.get("minimum_low_usd"))
        high = _as_float(item.get("minimum_high_usd"))
        minimum = ""
        if low is not None and high is not None:
            minimum = f"${low:,.0f}–${high:,.0f}"
        rows.append(
            {
                "Style": str(item.get("name", "")),
                "Layout": _STAIR_LAYOUTS.get(layout, layout),
                "Stringers": item.get("stringer_count"),
                "Stringer steel": stringer,
                "Tread (lb/ft²)": _as_float(item.get("tread_lb_per_sq_ft")),
                "Riser (lb/ft²)": riser if riser else None,
                "Connector (lb/step)": _as_float(item.get("connector_lb_per_step")),
                "Low ($/lb)": _as_float(item.get("low_usd_per_lb")),
                "High ($/lb)": _as_float(item.get("high_usd_per_lb")),
                "Shop minimum": minimum,
            }
        )
    if not rows:
        st.info("The catalog response did not include stair styles.")
        return
    st.dataframe(
        rows,
        hide_index=True,
        width="stretch",
        column_config={
            "Style": st.column_config.TextColumn("Style", width="medium"),
            "Layout": st.column_config.TextColumn("Layout", width="small"),
            "Stringers": st.column_config.NumberColumn("Stringers", format="%d"),
            "Stringer steel": st.column_config.TextColumn("Stringer steel", width="large"),
            "Tread (lb/ft²)": st.column_config.NumberColumn("Tread (lb/ft²)", format="%.2f"),
            "Riser (lb/ft²)": st.column_config.NumberColumn("Riser (lb/ft²)", format="%.2f"),
            "Connector (lb/step)": st.column_config.NumberColumn("Connector (lb/step)", format="%.1f"),
            "Low ($/lb)": st.column_config.NumberColumn("Low ($/lb)", format="%.2f"),
            "High ($/lb)": st.column_config.NumberColumn("High ($/lb)", format="%.2f"),
            "Shop minimum": st.column_config.TextColumn("Shop minimum", width="medium"),
        },
    )


def _show_railing_table(railings: list[dict[str, Any]]) -> None:
    """Show every railing style and the steel used to price it."""
    st.subheader(f"Railings and balustrades · {len(railings)}")
    st.caption(
        "Posts include both ends of the run. Vertical, horizontal, and ornamental infill uses 1/2 in square bar. "
        "Cable styles add an end fitting on each cable. The selected mount is added once per post."
    )
    rows = []
    for item in railings:
        infill = str(item.get("infill", ""))
        low = _as_float(item.get("minimum_low_usd"))
        high = _as_float(item.get("minimum_high_usd"))
        minimum = ""
        if low is not None and high is not None:
            minimum = f"${low:,.0f}–${high:,.0f}"
        spacing = _as_float(item.get("post_spacing_ft"))
        height = _as_float(item.get("default_height_ft"))
        rows.append(
            {
                "Style": str(item.get("name", "")),
                "Posts": str(item.get("post", "")),
                "Top rail": str(item.get("top_rail", "")),
                "Bottom rail": str(item.get("bottom_rail", "")),
                "Infill": _RAILING_INFILL.get(infill, infill),
                "Post spacing": f"{spacing:g} ft" if spacing is not None else "",
                "Default height": f"{height:g} ft" if height is not None else "",
                "Low ($/lb)": _as_float(item.get("low_usd_per_lb")),
                "High ($/lb)": _as_float(item.get("high_usd_per_lb")),
                "Shop minimum": minimum,
            }
        )
    if not rows:
        st.info("The catalog response did not include railing styles.")
        return
    st.dataframe(
        rows,
        hide_index=True,
        width="stretch",
        column_config={
            "Style": st.column_config.TextColumn("Style", width="medium"),
            "Posts": st.column_config.TextColumn("Posts", width="small"),
            "Top rail": st.column_config.TextColumn("Top rail", width="small"),
            "Bottom rail": st.column_config.TextColumn("Bottom rail", width="small"),
            "Infill": st.column_config.TextColumn("Infill", width="medium"),
            "Post spacing": st.column_config.TextColumn("Post spacing", width="small"),
            "Default height": st.column_config.TextColumn("Default height", width="small"),
            "Low ($/lb)": st.column_config.NumberColumn("Low ($/lb)", format="%.2f"),
            "High ($/lb)": st.column_config.NumberColumn("High ($/lb)", format="%.2f"),
            "Shop minimum": st.column_config.TextColumn("Shop minimum", width="medium"),
        },
    )


def _show_railing_mounts(mounts: list[dict[str, Any]]) -> None:
    """Show the hardware choices that can be added to a railing run."""
    st.subheader(f"Railing mounting hardware · {len(mounts)}")
    st.caption("One piece at each post. Anchors and core drilling stay off the railing bracket.")
    rows = []
    for item in mounts:
        kind = str(item.get("kind", ""))
        if kind == "plate":
            width = _as_float(item.get("width_in"))
            length = _as_float(item.get("length_in"))
            stock = f"{item.get('stock', '')} in plate"
            if width is not None and length is not None:
                stock = f"{stock}, {width:g} in x {length:g} in"
        else:
            length_ft = _as_float(item.get("length_ft"))
            stock = str(item.get("stock", ""))
            if length_ft is not None:
                stock = f"{stock}, {length_ft:g} ft"
        rows.append(
            {
                "Hardware": str(item.get("name", "")),
                "Stock": stock,
                "What stays separate": str(item.get("description", "")),
            }
        )
    if not rows:
        st.info("The catalog response did not include railing hardware.")
        return
    st.dataframe(
        rows,
        hide_index=True,
        width="stretch",
        column_config={
            "Hardware": st.column_config.TextColumn("Hardware", width="medium"),
            "Stock": st.column_config.TextColumn("Stock", width="medium"),
            "What stays separate": st.column_config.TextColumn("What stays separate", width="large"),
        },
    )


def _show_patio_table(patios: list[dict[str, Any]]) -> None:
    """Show every outdoor structure and the shop band used to price it."""
    st.subheader(f"Patio and outdoor structures · {len(patios)}")
    st.caption(
        "Profiles step up with the span. A pergola spaces rafters at 4 ft. "
        "A patio cover spaces them at 2 ft and adds purlins. Span over 16 ft adds a center beam line."
    )
    rows = []
    for item in patios:
        kind = str(item.get("kind", ""))
        low = _as_float(item.get("minimum_low_usd"))
        high = _as_float(item.get("minimum_high_usd"))
        minimum = ""
        if low is not None and high is not None:
            minimum = f"${low:,.0f}–${high:,.0f}"
        rows.append(
            {
                "Structure": str(item.get("name", "")),
                "Steel in the frame": _PATIO_KINDS.get(kind, kind),
                "Low ($/lb)": _as_float(item.get("low_usd_per_lb")),
                "High ($/lb)": _as_float(item.get("high_usd_per_lb")),
                "Shop minimum": minimum,
            }
        )
    if not rows:
        st.info("The catalog response did not include outdoor structures.")
        return
    st.dataframe(
        rows,
        hide_index=True,
        width="stretch",
        column_config={
            "Structure": st.column_config.TextColumn("Structure", width="medium"),
            "Steel in the frame": st.column_config.TextColumn("Steel in the frame", width="large"),
            "Low ($/lb)": st.column_config.NumberColumn("Low ($/lb)", format="%.2f"),
            "High ($/lb)": st.column_config.NumberColumn("High ($/lb)", format="%.2f"),
            "Shop minimum": st.column_config.TextColumn("Shop minimum", width="medium"),
        },
    )


def _show_addon_table(addons: list[dict[str, Any]]) -> None:
    """Show the upgrade rates the estimator can suggest."""
    st.subheader(f"Upgrades · {len(addons)}")
    st.caption("Suggested on their own lines. They are not inside the base bracket.")
    labels = {
        "gate": "Gate",
        "stair": "Stairs",
        "patio": "Patio",
        "railing": "Railings & Balustrades",
        "structural": "Structural",
    }
    rows = []
    for item in addons:
        families = item.get("families") if isinstance(item.get("families"), list) else []
        method = str(item.get("method", ""))
        low = _as_float(item.get("low_rate"))
        high = _as_float(item.get("high_rate"))
        if method == "per_lb" and low is not None and high is not None:
            priced = f"${low:,.2f}–${high:,.2f} / lb"
        elif method == "added_steel" and low is not None and high is not None:
            priced = f"{low:.0%}–{high:.0%} of the base bracket"
        else:
            priced = method
        min_low = _as_float(item.get("minimum_low_usd"))
        min_high = _as_float(item.get("minimum_high_usd"))
        minimum = ""
        if min_low and min_high:
            minimum = f"${min_low:,.0f}–${min_high:,.0f}"
        rows.append(
            {
                "Upgrade": str(item.get("name", "")),
                "Projects": ", ".join(labels.get(str(name), str(name)) for name in families),
                "How it is priced": priced,
                "Shop minimum": minimum,
            }
        )
    if not rows:
        st.info("The catalog response did not include upgrades.")
        return
    st.dataframe(
        rows,
        hide_index=True,
        width="stretch",
        column_config={
            "Upgrade": st.column_config.TextColumn("Upgrade", width="medium"),
            "Projects": st.column_config.TextColumn("Projects", width="large"),
            "How it is priced": st.column_config.TextColumn("How it is priced", width="medium"),
            "Shop minimum": st.column_config.TextColumn("Shop minimum", width="small"),
        },
    )


def _stock_table(
    rows: list[dict[str, Any]],
    total: int,
    empty_message: str,
    column_config: dict[str, Any],
) -> None:
    """Render one filtered stock table."""
    st.caption(f"Showing {len(rows)} of {total}")
    if not rows:
        st.info(empty_message)
        return
    st.dataframe(
        rows,
        hide_index=True,
        width="stretch",
        height=460,
        column_config=column_config,
    )


def _as_float(value: Any) -> float | None:
    """Return a number for a table cell, or nothing when the API omitted it."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _blank_text(value: Any) -> str:
    """Return a display string, with missing catalog fields left blank."""
    if value is None:
        return ""
    return str(value)


def _dollars_per_lb(value: Any) -> str:
    """Format a rate as dollars per pound."""
    if not isinstance(value, (int, float)):
        return str(value)
    return f"${value:,.2f} / lb"


def _percent(value: Any) -> str:
    """Format an allowance fraction as a percent."""
    if not isinstance(value, (int, float)):
        return str(value)
    return f"{value:.0%}"


def _plate_factor(value: Any) -> str:
    """Format the plate weight factor."""
    if not isinstance(value, (int, float)):
        return str(value)
    return f"{value:.2f} lb/ft² per in"


def _inches(value: Any) -> str:
    """Format a stair dimension in inches."""
    if not isinstance(value, (int, float)):
        return str(value)
    return f"{value:g} in"


def _degrees(value: Any) -> str:
    """Format a spiral sector in degrees."""
    if not isinstance(value, (int, float)):
        return str(value)
    return f"{value:g}°"


def _feet(value: Any) -> str:
    """Format a planning length in feet."""
    if not isinstance(value, (int, float)):
        return str(value)
    return f"{value:g} ft"


def _practice_notes() -> None:
    """Show the standing rules that apply to every bracket."""
    with st.container(border=True):
        st.markdown("**How to use a bracket**")
        st.markdown(
            "\n".join(
                [
                    "- Carry the low and the high on the estimate. The bracket is the price range.",
                    "- A profile, tube, plate, or weight line is cut steel. Fabrication, coatings, tax, and freight stay separate.",
                    "- A gate line is a primed shop leaf with standard hangers. The operator, footings, and galvanizing stay separate.",
                    "- A stair line is a primed shop flight. Handrail, guard, landings, and installation stay separate.",
                    "- A railing line is a primed run with posts, rails, infill, and mounting plates. Anchors and installation stay separate.",
                    "- A patio line is a primed outdoor frame. Roof panels, footings, flashing, and installation stay separate.",
                ]
            )
        )


def _takeoff_form(catalog: dict[str, Any]) -> dict[str, Any]:
    """Collect one takeoff and return the estimate request body."""
    material = st.selectbox("Material type", _MATERIAL_TYPES)
    quantity = int(st.number_input("Quantity", min_value=1, max_value=10000, value=1, step=1))
    if material == "Rolled profile":
        return _profile_fields(catalog, quantity)
    if material == "Tube or pipe":
        return _tube_fields(catalog, quantity)
    if material == "Plate":
        return _plate_fields(catalog, quantity)
    if material == "Gate":
        return _gate_fields(catalog, quantity)
    if material == "Stairs":
        return _stair_fields(catalog, quantity)
    if material == "Railings & Balustrades":
        return _railing_fields(catalog, quantity)
    if material == "Patio & Outdoor Structures":
        return _patio_fields(catalog, quantity)
    weight = st.number_input("Weight (lb)", min_value=0.01, max_value=1_000_000.0, value=100.0, step=1.0)
    st.caption("Use this when the steel is already weighed and no stock size is selected.")
    return {"basis": "weight", "weight_lb": float(weight), "quantity": quantity}


def _profile_fields(catalog: dict[str, Any], quantity: int) -> dict[str, Any]:
    """Select a rolled shape and a length."""
    family_key = st.selectbox(
        "Family",
        [key for key, _label in _FAMILIES],
        format_func=lambda key: dict(_FAMILIES)[key],
    )
    rows = [row for row in catalog.get("profiles", []) if row.get("family") == family_key]
    designation = _member_select("Profile", rows, preferred="W12x26")
    length_ft = st.number_input("Length (ft)", min_value=0.1, max_value=1000.0, value=20.0, step=0.5)
    _member_caption(rows, designation, length_ft, quantity)
    return {
        "basis": "profile",
        "designation": designation,
        "length_ft": float(length_ft),
        "quantity": quantity,
    }


def _tube_fields(catalog: dict[str, Any], quantity: int) -> dict[str, Any]:
    """Select an HSS or pipe size and a length."""
    shape = st.selectbox(
        "Tube shape",
        [key for key, _label in _TUBE_SHAPES],
        format_func=lambda key: dict(_TUBE_SHAPES)[key],
    )
    rows = [row for row in catalog.get("tubes", []) if row.get("shape") == shape]
    preferred = "PIPE 4 SCH 40" if shape == "pipe" else "HSS4x4x1/4"
    designation = _member_select("Size", rows, preferred=preferred)
    length_ft = st.number_input("Length (ft)", min_value=0.1, max_value=1000.0, value=10.0, step=0.5)
    _member_caption(rows, designation, length_ft, quantity)
    return {
        "basis": "tube",
        "designation": designation,
        "length_ft": float(length_ft),
        "quantity": quantity,
    }


def _plate_fields(catalog: dict[str, Any], quantity: int) -> dict[str, Any]:
    """Select a plate thickness and the blank size."""
    plates = catalog.get("plates", [])
    names = [row["name"] for row in plates]
    index = names.index("1/2") if "1/2" in names else 0
    name = st.selectbox("Plate thickness", names, index=index)
    selected = next(row for row in plates if row["name"] == name)
    width_in, length_in = st.columns(2)
    with width_in:
        width = st.number_input("Width (in)", min_value=0.1, max_value=480.0, value=48.0, step=1.0)
    with length_in:
        length = st.number_input("Length (in)", min_value=0.1, max_value=480.0, value=96.0, step=1.0)
    area = (float(width) / 12.0) * (float(length) / 12.0) * quantity
    weight = area * float(selected["weight_lb_per_sq_ft"])
    st.caption(f"{name} · {selected['weight_lb_per_sq_ft']:.2f} lb/ft² · about {weight:,.1f} lb at this size")
    return {
        "basis": "plate",
        "plate": name,
        "width_in": float(width),
        "length_in": float(length),
        "quantity": quantity,
    }


def _gate_fields(catalog: dict[str, Any], quantity: int) -> dict[str, Any]:
    """Select a gate style and the clear opening."""
    gates = catalog.get("gates", [])
    labels = {row["key"]: row["name"] for row in gates}
    keys = list(labels)
    index = keys.index("cantilever_slide") if "cantilever_slide" in keys else 0
    key = st.selectbox("Gate style", keys, index=index, format_func=lambda item: labels[item])
    selected = next(row for row in gates if row["key"] == key)
    st.caption(selected.get("description", ""))
    width_col, height_col = st.columns(2)
    with width_col:
        width = st.number_input("Clear opening width (ft)", min_value=0.5, max_value=120.0, value=12.0, step=0.5)
    with height_col:
        height = st.number_input("Clear opening height (ft)", min_value=0.5, max_value=120.0, value=6.0, step=0.5)
    low = selected.get("low_usd_per_sq_ft")
    high = selected.get("high_usd_per_sq_ft")
    if low is not None and high is not None:
        st.caption(f"Planning band ${low:,.0f}–${high:,.0f} per square foot of clear opening, before the shop minimum.")
    return {
        "basis": "gate",
        "gate_style": key,
        "opening_width_ft": float(width),
        "opening_height_ft": float(height),
        "quantity": quantity,
    }


def _stair_fields(catalog: dict[str, Any], quantity: int) -> dict[str, Any]:
    """Select a stair type, the step count, and the width."""
    stairs = [row for row in catalog.get("stairs", []) if isinstance(row, dict) and row.get("key")]
    if not stairs:
        st.warning("The catalog has no stair styles. Reload the catalog after the API is updated.")
        return {"basis": "stair", "stair_style": "", "step_count": 14, "stair_width_in": 42.0, "quantity": quantity}
    labels = {str(row["key"]): str(row.get("name", row["key"])) for row in stairs}
    keys = list(labels)
    index = keys.index("channel_stringer") if "channel_stringer" in keys else 0
    key = st.selectbox("Stair type", keys, index=index, format_func=lambda item: labels[item])
    selected = next(row for row in stairs if row["key"] == key)
    st.caption(str(selected.get("description", "")))
    layout = str(selected.get("layout", "straight"))
    steps = int(st.number_input("Number of steps", min_value=2, max_value=36, value=14, step=1))
    if layout == "spiral":
        width = st.number_input(
            "Overall diameter (in)",
            min_value=48.0,
            max_value=96.0,
            value=60.0,
            step=1.0,
            key="stair_width_spiral",
        )
    else:
        width = st.number_input(
            "Width (in)",
            min_value=24.0,
            max_value=120.0,
            value=42.0,
            step=1.0,
            key="stair_width_straight",
        )
    rise_default = _rate_default(catalog, "stair_default_rise_in", 7.0)
    run_default = _rate_default(catalog, "stair_default_run_in", 11.0)
    rise_col, run_col = st.columns(2)
    with rise_col:
        rise = st.number_input("Rise (in)", min_value=5.0, max_value=9.0, value=rise_default, step=0.25)
    run = run_default
    with run_col:
        if layout == "spiral":
            sector = _rate_default(catalog, "stair_spiral_degrees_per_step", 30.0)
            st.caption(f"Each step is a {sector:g}° sector. The run does not apply.")
        else:
            run = st.number_input("Run (in)", min_value=9.0, max_value=14.0, value=run_default, step=0.25)
    low = selected.get("low_usd_per_lb")
    high = selected.get("high_usd_per_lb")
    if isinstance(low, (int, float)) and isinstance(high, (int, float)):
        st.caption(f"Shop band ${low:,.2f}–${high:,.2f} per pound of stair steel, before the shop minimum.")
    payload: dict[str, Any] = {
        "basis": "stair",
        "stair_style": key,
        "step_count": steps,
        "stair_width_in": float(width),
        "rise_in": float(rise),
        "quantity": quantity,
    }
    if layout != "spiral":
        payload["run_in"] = float(run)
    return payload


def _railing_fields(catalog: dict[str, Any], quantity: int) -> dict[str, Any]:
    """Select a railing style, the linear footage, and the height."""
    railings = [row for row in catalog.get("railings", []) if isinstance(row, dict) and row.get("key")]
    if not railings:
        st.warning("The catalog has no railing styles. Reload the catalog after the API is updated.")
        return {
            "basis": "railing",
            "railing_style": "",
            "railing_mount": "",
            "railing_length_ft": 20.0,
            "railing_height_ft": 3.5,
            "quantity": quantity,
        }
    labels = {str(row["key"]): str(row.get("name", row["key"])) for row in railings}
    keys = list(labels)
    index = keys.index("vertical_picket") if "vertical_picket" in keys else 0
    key = st.selectbox("Railing style", keys, index=index, format_func=lambda item: labels[item])
    selected = next(row for row in railings if row["key"] == key)
    st.caption(str(selected.get("description", "")))
    length_col, height_col = st.columns(2)
    with length_col:
        length = st.number_input("Linear footage (ft)", min_value=4.0, max_value=300.0, value=20.0, step=1.0)
    default_height = _as_float(selected.get("default_height_ft")) or 3.5
    with height_col:
        height = st.number_input(
            "Height (ft)",
            min_value=2.5,
            max_value=5.0,
            value=default_height,
            step=0.5,
            key=f"railing_height_{key}",
        )
    spacing = _as_float(selected.get("post_spacing_ft"))
    posts = str(selected.get("post", ""))
    top = str(selected.get("top_rail", ""))
    bottom = str(selected.get("bottom_rail", ""))
    mounts = [row for row in catalog.get("railing_mounts", []) if isinstance(row, dict) and row.get("key")]
    mount_labels = {str(row["key"]): str(row.get("name", row["key"])) for row in mounts}
    mount_keys = list(mount_labels)
    mount_index = mount_keys.index("base_plate") if "base_plate" in mount_keys else 0
    mount_key = ""
    if mount_keys:
        mount_key = st.selectbox(
            "Mounting hardware",
            mount_keys,
            index=mount_index,
            format_func=lambda item: mount_labels[item],
        )
        chosen_mount = next(row for row in mounts if row["key"] == mount_key)
        st.caption(str(chosen_mount.get("description", "")))
    if spacing is not None:
        st.caption(f"Posts {posts} at {spacing:g} ft. Top rail {top}. Bottom rail {bottom}.")
    low = selected.get("low_usd_per_lb")
    high = selected.get("high_usd_per_lb")
    if isinstance(low, (int, float)) and isinstance(high, (int, float)):
        st.caption(f"Shop band ${low:,.2f}–${high:,.2f} per pound of railing steel, before the shop minimum.")
    return {
        "basis": "railing",
        "railing_style": key,
        "railing_mount": mount_key,
        "railing_length_ft": float(length),
        "railing_height_ft": float(height),
        "quantity": quantity,
    }


def _patio_fields(catalog: dict[str, Any], quantity: int) -> dict[str, Any]:
    """Select an outdoor structure, the footprint, and the span."""
    patios = [row for row in catalog.get("patios", []) if isinstance(row, dict) and row.get("key")]
    if not patios:
        st.warning("The catalog has no outdoor structures. Reload the catalog after the API is updated.")
        return {"basis": "patio", "patio_style": "", "area_sq_ft": 240.0, "span_ft": 12.0, "quantity": quantity}
    labels = {str(row["key"]): str(row.get("name", row["key"])) for row in patios}
    keys = list(labels)
    index = keys.index("pergola") if "pergola" in keys else 0
    key = st.selectbox("Structure", keys, index=index, format_func=lambda item: labels[item])
    selected = next(row for row in patios if row["key"] == key)
    st.caption(str(selected.get("description", "")))
    area_col, span_col = st.columns(2)
    with area_col:
        area = st.number_input("Area (sq ft)", min_value=64.0, max_value=1500.0, value=240.0, step=10.0)
    with span_col:
        span = st.number_input("Span (ft)", min_value=8.0, max_value=24.0, value=12.0, step=1.0)
    width = float(area) / float(span) if float(span) else 0.0
    if 8.0 <= width <= 60.0:
        st.caption(f"{float(span):g} ft span by {width:g} ft wide · {float(area):g} sq ft")
    else:
        st.caption(
            f"That span and area make a {width:g} ft width. "
            "Keep the width between 8 and 60 ft before building a range."
        )
    low = selected.get("low_usd_per_lb")
    high = selected.get("high_usd_per_lb")
    if isinstance(low, (int, float)) and isinstance(high, (int, float)):
        st.caption(f"Shop band ${low:,.2f}–${high:,.2f} per pound of frame steel, before the shop minimum.")
    return {
        "basis": "patio",
        "patio_style": key,
        "area_sq_ft": float(area),
        "span_ft": float(span),
        "quantity": quantity,
    }


def _rate_default(catalog: dict[str, Any], key: str, fallback: float) -> float:
    """Return a catalog rate for a number input, or the local fallback."""
    rates = catalog.get("rates")
    if not isinstance(rates, dict):
        return fallback
    value = rates.get(key, fallback)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return fallback
    return float(value)


def _member_select(label: str, rows: list[dict[str, Any]], preferred: str) -> str:
    """Return the designation chosen from a weight-labeled list."""
    designations = [str(row["designation"]) for row in rows]
    if not designations:
        st.warning(f"The catalog has no entries for this {label.lower()}.")
        return ""
    index = designations.index(preferred) if preferred in designations else 0

    def _label(designation: str) -> str:
        row = next(item for item in rows if item["designation"] == designation)
        return f"{designation}  ·  {row['weight_lb_per_ft']:g} lb/ft"

    return str(st.selectbox(label, designations, index=index, format_func=_label))


def _member_caption(rows: list[dict[str, Any]], designation: str, length_ft: float, quantity: int) -> None:
    """Show the catalog weight for the selected length."""
    row = next((item for item in rows if item["designation"] == designation), None)
    if row is None:
        return
    total = float(row["weight_lb_per_ft"]) * float(length_ft) * quantity
    detail = f"{designation} · {row['weight_lb_per_ft']:g} lb/ft · about {total:,.1f} lb"
    wall = row.get("wall_in")
    if wall:
        detail += f" · {wall:g} in wall"
    st.caption(detail)


def _submit(api_base: str, api_key: str, payload: dict[str, Any], signature: str) -> None:
    """Request a bracket and store it for the result panel."""
    if payload.get("basis") in {"profile", "tube"} and not payload.get("designation"):
        st.error("Choose a stock size before building a price range.")
        return
    if payload.get("basis") == "stair" and not payload.get("stair_style"):
        st.error("Choose a stair type before building a price range.")
        return
    if payload.get("basis") == "patio" and not payload.get("patio_style"):
        st.error("Choose a structure before building a price range.")
        return
    if payload.get("basis") == "railing" and not payload.get("railing_style"):
        st.error("Choose a railing style before building a price range.")
        return
    if payload.get("basis") == "railing" and not payload.get("railing_mount"):
        st.error("Choose mounting hardware before building a price range.")
        return
    try:
        with st.spinner("Building the price range…"):
            _status, body = _request("POST", api_base, "/estimate", api_key=api_key, payload=payload)
    except ApiError as exc:
        st.error(str(exc))
        return
    st.session_state.estimate = body
    st.session_state.estimate_key = signature


def _show_saved_result(signature: str) -> None:
    """Show the bracket when it still matches the form."""
    if st.session_state.get("estimate_key") != signature:
        st.info("Build a price range to see the bracket and the recommendations for this takeoff.")
        return
    result = st.session_state.get("estimate")
    if not isinstance(result, dict):
        return
    range_col, advice_col = st.columns([3, 2], gap="large")
    with range_col:
        _show_bracket(result)
    with advice_col:
        _show_recommendations(result.get("recommendations", []))


def _show_bracket(result: dict[str, Any]) -> None:
    """Show the low and high edges without a single price."""
    with st.container(border=True):
        st.caption("Price range")
        st.header(str(result.get("range", "")))
        low, high, weight = st.columns(3)
        low.metric("Low", _dollars(result.get("low_usd")))
        high.metric("High", _dollars(result.get("high_usd")))
        weight_lb = result.get("weight_lb")
        weight.metric("Weight", f"{weight_lb:,.1f} lb" if isinstance(weight_lb, (int, float)) else "Shop leaf")
        st.write(str(result.get("summary", "")))
        st.caption("Use the low and the high together. A later firm quote should land inside this range.")


def _show_addons(
    catalog: dict[str, Any],
    payload: dict[str, Any],
    signature: str,
    api_base: str,
    api_key: str,
    vision_key: str,
) -> None:
    """Suggest upgrades for the selected project and read property photos."""
    family = _project_family(str(payload.get("basis", "")))
    labels = {
        "gate": "Gate",
        "stair": "Stairs",
        "patio": "Patio",
        "railing": "Railings & Balustrades",
        "structural": "Structural",
    }
    priced = _priced_addons(signature)
    st.subheader("Smart Recommendations & Add-ons")
    st.caption(
        f"Suggestions for a {labels[family]} project. "
        "Each upgrade stays on its own line and is not inside the price range above."
    )
    offer_col, photo_col = st.columns([3, 2], gap="large")
    with offer_col:
        if priced:
            for item in priced:
                _show_addon_card(item)
        else:
            for item in _catalog_addons(catalog, family):
                with st.container(border=True):
                    st.markdown(f"**{item.get('name', 'Upgrade')}**")
                    details = item.get("details") if isinstance(item.get("details"), dict) else {}
                    st.write(str(details.get(family, "")))
            st.caption("Build a price range to price these add-ons.")
    with photo_col:
        _property_photos(api_base, api_key, _default_target(payload), vision_key)


def _priced_addons(signature: str) -> list[dict[str, Any]]:
    """Return priced add-ons when they still match the form."""
    if st.session_state.get("estimate_key") != signature:
        return []
    result = st.session_state.get("estimate")
    if not isinstance(result, dict):
        return []
    addons = result.get("addons")
    if not isinstance(addons, list):
        return []
    return [item for item in addons if isinstance(item, dict)]


def _catalog_addons(catalog: dict[str, Any], family: str) -> list[dict[str, Any]]:
    """Return catalog upgrades that apply to this project type."""
    rows = []
    for item in catalog.get("addons", []):
        if not isinstance(item, dict):
            continue
        families = item.get("families")
        if isinstance(families, list) and family in families:
            rows.append(item)
    order = {"hot_dip_galvanizing": 0, "heavy_gauge": 1, "marine_powder": 2}
    if family == "gate":
        order = {"hot_dip_galvanizing": 0, "marine_powder": 1, "heavy_gauge": 2}
    elif family == "structural":
        order = {"heavy_gauge": 0, "hot_dip_galvanizing": 1, "marine_powder": 2}
    return sorted(rows, key=lambda item: order.get(str(item.get("key")), 9))


def _show_addon_card(item: dict[str, Any]) -> None:
    """Show one priced upgrade."""
    with st.container(border=True):
        st.markdown(f"**{item.get('name', 'Upgrade')}**")
        if item.get("range"):
            st.write(str(item["range"]))
        st.write(str(item.get("detail", "")))


def _property_photos(api_base: str, api_key: str, default_target: str, vision_key: str) -> None:
    """Upload site photos and show a vision reading under each one."""
    labels = [label for label, _code in _TARGET_PROJECTS]
    if st.session_state.get("target_project_type") not in labels:
        st.session_state.target_project_type = default_target
    with st.container(border=True):
        st.markdown("**Project property photos**")
        target_label = st.selectbox(
            "Target Project Type",
            labels,
            key="target_project_type",
            help="The sales note and Visual Output Preview follow this project.",
        )
        family = dict(_TARGET_PROJECTS)[str(target_label)]
        photos = st.file_uploader(
            "Property photos",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            help=(
                "Each photo is sent for a sales note on the Target Project Type. "
                "Patio, gate, stair, and railing photos include a generated Visual Output Preview of the steel on this photo. Photos are not priced."
            ),
            key="property_photos",
            label_visibility="collapsed",
        )
        if not photos:
            st.caption(
                "Drop property photos here. The sales note follows the Target Project Type. "
                "Patio, gate, stair, and railing photos include a generated Visual Output Preview of the steel on this photo. Photos are not priced."
            )
            return
        shown = list(photos[:4])
        extra = len(photos) - len(shown)
        st.caption(
            f"{len(shown)} photo{'s' if len(shown) != 1 else ''} sent for a site reading. "
            f"The reading follows {target_label} and is not priced."
        )
        if vision_key.strip():
            st.caption("Using the Vision model key from the sidebar.")
        else:
            st.caption("No sidebar key. The server uses OPENAI_API_KEY for the sales note and the generated preview.")
        for index, photo in enumerate(shown):
            data = photo.getvalue()
            try:
                photo.seek(0)
            except (AttributeError, OSError):
                pass
            st.image(photo, caption=photo.name)
            _show_photo_analysis(api_base, api_key, photo.name, data, family, vision_key)
            if index == 0 and family in _PREVIEW_FAMILIES:
                _show_mockup(api_base, api_key, photo.name, data, family, vision_key)
                _show_portfolio(family)
            elif index == 0:
                _show_preview_suggestion(family)
        if shown and family in _PREVIEW_FAMILIES and len(shown) > 1:
            st.caption("The Visual Output Preview uses the first photo.")
        if extra:
            st.caption(f"{extra} more photo{'s' if extra != 1 else ''} stayed on the form and were not sent.")


def _show_photo_analysis(
    api_base: str,
    api_key: str,
    filename: str,
    data: bytes,
    family: str,
    vision_key: str,
) -> None:
    """Show the cached site reading for one photo, requesting it on the first view."""
    if not data:
        st.warning(f"{filename} is empty.")
        return
    typed_key = vision_key.strip() or str(st.session_state.get("vision_model_key", "")).strip()
    key_mark = hashlib.sha256(typed_key.encode("utf-8")).hexdigest()[:12]
    token = hashlib.sha256(data).hexdigest() + ":" + family + ":" + key_mark + ":target"
    store = st.session_state.setdefault("photo_analyses", {})
    entry = store.get(token)
    if not isinstance(entry, dict):
        with st.spinner("Preparing the sales note..."):
            try:
                _status, body = _post_photo(api_base, api_key, filename, data, family, typed_key)
            except ApiError as exc:
                entry = {"error": str(exc)}
            else:
                summary = body.get("summary")
                if isinstance(summary, str) and summary.strip():
                    entry = {"summary": summary.strip(), "key_source": str(body.get("key_source", ""))}
                else:
                    entry = {"error": "The vision model returned an empty reading."}
        store[token] = entry
    if entry.get("error"):
        st.warning(str(entry["error"]))
        if st.button("Read again", key=f"reread_{token[:16]}"):
            store.pop(token, None)
            st.rerun()
        return
    st.markdown("**Property analysis**")
    st.markdown(str(entry.get("summary", "")))
    source = _key_source_label(str(entry.get("key_source", "")))
    if source:
        st.caption(source)
    st.caption("Sales note for this project. It is not a site inspection or a price.")


def _portfolio_photos(family: str) -> list[Path]:
    """Return up to four local catalog photos for a patio, gate, stair, or railing."""
    folder = _PORTFOLIO_FOLDERS.get(family)
    if folder is None:
        return []
    directory = _PORTFOLIO_ROOT / folder
    if not directory.is_dir():
        return []
    photos = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in _PORTFOLIO_SUFFIXES
    ]
    return sorted(photos, key=lambda path: path.name.lower())[:4]


@st.dialog("Full size photo", width="large")
def _show_photo_dialog(image: str | bytes, caption: str) -> None:
    """Expand one sample or mockup, with Close above and below the picture."""
    if st.button("Close", type="primary", width="stretch", key="close_expanded_photo"):
        st.rerun()
    st.image(image, caption=caption, width="stretch")
    if st.button("Close", width="stretch", key="close_expanded_photo_bottom"):
        st.rerun()


def _expandable_photo(image: str | bytes, caption: str, key: str) -> None:
    """Show a sample or mockup and offer a full-size view that can be closed."""
    st.image(image, caption=caption)
    if st.button("View full size", key=key):
        _show_photo_dialog(image, caption)


def _show_portfolio(family: str) -> None:
    """Show catalog samples directly under the visual preview."""
    photos = _portfolio_photos(family)
    st.markdown("**Completed work**")
    if len(photos) < 3:
        st.caption("The local catalog needs at least three sample photos for this project type.")
        return
    kind = _PORTFOLIO_KINDS.get(family, "Sample")
    st.caption(
        f"{len(photos)} catalog samples for this project type. "
        "They show finished steel of this kind. They are not a photo of this property and they are not priced."
    )
    columns = st.columns(len(photos))
    for column, path in zip(columns, photos):
        number = "".join(character for character in path.stem if character.isdigit()) or path.stem
        caption = f"{kind} sample {number}"
        with column:
            _expandable_photo(str(path), caption, f"view_sample_{family}_{path.name}")


def _show_preview_suggestion(family: str) -> None:
    """Show the Visual Output Preview as a suggestion when no image is drawn."""
    st.markdown("**Visual Output Preview / Mockup**")
    st.write(_PREVIEW_SUGGESTIONS.get(family, _PREVIEW_SUGGESTIONS["structural"]))
    st.caption("Visualization only. It is not a construction document.")


def _mockup_unavailable(message: str) -> bool:
    """Return whether the image model is missing rather than the edit failing."""
    text = message.lower()
    return "openai image key" in text or "cannot draw" in text


def _show_mockup(
    api_base: str,
    api_key: str,
    filename: str,
    data: bytes,
    family: str,
    vision_key: str,
) -> None:
    """Show a superimposed patio, gate, stair, or railing preview for the first photo."""
    if not data:
        return
    typed_key = vision_key.strip() or str(st.session_state.get("vision_model_key", "")).strip()
    key_mark = hashlib.sha256(typed_key.encode("utf-8")).hexdigest()[:12]
    token = hashlib.sha256(data).hexdigest() + ":" + family + ":" + key_mark + ":mockup"
    store = st.session_state.setdefault("photo_mockups", {})
    entry = store.get(token)
    st.markdown("**Visual Output Preview / Mockup**")
    if not isinstance(entry, dict):
        with st.spinner("Placing the steel structure on the photo..."):
            try:
                _status, body = _post_photo(
                    api_base,
                    api_key,
                    filename,
                    data,
                    family,
                    typed_key,
                    path="/mockup",
                    timeout=MOCKUP_TIMEOUT_SECONDS,
                )
            except ApiError as exc:
                entry = {"error": str(exc)}
            else:
                encoded = body.get("image_base64")
                try:
                    image = base64.b64decode(encoded, validate=True) if isinstance(encoded, str) else b""
                except ValueError:
                    image = b""
                if image.startswith(b"\x89PNG\r\n\x1a\n"):
                    entry = {"image": image, "model": str(body.get("model", ""))}
                else:
                    entry = {"error": "The image model returned an unreadable mockup."}
        store[token] = entry
    if entry.get("error"):
        message = str(entry["error"])
        if _mockup_unavailable(message):
            st.write(_PREVIEW_SUGGESTIONS.get(family, _PREVIEW_SUGGESTIONS["structural"]))
            st.caption(
                "This is a suggested preview. A drawn image needs an OpenAI image key. "
                "A Groq key can read the photo but cannot draw the mockup."
            )
        else:
            st.warning(message)
        if st.button("Create mockup again", key=f"remock_{token[:16]}"):
            store.pop(token, None)
            st.rerun()
        return
    _expandable_photo(
        entry["image"],
        "Proposed steel structure installed on this photo",
        f"view_mockup_{token[:16]}",
    )
    st.caption("Generated with OpenAI from this property photo. Visualization only. It is not a construction document.")


def _post_photo(
    api_base: str,
    api_key: str,
    filename: str,
    data: bytes,
    project_type: str,
    vision_key: str = "",
    path: str = "/analyze-photo",
    timeout: int = PHOTO_TIMEOUT_SECONDS,
) -> tuple[int, dict[str, Any]]:
    """Send one property photo to the vision endpoint.

    Args:
        api_base: Flask origin, without a trailing path.
        api_key: Optional ``APP_API_KEY`` value.
        filename: Original file name.
        data: Raw image bytes.
        project_type: Gate, stair, patio, or structural.
        vision_key: Optional OpenAI or Groq key for this photo only.

    Returns:
        HTTP status and the parsed JSON body.

    Raises:
        ApiError: The API returned an error or could not be reached.
    """
    boundary = uuid.uuid4().hex
    safe_name = "".join(ch for ch in filename if ch not in '"\r\n') or "photo.jpg"
    typed_key = vision_key.strip()
    fields = [("project_type", project_type)]
    if typed_key:
        fields.insert(0, ("vision_key", typed_key))
    project = b"".join(_form_field(boundary, name, value) for name, value in fields)
    header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{safe_name}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode("utf-8")
    ending = f"\r\n--{boundary}--\r\n".encode("utf-8")
    body = project + header + data + ending
    headers = {"X-API-Key": api_key.strip()} if api_key.strip() else {}
    if typed_key:
        headers["X-Vision-Key"] = typed_key
    headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    if _use_local(api_base):
        return _local_photo(filename, data, project_type, vision_key, path)
    url = api_base.strip().rstrip("/") + path
    if not url.startswith("http://") and not url.startswith("https://"):
        raise ApiError("Enter an API base URL that starts with http:// or https://.")
    req = request.Request(url, data=body, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return response.status, _decode_json(response.read())
    except error.HTTPError as exc:
        raise ApiError(_error_message(exc.read())) from exc
    except error.URLError:
        return _local_photo(filename, data, project_type, vision_key, path)
    except TimeoutError as exc:
        raise ApiError("The API took too long to respond.") from exc


def _form_field(boundary: str, name: str, value: str) -> bytes:
    """Encode one text field for a multipart body."""
    return (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
        f"{value}\r\n"
    ).encode("utf-8")


def _key_source_label(source: str) -> str:
    """Describe which key was used, without revealing the key."""
    labels = {
        "sidebar": "Read with the Vision model key from the sidebar.",
        "OPENAI_API_KEY": "Read with OPENAI_API_KEY from the server environment.",
        "GROQ_API_KEY": "Read with GROQ_API_KEY from the server environment.",
        "LLM_API_KEY": "Read with LLM_API_KEY from the server environment.",
    }
    return labels.get(source, "")


def _default_target(payload: dict[str, Any]) -> str:
    """Choose the photo-target label that matches the current takeoff."""
    basis = str(payload.get("basis", ""))
    if basis == "patio":
        return "Patio & Outdoor Structures"
    if basis == "gate":
        return "Custom Gates"
    if basis == "stair":
        return "Stairs & Railings"
    if basis == "railing":
        return "Railings & Balustrades"
    return "Structural Steel"


def _project_family(basis: str) -> str:
    """Map a takeoff basis to the project group used for add-ons."""
    if basis in {"gate", "stair", "patio", "railing"}:
        return basis
    return "structural"


def _client_language() -> str:
    """Return the language selected for the client chat."""
    selected = st.session_state.get("client_language", "en")
    return i18n.normalize_client(selected)


def _client_direction(lang: str) -> None:
    """Use right-to-left layout for Farsi chat bubbles."""
    if not i18n.is_rtl(lang):
        return
    st.markdown(
        """
        <style>
        [data-testid="stChatMessage"] { direction: rtl; text-align: right; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _rtl_text(body: str, rtl: bool) -> None:
    """Show a block of text, right to left when the shop or client is reading Farsi."""
    if not rtl:
        st.write(body)
        return
    safe = html.escape(body).replace("\n", "<br>")
    st.markdown(
        f'<div dir="rtl" style="text-align: right;">{safe}</div>',
        unsafe_allow_html=True,
    )


def _client_chat(api_base: str, api_key: str) -> None:
    """Show the catalog Q&A box and the hand-off to the shop.

    Args:
        api_base: Flask origin, without a trailing path.
        api_key: Optional ``APP_API_KEY`` value.
    """
    lang = st.selectbox(
        i18n.text(_client_language(), "language"),
        i18n.CLIENT_LANGUAGES,
        format_func=lambda code: i18n.LANGUAGE_LABELS[code],
        key="client_language",
    )
    lang = i18n.normalize_client(lang)
    _client_direction(lang)
    st.subheader(i18n.text(lang, "ask_title"))
    if i18n.is_rtl(lang):
        _rtl_text(i18n.text(lang, "banner"), True)
    else:
        st.info(i18n.text(lang, "banner"))
    history: list[dict[str, Any]] = st.session_state.setdefault("client_chat", [])
    shown = [item for item in history if item.get("role") in {"user", "assistant"}]
    earlier = shown[:-5] if len(shown) > 5 else []
    recent = shown[-5:]
    if earlier:
        with st.expander(i18n.text(lang, "earlier", count=len(earlier))):
            for item in earlier:
                _render_chat_turn(item, lang)
    for item in recent:
        _render_chat_turn(item, lang)
    user_turns = sum(1 for item in history if item.get("role") == "user")
    if user_turns >= 8:
        if i18n.is_rtl(lang):
            _rtl_text(i18n.text(lang, "turn_limit"), True)
        else:
            st.info(i18n.text(lang, "turn_limit"))
    else:
        prompt = st.chat_input(
            i18n.text(lang, "chat_placeholder"),
            key="client_chat_input",
            accept_file="multiple",
            file_type=["png", "jpg", "jpeg", "webp"],
        )
        text = ""
        uploads: list[Any] = []
        if isinstance(prompt, str):
            text = prompt.strip()
        elif prompt is not None:
            text = str(getattr(prompt, "text", "") or "").strip()
            uploads = list(getattr(prompt, "files", []) or [])
        if text or uploads:
            try:
                images = [_chat_attachment(upload) for upload in uploads[:4]]
            except ValueError:
                st.error(i18n.text(lang, "photo_unread"))
                return
            prior = [
                {"role": item["role"], "content": item["content"]}
                for item in history
                if item.get("role") in {"user", "assistant"} and isinstance(item.get("content"), str)
            ]
            history.append({"role": "user", "content": text, "images": images})
            request_images = [
                {
                    "media_type": image["media_type"],
                    "data_base64": base64.b64encode(image["data"]).decode("ascii"),
                }
                for image in images
            ]
            try:
                _status, body = _request(
                    "POST",
                    api_base,
                    "/chat",
                    api_key=api_key,
                    payload={
                        "message": text,
                        "history": prior[-16:],
                        "language": lang,
                        "images": request_images,
                    },
                )
                reply = str(body.get("reply", "")).strip() or i18n.text(lang, "no_answer")
                examples = body.get("examples")
                example_keys = [key for key in examples if isinstance(key, str)] if isinstance(examples, list) else []
                facts = body.get("facts") if isinstance(body.get("facts"), dict) else {}
            except ApiError as exc:
                reply = i18n.present_error(str(exc), lang)
                example_keys = []
                facts = {}
            history.append({"role": "assistant", "content": reply, "examples": example_keys, "facts": facts})
            st.rerun()
    if history and st.button(i18n.text(lang, "new_conversation"), key="clear_client_chat"):
        st.session_state.client_chat = []
        st.rerun()
    _send_chat_to_shop(api_base, api_key, history, lang)


def _example_files(folder: str) -> list[Path]:
    """Return the pictures dropped in one example folder."""
    root = _EXAMPLE_ROOT / folder
    if not root.is_dir():
        return []
    files = [path for path in root.iterdir() if path.is_file() and path.suffix.lower() in _PORTFOLIO_SUFFIXES]
    real = [path for path in files if path.stem.lower() != "sample-preview"]
    chosen = real or files
    return sorted(chosen, key=lambda path: path.name.lower())


def _fit_chat_image(data: bytes) -> tuple[bytes, str]:
    """Shrink one chat photo so its longest side is at most 1024 pixels.

    Args:
        data: Raw PNG, JPEG, or WebP bytes.

    Returns:
        Fitted bytes and their media type.

    Raises:
        ValueError: The file is not a readable PNG, JPEG, or WebP.
    """
    from PIL import Image

    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.mode in {"RGBA", "LA", "P"}:
                fitted = image.convert("RGBA")
            else:
                fitted = image.convert("RGB")
            fitted.thumbnail((1024, 1024))
            buffer = io.BytesIO()
            if fitted.mode == "RGBA":
                fitted.save(buffer, format="PNG")
                return buffer.getvalue(), "image/png"
            fitted.save(buffer, format="JPEG", quality=85)
            return buffer.getvalue(), "image/jpeg"
    except (OSError, ValueError) as exc:
        raise ValueError("unread photo") from exc


def _chat_attachment(uploaded: Any) -> dict[str, Any]:
    """Store one chat attachment after fitting it for the model and the shop.

    Args:
        uploaded: A file from the chat input.

    Returns:
        The file name, media type, and fitted bytes.
    """
    data, media_type = _fit_chat_image(uploaded.getvalue())
    name = str(getattr(uploaded, "name", "") or "site-photo")
    return {"name": name, "media_type": media_type, "data": data}


def _render_chat_turn(item: dict[str, Any], lang: str) -> None:
    """Show one chat message, with a sample photo inside the assistant bubble."""
    role = str(item.get("role") or "assistant")
    with st.chat_message(role):
        if role == "user":
            images = item.get("images") if isinstance(item.get("images"), list) else []
            for image in images:
                if isinstance(image, dict) and isinstance(image.get("data"), bytes):
                    st.image(image["data"], width=280)
        content = str(item.get("content", "")).strip()
        if content:
            _rtl_text(content, i18n.is_rtl(lang))
        if role == "assistant":
            _render_bubble_examples(item.get("examples"), lang)


def _render_bubble_examples(keys: Any, lang: str) -> None:
    """Show the sample photo for each category attached to this reply.

    Args:
        keys: Category keys returned by the chat for this turn.
        lang: Client language used for the caption.
    """
    if not isinstance(keys, list):
        return
    folders = {key: folder for key, _label, folder in _STYLE_EXAMPLES}
    for key in keys:
        if not isinstance(key, str) or key not in folders:
            continue
        files = _example_files(folders[key])
        if not files:
            continue
        st.image(files[0].read_bytes(), caption=i18n.caption(lang, key), width=320)


def _send_chat_to_shop(api_base: str, api_key: str, history: list[dict[str, Any]], lang: str) -> None:
    """Let the client send a note, a photo, and any chat questions to the shop.

    Args:
        api_base: Flask origin, without a trailing path.
        api_key: Optional ``APP_API_KEY`` value.
        history: Messages already shown in the chat.
        lang: Client language stored with the request.
    """
    questions = [item["content"] for item in history if item.get("role") == "user" and item.get("content")]
    st.subheader(i18n.text(lang, "send_title"))
    st.caption(i18n.text(lang, "send_caption"))
    with st.form("shop_lead"):
        picked: list[str] = []
        if questions:
            st.caption(i18n.text(lang, "questions_include"))
            for index, question in enumerate(questions):
                if st.checkbox(str(question), value=True, key=f"shop_question_{index}"):
                    picked.append(str(question))
        has_estimate = any(
            item.get("role") == "assistant" and isinstance(item.get("content"), str) and item.get("content")
            for item in history
        )
        include_estimate = False
        if has_estimate:
            include_estimate = st.checkbox(
                i18n.text(lang, "include_estimate"),
                value=True,
                key="shop_include_estimate",
            )
        name = st.text_input(i18n.text(lang, "name"))
        contact = st.text_input(i18n.text(lang, "contact"))
        extra = st.text_area(i18n.text(lang, "note_for_shop"), max_chars=1000)
        submitted = st.form_submit_button(i18n.text(lang, "send_button"))
    if not submitted:
        return
    facts: dict[str, Any] = {}
    estimate_note = ""
    if include_estimate:
        for item in reversed(history):
            if item.get("role") != "assistant":
                continue
            if not estimate_note and isinstance(item.get("content"), str):
                estimate_note = item["content"]
            if not facts and isinstance(item.get("facts"), dict):
                facts = item["facts"]
            if facts and estimate_note:
                break
    payload: dict[str, Any] = {
        "name": name,
        "contact": contact,
        "notes": picked,
        "extra": extra,
        "client_language": lang,
        "shop_language": _shop_language(),
        "facts": facts,
        "estimate_note": estimate_note[:1000],
    }
    chat_photos: list[dict[str, Any]] = []
    for item in history:
        if item.get("role") != "user" or not isinstance(item.get("images"), list):
            continue
        for image in item["images"]:
            if isinstance(image, dict) and isinstance(image.get("data"), bytes):
                chat_photos.append(image)
    if chat_photos:
        payload["photos"] = [
            {
                "filename": str(image.get("name") or "site-photo"),
                "data_base64": base64.b64encode(image["data"]).decode("ascii"),
            }
            for image in chat_photos[:6]
        ]
    if not picked and not str(extra).strip() and not chat_photos:
        st.warning(i18n.text(lang, "need_shop_detail"))
        return
    try:
        _status, body = _request(
            "POST",
            api_base,
            "/lead",
            api_key=api_key,
            payload=payload,
        )
    except ApiError as exc:
        st.error(str(exc))
        return
    st.success(i18n.text(lang, "sent"))


def _shop_language() -> str:
    """Return the language selected for the shop review."""
    return i18n.normalize_shop(st.session_state.get("shop_language", "en"))


def _shop_inbox(api_base: str, api_key: str) -> None:
    """Show stored shop requests after the shop passcode has been entered.

    Args:
        api_base: Flask origin, without a trailing path. Blank runs in this process.
        api_key: Optional ``APP_API_KEY`` value.
    """
    if not _shop_unlocked():
        return
    shop_lang = st.selectbox(
        i18n.text(_shop_language(), "shop_language"),
        i18n.SHOP_LANGUAGES,
        format_func=lambda code: i18n.LANGUAGE_LABELS[code],
        key="shop_language",
    )
    shop_lang = i18n.normalize_shop(shop_lang)
    rtl = i18n.is_rtl(shop_lang)
    st.subheader(i18n.text(shop_lang, "shop_requests"))
    try:
        _status, body = _request("GET", api_base, "/leads", api_key=api_key)
    except ApiError as exc:
        st.caption(str(exc))
        return
    rows = body.get("requests")
    if not isinstance(rows, list) or not rows:
        _rtl_text(i18n.text(shop_lang, "no_requests"), rtl)
        return
    for row in reversed(rows):
        if not isinstance(row, dict):
            continue
        who = str(row.get("name") or i18n.text(shop_lang, "no_name"))
        contact = str(row.get("contact") or i18n.text(shop_lang, "no_contact"))
        header = f"{row.get('reference', '')} · {who} · {contact}"
        _rtl_text(header, rtl)
        facts = row.get("facts") if isinstance(row.get("facts"), dict) else {}
        summary = i18n.shop_summary(shop_lang, facts, str(row.get("client_language") or ""))
        shop_copy = row.get("shop_copy") if isinstance(row.get("shop_copy"), dict) else {}
        side = shop_copy.get(shop_lang) if isinstance(shop_copy.get(shop_lang), dict) else {}
        translated_summary = str(side.get("summary") or "").strip()
        if summary:
            _rtl_text(summary, rtl)
        elif translated_summary:
            _rtl_text(translated_summary, rtl)
        elif str(row.get("estimate_note") or "").strip():
            _rtl_text(str(row.get("estimate_note")), rtl)
        questions = side.get("questions") if isinstance(side.get("questions"), list) else []
        if not questions:
            original_notes = row.get("notes")
            questions = original_notes if isinstance(original_notes, list) else []
        visible_questions = [str(item).strip() for item in questions if str(item).strip()]
        if visible_questions:
            _rtl_text(i18n.text(shop_lang, "customer_questions"), rtl)
            for note in visible_questions:
                _rtl_text(f"- {note}", rtl)
        extra = str(side.get("note") or row.get("extra") or "").strip()
        if extra:
            _rtl_text(i18n.text(shop_lang, "extra_note"), rtl)
            _rtl_text(extra, rtl)
        photo_count = row.get("photo_count")
        if not isinstance(photo_count, int):
            photo_count = 1 if row.get("has_photo") else 0
        reference = str(row.get("reference", ""))
        for index in range(photo_count):
            try:
                image = _get_bytes(api_base, f"/leads/{reference}/photos/{index}", api_key)
                st.image(image, caption=i18n.text(shop_lang, "site_photo"), width=320)
            except ApiError as photo_error:
                st.caption(str(photo_error))
        _shop_text_tools(row, shop_lang)


def _shop_text_tools(row: dict[str, Any], shop_lang: str) -> None:
    """Offer a copyable summary and a text-message link in the shop language.

    Args:
        row: One request returned by the shop list.
        shop_lang: English or Farsi selected for the shop.
    """
    facts = row.get("facts") if isinstance(row.get("facts"), dict) else {}
    shop_copy = row.get("shop_copy") if isinstance(row.get("shop_copy"), dict) else {}
    side = shop_copy.get(shop_lang) if isinstance(shop_copy.get(shop_lang), dict) else {}
    questions = side.get("questions") if isinstance(side.get("questions"), list) else row.get("notes")
    if not isinstance(questions, list):
        questions = []
    note = str(side.get("note") or row.get("extra") or "")
    summary = i18n.shop_alert(
        shop_lang,
        str(row.get("reference") or ""),
        str(row.get("name") or ""),
        str(row.get("contact") or ""),
        str(row.get("client_language") or ""),
        facts,
        [str(item) for item in questions if str(item).strip()],
        note,
    )
    reference = str(row.get("reference") or "request")
    st.caption(i18n.text(shop_lang, "text_ready"))
    st.text_area(
        i18n.text(shop_lang, "copy_summary"),
        value=summary,
        height=180,
        key=f"shop_text_{reference}_{shop_lang}",
    )
    phone = os.getenv("SHOP_PHONE", "").strip()
    link = i18n.sms_link(phone, summary)
    if link:
        label = html.escape(i18n.text(shop_lang, "send_text", phone=phone))
        href = html.escape(link, quote=True)
        st.markdown(f'<a href="{href}">{label}</a>', unsafe_allow_html=True)


def _show_recommendations(items: list[Any]) -> None:
    """Show the planning notes returned with the bracket."""
    with st.container(border=True):
        st.caption("Recommendations")
        if not items:
            st.write("The catalog did not return a recommendation for this takeoff.")
            return
        for item in items:
            if not isinstance(item, dict):
                continue
            st.markdown(f"**{item.get('title', 'Recommendation')}**")
            st.write(str(item.get("detail", "")))


def _dollars(value: Any) -> str:
    """Format a number as dollars."""
    if not isinstance(value, (int, float)):
        return "—"
    return f"${value:,.2f}"


def _get_bytes(api_base: str, path: str, api_key: str = "") -> bytes:
    """Read one binary response from the Flask API.

    Args:
        api_base: Flask origin, without a trailing path.
        path: Absolute path such as ``/leads/<id>/photo``.
        api_key: Optional ``APP_API_KEY`` value.

    Returns:
        The response body.

    Raises:
        ApiError: The API returned an error or could not be reached.
    """
    if _use_local(api_base):
        return _local_photo_bytes(path)
    headers = {"X-API-Key": api_key.strip()} if api_key.strip() else {}
    url = api_base.strip().rstrip("/") + path
    if not url.startswith("http://") and not url.startswith("https://"):
        raise ApiError("Enter an API base URL that starts with http:// or https://.")
    req = request.Request(url, headers=headers, method="GET")
    try:
        with request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            return response.read()
    except error.HTTPError as exc:
        raise ApiError(_error_message(exc.read())) from exc
    except error.URLError:
        return _local_photo_bytes(path)
    except TimeoutError as exc:
        raise ApiError("The API took too long to respond.") from exc


def _request(
    method: str,
    api_base: str,
    path: str,
    api_key: str = "",
    payload: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Call a JSON endpoint on the Flask API.

    Args:
        method: HTTP method.
        api_base: Flask origin, without a trailing path.
        path: Absolute path such as ``/estimate``.
        api_key: Optional ``APP_API_KEY`` value.
        payload: JSON body for POST requests.

    Returns:
        HTTP status and the parsed JSON body.

    Raises:
        ApiError: The API returned an error or could not be reached.
    """
    if _use_local(api_base):
        return _local_json(path, payload)
    headers = {"X-API-Key": api_key.strip()} if api_key.strip() else {}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    url = api_base.strip().rstrip("/") + path
    if not url.startswith("http://") and not url.startswith("https://"):
        raise ApiError("Enter an API base URL that starts with http:// or https://.")
    req = request.Request(url, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            return response.status, _decode_json(response.read())
    except error.HTTPError as exc:
        raise ApiError(_error_message(exc.read())) from exc
    except error.URLError:
        return _local_json(path, payload)
    except TimeoutError as exc:
        raise ApiError("The API took too long to respond.") from exc


def _use_local(api_base: str) -> bool:
    """Return whether this page should run the estimator in-process."""
    return not api_base.strip()


def _services() -> Any:
    """Load the estimator functions without starting a server."""
    import app as api

    return api


def _visitor() -> str:
    """Return the visitor address for the in-process chat limit."""
    try:
        address = st.context.ip_address
    except Exception:
        return "local"
    return str(address or "local")


def _local_error(exc: Exception) -> ApiError:
    """Turn an estimator error into the message the page already shows."""
    return ApiError(str(exc))


def _local_json(path: str, payload: dict[str, Any] | None) -> tuple[int, dict[str, Any]]:
    """Run one JSON action inside this process."""
    api = _services()
    try:
        if path == "/catalog":
            return 200, api.run_catalog()
        if path == "/estimate":
            return 200, api.run_estimate(payload or {})
        if path == "/chat":
            return 200, api.run_chat(payload or {}, _visitor())
        if path == "/lead":
            return 201, api.run_lead(payload or {}, _visitor())
        if path == "/leads":
            return 200, api.run_leads()
    except api.AppError as exc:
        raise _local_error(exc) from exc
    raise ApiError("This page cannot run that request.")


def _local_photo(
    filename: str,
    data: bytes,
    project_type: str,
    vision_key: str,
    path: str,
) -> tuple[int, dict[str, Any]]:
    """Read or draw one property photo inside this process."""
    api = _services()
    try:
        if path == "/mockup":
            return 200, api.run_mockup(filename, data, project_type, vision_key)
        return 200, api.run_analyze_photo(filename, data, project_type, vision_key)
    except api.AppError as exc:
        raise _local_error(exc) from exc


def _local_photo_bytes(path: str) -> bytes:
    """Read one shop photo from local storage."""
    parts = [part for part in path.split("/") if part]
    if len(parts) != 4 or parts[0] != "leads" or parts[2] != "photos" or not parts[3].isdigit():
        raise ApiError("This photo is not available.")
    api = _services()
    try:
        return api.run_lead_photo(parts[1], int(parts[3]))
    except api.AppError as exc:
        raise _local_error(exc) from exc


def _decode_json(raw: bytes) -> dict[str, Any]:
    """Parse a response body as a JSON object."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ApiError("The API returned a response that was not JSON.") from exc
    if not isinstance(parsed, dict):
        raise ApiError("The API returned an unexpected response.")
    return parsed


def _error_message(raw: bytes) -> str:
    """Read the Flask error message, falling back to a generic sentence."""
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "The API request failed."
    if isinstance(parsed, dict):
        error_body = parsed.get("error")
        if isinstance(error_body, dict) and error_body.get("message"):
            return str(error_body["message"])
    return "The API request failed."


if __name__ == "__main__":
    main()
