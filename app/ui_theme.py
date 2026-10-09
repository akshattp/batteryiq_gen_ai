"""app/ui_theme.py - Angular-inspired Light Enterprise Design System for BatteryIQ & AI Battery Safety Guardian.

Exact 5-Color Palette Specification:
- #0091B9: Cerulean Teal (Primary Interactive / Active Highlights / H3 Headers / Key Borders)
- #BAE4F0: Soft Ice Blue (Secondary Contrast / Badges / Subtle Accents)
- #004E9B: Deep Navy (Brand Foundation / Primary Typography / Section Headers)
- #FF6500: Safety Orange (Critical Alerts / High Risk / Override Badges / Active Navigation Indicator)
- #FFD500: Warning Yellow (Caution / Threshold Advisories / Accent Tickers)
"""

from __future__ import annotations

import base64
from pathlib import Path
import plotly.graph_objects as go
import streamlit as st


def get_logo_base64() -> str:
    """Return base64-encoded string of BatteryIQ logo if exists."""
    candidates = [
        Path("app/assets/batteryiq_logo.jpg"),
        Path("app/assets/batteryiq_logo.png"),
    ]
    for p in candidates:
        if p.exists():
            try:
                with open(p, "rb") as f:
                    data = base64.b64encode(f.read()).decode("utf-8")
                    return f"data:image/jpeg;base64,{data}"
            except Exception:
                pass
    return ""


def render_sidebar_brand() -> None:
    """Render high-end BatteryIQ brand logo at the top of the sidebar."""
    logo_file = Path("app/assets/batteryiq_logo.jpg")
    if logo_file.exists() and hasattr(st, "logo"):
        try:
            st.logo(str(logo_file), size="large")
        except Exception:
            pass

    logo_src = get_logo_base64()
    if logo_src:
        img_html = f'<img src="{logo_src}" style="width: 48px; height: 48px; border-radius: 10px; object-fit: cover; border: 2px solid #BAE4F0; box-shadow: 0 2px 10px rgba(0, 78, 155, 0.12);" />'
    else:
        img_html = '<div style="width: 48px; height: 48px; border-radius: 10px; background: linear-gradient(135deg, #0091B9, #004E9B); display: flex; align-items: center; justify-content: center; font-size: 1.5rem; color: #fff;">⚡</div>'

    st.sidebar.markdown(
        f"""
        <div style="padding: 12px 14px; margin-bottom: 14px; background: #FFFFFF; border: 1px solid rgba(0, 145, 185, 0.3); border-radius: 12px; box-shadow: 0 4px 16px rgba(0, 78, 155, 0.08);">
            <div style="display: flex; align-items: center; gap: 12px;">
                {img_html}
                <div>
                    <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.35rem; font-weight: 800; color: #002B5B; line-height: 1.1; letter-spacing: -0.02em;">
                        Battery<span style="color: #0091B9;">IQ</span>
                    </div>
                    <div style="font-size: 0.68rem; font-weight: 700; color: #FF6500; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 2px;">
                        Energy Intelligence
                    </div>
                </div>
            </div>
            <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid rgba(0, 145, 185, 0.15); display: flex; justify-content: space-between; align-items: center; font-size: 0.72rem; color: #004E9B;">
                <span><span class="chip-pulse" style="width: 6px; height: 6px; background-color: #0091B9; display: inline-block;"></span> Digital Twin Platform</span>
                <span style="font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #004E9B; background: #EAF4F9; padding: 2px 6px; border-radius: 4px;">v2.4</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def inject_theme() -> None:
    """Inject modern Angular-inspired enterprise Light Mode CSS styling with full custom palette."""
    st.markdown(
        """
        <style>
        /* 1. Google Fonts Import */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

        /* 2. Global Palette Variables (LIGHT THEME) */
        :root {
            --c-teal: #0091B9;
            --c-ice-blue: #BAE4F0;
            --c-navy: #004E9B;
            --c-orange: #FF6500;
            --c-yellow: #FFD500;

            --bg-app: #F4F8FC;
            --bg-surface: #FFFFFF;
            --bg-card: #FFFFFF;
            --border-card: rgba(0, 145, 185, 0.22);
            --border-glow: #0091B9;
            --text-primary: #002B5B;
            --text-secondary: #004E9B;
            --text-muted: #486581;
        }

        /* 3. Global Light Canvas Background */
        html, body, .stApp, 
        [data-testid="stAppViewContainer"], 
        [data-testid="stAppViewContainer"] > .main,
        section.main,
        .main .block-container {
            background: #F4F8FC !important;
            background-color: #F4F8FC !important;
            color: #002B5B !important;
            font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
        }

        [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at 10% 10%, #EBF4FA 0%, #F4F8FC 100%) !important;
        }

        /* 4. Streamlit Main Container Spacing */
        .main .block-container {
            max-width: 1420px;
            padding-top: 3.6rem;
            padding-bottom: 3.5rem;
        }

        /* ========================================================
           5. FIXED TOP BAR - BATTERY SAFETY GUARDIAN TITLE
           ======================================================== */
        header[data-testid="stHeader"] {
            background: #FFFFFF !important;
            background-color: #FFFFFF !important;
            border-bottom: 2px solid #0091B9 !important;
            box-shadow: 0 2px 14px rgba(0, 78, 155, 0.08) !important;
            height: 52px !important;
            display: flex !important;
            align-items: center !important;
            z-index: 999990 !important;
        }

        header[data-testid="stHeader"]::after {
            content: "⚡ Battery Safety Guardian — AI-Enabled Battery Safety Guardian for Electric Vehicles using Anomaly Detection, Predictive Analytics and Generative AI";
            position: absolute;
            left: 5.5rem;
            right: 6rem;
            top: 50%;
            transform: translateY(-50%);
            color: #002B5B;
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 0.86rem;
            font-weight: 700;
            letter-spacing: -0.01em;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            pointer-events: none;
        }

        @media (max-width: 960px) {
            header[data-testid="stHeader"]::after {
                content: "⚡ Battery Safety Guardian — AI for Electric Vehicles (BatteryIQ)";
            }
        }

        /* Corporate Top Banner Card */
        .fixed-top-banner {
            background: #FFFFFF;
            border: 1px solid rgba(0, 145, 185, 0.28);
            border-left: 5px solid #0091B9;
            border-radius: 10px;
            padding: 9px 16px;
            margin-bottom: 18px;
            display: flex;
            align-items: center;
            gap: 12px;
            box-shadow: 0 2px 10px rgba(0, 78, 155, 0.05);
            font-size: 0.85rem;
        }

        /* ========================================================
           6. SIDEBAR NAVIGATION - CLEAN LIGHT ENTERPRISE CONTRAST
           ======================================================== */
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] > div,
        [data-testid="stSidebarContent"],
        [data-testid="stSidebarNav"],
        [data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
            background: #F0F6FA !important;
            background-color: #F0F6FA !important;
            border-right: 2px solid rgba(0, 145, 185, 0.25) !important;
            box-shadow: 2px 0 16px rgba(0, 78, 155, 0.06) !important;
        }

        /* Top Brand Logo Container */
        [data-testid="stLogo"] {
            padding: 8px 14px 4px 14px !important;
            margin-bottom: 4px !important;
        }
        [data-testid="stLogo"] img {
            border-radius: 10px !important;
            border: 2px solid #BAE4F0 !important;
            box-shadow: 0 2px 12px rgba(0, 78, 155, 0.12) !important;
        }

        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.2rem !important;
        }

        /* Force high contrast text on sidebar content */
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] div,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] a {
            color: #002B5B !important;
        }

        /* Sidebar Navigation Container */
        [data-testid="stSidebarNav"] {
            padding-top: 0.4rem;
        }

        [data-testid="stSidebarNav"] ul {
            gap: 6px;
        }

        /* All navigation links in the sidebar */
        [data-testid="stSidebarNav"] ul li a,
        [data-testid="stSidebarNav"] a,
        [data-testid="stSidebarNavLink"] {
            background: #FFFFFF !important;
            background-color: #FFFFFF !important;
            border: 1px solid rgba(0, 145, 185, 0.25) !important;
            border-radius: 8px !important;
            padding: 10px 16px !important;
            margin: 4px 8px !important;
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
            text-decoration: none !important;
            display: flex !important;
            align-items: center !important;
            box-shadow: 0 2px 6px rgba(0, 78, 155, 0.04) !important;
        }

        /* Text inside navigation pills */
        [data-testid="stSidebarNav"] ul li a span,
        [data-testid="stSidebarNav"] a span,
        [data-testid="stSidebarNavLink"] span,
        [data-testid="stSidebarNav"] span,
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] span {
            color: #004E9B !important;
            font-size: 0.94rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.01em !important;
            display: inline-block !important;
            opacity: 1 !important;
            visibility: visible !important;
        }

        /* Hover Navigation State */
        [data-testid="stSidebarNav"] ul li a:hover,
        [data-testid="stSidebarNav"] a:hover,
        [data-testid="stSidebarNavLink"]:hover {
            background: #E1F2F8 !important;
            background-color: #E1F2F8 !important;
            border-color: #0091B9 !important;
            transform: translateX(4px) !important;
            box-shadow: 0 4px 14px rgba(0, 145, 185, 0.16) !important;
        }

        [data-testid="stSidebarNav"] ul li a:hover span,
        [data-testid="stSidebarNav"] a:hover span,
        [data-testid="stSidebarNavLink"]:hover span {
            color: #002B5B !important;
            font-weight: 800 !important;
        }

        /* Active Navigation Page */
        [data-testid="stSidebarNav"] ul li a[aria-current="page"],
        [data-testid="stSidebarNav"] a[aria-current="page"],
        [data-testid="stSidebarNavLink"][aria-current="page"],
        [data-testid="stSidebarNav"] li[aria-current="page"] a {
            background: #DCEEF5 !important;
            background-color: #DCEEF5 !important;
            border-left: 5px solid #FF6500 !important;
            border-top: 1px solid #0091B9 !important;
            border-right: 1px solid #0091B9 !important;
            border-bottom: 1px solid #0091B9 !important;
            box-shadow: 0 4px 14px rgba(0, 145, 185, 0.16) !important;
        }

        [data-testid="stSidebarNav"] ul li a[aria-current="page"] span,
        [data-testid="stSidebarNav"] a[aria-current="page"] span,
        [data-testid="stSidebarNavLink"][aria-current="page"] span,
        [data-testid="stSidebarNav"] li[aria-current="page"] a span {
            color: #002B5B !important;
            font-weight: 800 !important;
        }

        /* Sidebar Inputs & Controls */
        [data-testid="stSidebarNav"] input,
        section[data-testid="stSidebar"] input {
            background: #FFFFFF !important;
            color: #002B5B !important;
            border: 1px solid rgba(0, 145, 185, 0.4) !important;
            border-radius: 8px !important;
            padding: 8px 12px !important;
        }

        [data-testid="stSidebarNav"] input::placeholder,
        section[data-testid="stSidebar"] input::placeholder {
            color: #627D98 !important;
            opacity: 0.9 !important;
        }

        section[data-testid="stSidebar"] div[data-baseweb="select"] {
            background: #FFFFFF !important;
            border: 1px solid rgba(0, 145, 185, 0.4) !important;
            border-radius: 8px !important;
        }

        section[data-testid="stSidebar"] div[data-baseweb="select"] span {
            color: #002B5B !important;
        }

        /* ========================================================
           7. TYPOGRAPHY HIERARCHY (H1, H2, H3, H4) - LIGHT THEME
           ======================================================== */
        /* H1: Primary Crown Header */
        h1, .h1-title, [data-testid="stMarkdownContainer"] h1 {
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            font-size: 2.3rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.03em !important;
            color: #002B5B !important;
            line-height: 1.2 !important;
            margin-top: 0 !important;
            margin-bottom: 0.75rem !important;
        }

        /* H2: Standout Section Header */
        h2, .h2-title, [data-testid="stMarkdownContainer"] h2 {
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            font-size: 1.65rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em !important;
            color: #004E9B !important;
            line-height: 1.3 !important;
            border-left: 5px solid #0091B9 !important;
            padding-left: 14px !important;
            margin-top: 2rem !important;
            margin-bottom: 1rem !important;
        }

        /* H3: Subsection / Card Header */
        h3, .h3-title, [data-testid="stMarkdownContainer"] h3 {
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            font-size: 1.25rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.01em !important;
            color: #0091B9 !important;
            line-height: 1.35 !important;
            margin-top: 1.3rem !important;
            margin-bottom: 0.6rem !important;
        }

        /* H4: Group Label / Kicker / Metric Category */
        h4, .h4-title, [data-testid="stMarkdownContainer"] h4 {
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            font-size: 0.95rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.08em !important;
            text-transform: uppercase !important;
            color: #004E9B !important;
            margin-top: 1rem !important;
            margin-bottom: 0.4rem !important;
        }

        /* Body & Paragraphs */
        p, li, [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {
            font-size: 0.95rem !important;
            color: #1A365D !important;
            line-height: 1.65 !important;
        }

        /* Code & Monospace */
        code, pre, .mono {
            font-family: 'JetBrains Mono', monospace !important;
            background: #EAF4F9 !important;
            color: #002B5B !important;
        }

        /* ========================================================
           8. ENTERPRISE CARDS & HERO BANNER (LIGHT THEME)
           ======================================================== */
        .angular-card {
            background: #FFFFFF !important;
            background-color: #FFFFFF !important;
            border: 1px solid rgba(0, 145, 185, 0.22) !important;
            border-radius: 14px !important;
            padding: 22px 26px !important;
            margin-bottom: 20px !important;
            box-shadow: 0 4px 20px rgba(0, 78, 155, 0.06) !important;
            transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        }

        .angular-card:hover {
            border-color: #0091B9 !important;
            box-shadow: 0 8px 28px rgba(0, 145, 185, 0.16) !important;
            transform: translateY(-2px) !important;
        }

        .hero-banner {
            background: linear-gradient(135deg, #EBF4FA 0%, #FFFFFF 100%) !important;
            border: 1px solid rgba(0, 145, 185, 0.28) !important;
            border-radius: 16px !important;
            padding: 26px 30px !important;
            margin-bottom: 24px !important;
            position: relative !important;
            overflow: hidden !important;
            box-shadow: 0 6px 24px rgba(0, 78, 155, 0.07) !important;
        }

        .hero-banner::before {
            content: "" !important;
            position: absolute !important;
            top: 0 !important;
            left: 0 !important;
            right: 0 !important;
            height: 4px !important;
            background: linear-gradient(90deg, #004E9B 0%, #0091B9 35%, #FFD500 70%, #FF6500 100%) !important;
        }

        /* ========================================================
           9. STAT TILES (LIGHT THEME)
           ======================================================== */
        .stat-tile {
            background: #FFFFFF !important;
            background-color: #FFFFFF !important;
            border: 1px solid rgba(0, 145, 185, 0.2) !important;
            border-left: 4px solid #0091B9 !important;
            border-radius: 12px !important;
            padding: 16px 18px !important;
            text-align: left !important;
            position: relative !important;
            transition: all 0.25s ease !important;
            box-shadow: 0 4px 14px rgba(0, 78, 155, 0.05) !important;
        }

        .stat-tile:hover {
            border-color: #0091B9 !important;
            border-left-color: #FF6500 !important;
            box-shadow: 0 6px 20px rgba(0, 145, 185, 0.15) !important;
            transform: translateY(-2px) !important;
        }

        .stat-tile-label {
            font-size: 0.8rem !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.08em !important;
            color: #004E9B !important;
            margin-bottom: 6px !important;
            display: flex !important;
            align-items: center !important;
            gap: 6px !important;
        }

        .stat-tile-val {
            font-size: 1.8rem !important;
            font-weight: 800 !important;
            color: #002B5B !important;
            font-family: 'JetBrains Mono', monospace !important;
            line-height: 1.2 !important;
        }

        .stat-tile-sub {
            font-size: 0.78rem !important;
            color: #486581 !important;
            margin-top: 4px !important;
            font-weight: 500 !important;
        }

        /* ========================================================
           10. PILL CHIPS & BADGES (LIGHT THEME CONTRAST)
           ======================================================== */
        .chip {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        .chip-info {
            background: #EAF4F9;
            color: #004E9B;
            border: 1px solid #0091B9;
        }

        .chip-warning {
            background: #FFF9E6;
            color: #8C5B00;
            border: 1px solid #FFD500;
        }

        .chip-danger {
            background: #FFF1E6;
            color: #B33C00;
            border: 1px solid #FF6500;
        }

        .chip-success {
            background: #E8F5F8;
            color: #004E9B;
            border: 1px solid #0091B9;
        }

        .chip-pulse {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: #0091B9;
            box-shadow: 0 0 8px rgba(0, 145, 185, 0.6);
            animation: pulse-dot 2s infinite;
        }

        @keyframes pulse-dot {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 145, 185, 0.5); }
            70% { transform: scale(1.05); box-shadow: 0 0 0 5px rgba(0, 145, 185, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 145, 185, 0); }
        }

        /* 11. Buttons & Interactive Controls */
        button[kind="primary"], div.stButton > button:first-child {
            background: linear-gradient(135deg, #0091B9 0%, #004E9B 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 700 !important;
            box-shadow: 0 4px 14px rgba(0, 145, 185, 0.28) !important;
            transition: all 0.2s ease !important;
        }

        button[kind="primary"]:hover, div.stButton > button:first-child:hover {
            box-shadow: 0 6px 20px rgba(0, 145, 185, 0.45) !important;
            transform: translateY(-1px) !important;
            color: #ffffff !important;
        }

        /* 12. Streamlit Metrics Box Polish */
        div[data-testid="stMetric"] {
            background: #FFFFFF !important;
            border: 1px solid rgba(0, 145, 185, 0.25) !important;
            border-radius: 10px !important;
            padding: 12px 16px !important;
            box-shadow: 0 2px 8px rgba(0, 78, 155, 0.04) !important;
        }

        div[data-testid="stMetricLabel"] {
            color: #004E9B !important;
            font-weight: 600 !important;
        }

        div[data-testid="stMetricValue"] {
            color: #002B5B !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-weight: 700 !important;
        }

        /* 13. Standardized Enterprise Footer */
        .angular-footer {
            margin-top: 45px;
            padding: 18px 24px;
            background: #FFFFFF;
            border: 1px solid rgba(0, 145, 185, 0.25);
            border-radius: 12px;
            text-align: center;
            font-size: 0.84rem;
            color: #486581;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
            box-shadow: 0 2px 12px rgba(0, 78, 155, 0.04);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(title: str, subtitle: str, badge_text: str = "PRODUCTION AIR-GAP", tag: str = "AI GUARDIAN v2.4") -> None:
    """Render enterprise Light header with status indicators, top bar, and sidebar brand logo."""
    inject_theme()
    render_sidebar_brand()

    # Prominent Corporate Top Bar
    st.markdown(
        """
        <div class="fixed-top-banner">
            <span class="chip chip-info" style="font-size: 0.72rem; padding: 3px 10px;">BATTERYIQ PLATFORM</span>
            <span style="font-weight: 800; color: #002B5B; font-size: 0.92rem;">Battery Safety Guardian</span>
            <span style="color: #BAE4F0; font-weight: 900;">&bull;</span>
            <span style="color: #004E9B; font-weight: 600; font-size: 0.86rem;">AI-Enabled Battery Safety Guardian for Electric Vehicles using Anomaly Detection, Predictive Analytics and Generative AI</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Hero Banner
    st.markdown(
        f"""
        <div class="hero-banner" style="background: linear-gradient(135deg, #EBF4FA 0%, #FFFFFF 100%) !important; border: 1px solid rgba(0, 145, 185, 0.28); border-radius: 16px; padding: 26px 30px; margin-bottom: 24px; box-shadow: 0 6px 24px rgba(0, 78, 155, 0.07);">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
                <div>
                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                        <span class="chip chip-info" style="font-weight: 700;">{tag}</span>
                        <span class="chip chip-success" style="font-weight: 700;"><span class="chip-pulse"></span> {badge_text}</span>
                    </div>
                    <h1 class="h1-title" style="color: #002B5B !important; font-size: 2.2rem; font-weight: 800; margin: 0 0 8px 0; letter-spacing: -0.03em;">
                        {title}
                    </h1>
                    <p style="font-size: 0.98rem; color: #004E9B !important; margin: 0; font-weight: 600;">
                        {subtitle}
                    </p>
                </div>
                <div style="text-align: right; font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #002B5B; background: #FFFFFF; padding: 10px 14px; border-radius: 8px; border: 1px solid rgba(0, 145, 185, 0.25); box-shadow: 0 2px 8px rgba(0, 78, 155, 0.05);">
                    <div>SYSTEM: <span style="color: #0091B9; font-weight: 700;">ACTIVE</span></div>
                    <div>HORIZON: <span style="color: #B33C00; font-weight: 700;">t + 10s</span></div>
                    <div>LATENCY: <span style="color: #004E9B; font-weight: 700;">&lt; 15ms</span></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    """Render standardized enterprise compliance and disclaimer footer with Light palette."""
    st.markdown(
        """
        <div class="angular-footer">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span class="chip-pulse"></span>
                <span style="color: #002B5B; font-weight: 700;">BatteryIQ &bull; Battery Safety Guardian &bull; High-Voltage Digital Twin</span>
            </div>
            <div style="color: #486581;">
                ⚠️ <strong>Decision-Support Prototype.</strong> Human-in-the-loop diagnostic guidance only. Not for autonomous vehicle disconnection.
            </div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #004E9B; font-weight: 700;">
                ISO 26262 &bull; IEC 62619 Guidelines
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def make_gauge_chart(score: float, title: str = "Risk Score") -> go.Figure:
    """Generate high-end Angular-styled gauge chart for risk scores (0–100) using Light palette."""
    if score >= 75.0:
        bar_color = "#FF6500"
    elif score >= 50.0:
        bar_color = "#FFD500"
    elif score >= 25.0:
        bar_color = "#0091B9"
    else:
        bar_color = "#004E9B"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "/100", "font": {"size": 28, "color": "#002B5B", "family": "JetBrains Mono"}},
            title={"text": title, "font": {"size": 13, "color": "#004E9B"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#004E9B"},
                "bar": {"color": bar_color, "thickness": 0.3},
                "bgcolor": "#F4F8FC",
                "borderwidth": 1,
                "bordercolor": "rgba(0, 145, 185, 0.4)",
                "steps": [
                    {"range": [0, 25], "color": "rgba(0, 145, 185, 0.15)"},
                    {"range": [25, 50], "color": "rgba(186, 228, 240, 0.4)"},
                    {"range": [50, 75], "color": "rgba(255, 213, 0, 0.28)"},
                    {"range": [75, 100], "color": "rgba(255, 101, 0, 0.28)"},
                ],
                "threshold": {
                    "line": {"color": "#002B5B", "width": 3},
                    "thickness": 0.75,
                    "value": score,
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=15, r=15, t=35, b=10),
        height=190,
    )
    return fig
