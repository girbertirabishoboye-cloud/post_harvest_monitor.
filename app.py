import json
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional, Tuple

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ----------------------------- Configuration -----------------------------
st.set_page_config(
    page_title="Post-Harvest Monitor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_DEFAULT = "https://postharvestmonitor-default-rtdb.firebaseio.com"
REFRESH_SECONDS = 8
RISK_COLORS = {"NORMAL": "#22C55E", "WARNING": "#F59E0B", "HIGH RISK": "#F87171", "UNKNOWN": "#94A3B8"}
PALETTE = {
    "background": "#101827", "card": "#1C2940", "blue": "#3B82F6",
    "light_blue": "#60A5FA", "teal": "#14B8A6", "green": "#22C55E",
    "amber": "#F59E0B", "coral": "#F87171", "text": "#F1F5F9",
    "muted": "#CBD5E1",
}

st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp { background: #101827; color: #F1F5F9; }
[data-testid="stHeader"] { background: rgba(16,24,39,.92); }
[data-testid="stSidebar"] { background: #111D30; border-right: 1px solid #2B3A52; }
.block-container { padding-top: 1.4rem; padding-bottom: 2.5rem; max-width: 1500px; }
h1,h2,h3,h4,p, label { color: #F1F5F9; }
p, .stMarkdown, [data-testid="stCaptionContainer"] { color: #CBD5E1; }
.metric-card {
  background: linear-gradient(145deg,#1C2940,#19263B);
  border: 1px solid #31415B; border-radius: 16px; padding: 17px 19px;
  min-height: 128px; box-shadow: 0 8px 24px rgba(0,0,0,.12);
}
.metric-label { color:#AFC0D5; font-size:.86rem; margin-bottom:8px; }
.metric-value { color:#F1F5F9; font-size:1.85rem; font-weight:750; line-height:1.15; }
.metric-sub { color:#94A3B8; font-size:.78rem; margin-top:8px; }
.panel {
  background:#1C2940; border:1px solid #31415B; border-radius:16px;
  padding:18px 20px; margin-bottom:12px;
}
.small-muted { color:#9FB0C6; font-size:.85rem; }
.pill { display:inline-block; border-radius:999px; padding:5px 11px; font-weight:700; font-size:.78rem; }
.flow-node { background:#243650; border:1px solid #3B82F6; border-radius:12px; padding:14px 8px; text-align:center; font-weight:700; }
div[data-testid="stDataFrame"] { border:1px solid #31415B; border-radius:12px; }
.stButton>button, .stDownloadButton>button { border-radius:10px; }
@media (max-width: 700px) {
 .block-container { padding-left: 1rem; padding-right: 1rem; }
 .metric-value { font-size:1.45rem; }
}
</style>
""", unsafe_allow_html=True)


# ----------------------------- Firebase layer -----------------------------
def get_config() -> Tuple[str, Optional[str]]:
    """Read Firebase settings from Streamlit Secrets; database URL can use the documented default."""
    try:
        url = st.secrets.get("firebase", {}).get("database_url", DB_DEFAULT)
        token = st.secrets.get("firebase", {}).get("auth_token")
    except Exception:
        url, token = DB_DEFAULT, None
    return str(url).rstrip("/"), token


def firebase_get(path: str, timeout: int = 8) -> Any:
    base_url, token = get_config()
    url = f"{base_url}/{path.strip('/')}.json"
    params = {"auth": token} if token else None
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and payload.get("error"):
        raise RuntimeError(str(payload["error"]))
    return payload


def fetch_latest() -> Any:
    return firebase_get("sensor_readings/latest")


def fetch_history() -> Any:
    return firebase_get("sensor_readings/history")


def fetch_status(path: str) -> Any:
    return firebase_get(path)


def safe_fetch_all() -> Dict[str, Any]:
    result = {"latest": None, "history": None, "system_status": None,
              "sensor_status": None, "error": None, "history_error": None}
    try:
        result["latest"] = fetch_latest()
    except Exception as exc:
        result["error"] = str(exc)
    try:
        result["history"] = fetch_history()
    except Exception as exc:
        result["history_error"] = str(exc)
    for key, path in [("system_status", "system_status"), ("sensor_status", "sensor_status")]:
        try:
            result[key] = fetch_status(path)
        except Exception:
            result[key] = None
    return result


# ----------------------------- Data handling -----------------------------
def parse_timestamp(value: Any) -> pd.Timestamp:
    if value is None or value == "":
        return pd.NaT
    try:
        if isinstance(value, (int, float)) or (isinstance(value, str) and value.strip().isdigit()):
            number = float(value)
            # Firebase timestamp_ms is milliseconds; smaller values are likely seconds.
            return pd.to_datetime(number, unit="ms" if number > 10_000_000_000 else "s", utc=True)
        return pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:
        return pd.NaT


def records_to_frame(raw: Any) -> pd.DataFrame:
    if raw is None:
        return pd.DataFrame()
    if isinstance(raw, dict):
        records = []
        for key, value in raw.items():
            if isinstance(value, dict):
                row = dict(value)
                row.setdefault("_firebase_key", str(key))
                records.append(row)
        if not records and any(k in raw for k in ("temperature", "humidity", "soil_moisture_percent")):
            records = [raw]
    elif isinstance(raw, list):
        records = [dict(x) for x in raw if isinstance(x, dict)]
    else:
        records = []
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if "timestamp_ms" in df.columns:
        df["timestamp"] = df["timestamp_ms"].apply(parse_timestamp)
    elif "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].apply(parse_timestamp)
    elif "_firebase_key" in df.columns:
        df["timestamp"] = df["_firebase_key"].apply(parse_timestamp)
    else:
        df["timestamp"] = pd.NaT
    for col in ["temperature", "humidity", "soil_moisture_raw", "soil_moisture_percent"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "environmental_risk" in df.columns:
        df["environmental_risk"] = df["environmental_risk"].apply(normalize_risk)
    if "soil_condition" in df.columns:
        df["soil_condition"] = df["soil_condition"].astype(str).str.upper()
    return df.sort_values("timestamp", na_position="last").reset_index(drop=True)


def normalize_risk(value: Any) -> str:
    if value is None:
        return "UNKNOWN"
    text = str(value).strip().upper().replace("_", " ").replace("-", " ")
    if "HIGH" in text or "DANGER" in text or "CRITICAL" in text:
        return "HIGH RISK"
    if "WARN" in text or "MODERATE" in text:
        return "WARNING"
    if "NORMAL" in text or "LOW" in text or "SAFE" in text:
        return "NORMAL"
    return text if text else "UNKNOWN"


def value_or_none(record: Any, key: str):
    if isinstance(record, dict):
        return record.get(key)
    return None


def fmt(value: Any, suffix: str = "", decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    try:
        return f"{float(value):.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return f"{value}{suffix}"


def format_time(value: Any) -> str:
    ts = parse_timestamp(value)
    if pd.isna(ts):
        return "Not provided"
    return ts.strftime("%d %b %Y, %H:%M:%S UTC")


def device_status(sensor_status: Any, system_status: Any, latest: Any) -> Tuple[str, str]:
    """Only claim ONLINE/OFFLINE when a timestamp heartbeat is present."""
    candidates = []
    for obj in [sensor_status, system_status, latest]:
        if isinstance(obj, dict):
            for key in ["last_seen", "last_seen_ms", "heartbeat", "heartbeat_ms", "timestamp_ms"]:
                if obj.get(key) is not None:
                    candidates.append(obj.get(key))
    if not candidates:
        return "UNVERIFIED", "No heartbeat/last_seen field found in Firebase."
    last_seen = parse_timestamp(candidates[0])
    if pd.isna(last_seen):
        return "UNVERIFIED", "Heartbeat exists but its timestamp could not be parsed."
    age = (pd.Timestamp.now(tz="UTC") - last_seen).total_seconds()
    if age <= 120:
        return "ONLINE", f"Last seen {format_time(candidates[0])}"
    return "OFFLINE", f"Last seen {format_time(candidates[0])} ({int(age/60)} min ago)"


def risk_actions(risk: str, recommendation: Any = None) -> list:
    if recommendation and str(recommendation).strip():
        return [str(recommendation).strip()]
    actions = {
        "NORMAL": ["Continue routine monitoring and keep storage conditions stable."],
        "WARNING": ["Check temperature and humidity trends.", "Inspect ventilation and storage practices.", "Recheck sensor readings soon."],
        "HIGH RISK": ["Inspect the storage environment promptly.", "Improve ventilation or adjust storage conditions where appropriate.", "Check produce condition and seek local post-harvest guidance."],
    }
    return actions.get(risk, ["No recommendation is available from the device. Verify sensor readings and firmware fields."])


def chart_layout(fig, height=310):
    fig.update_layout(
        template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=8, r=8, t=38, b=8),
        font=dict(color="#CBD5E1"), legend=dict(orientation="h", y=1.12),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,.15)")
    return fig


def line_chart(df: pd.DataFrame, column: str, title: str, unit: str):
    if df.empty or column not in df.columns or df[column].notna().sum() == 0:
        st.info(f"No {title.lower()} history is available yet.")
        return
    plot = df.dropna(subset=[column]).copy()
    x = "timestamp" if plot["timestamp"].notna().any() else "_firebase_key"
    fig = px.line(plot, x=x, y=column, markers=True, title=title,
                  labels={column: unit, x: "Time"})
    fig.update_traces(line=dict(color=PALETTE["light_blue"], width=2.5))
    st.plotly_chart(chart_layout(fig), use_container_width=True)


# ----------------------------- Header / sidebar -----------------------------
st.sidebar.markdown("## 🌾 Post-Harvest Monitor")
st.sidebar.caption("IoT environmental monitoring")
page = st.sidebar.radio(
    "NAVIGATION",
    ["Executive Overview", "Live Sensor Monitoring", "Environmental Risk",
     "Historical Analytics", "ML Prediction", "Alerts & Recommendations", "System Information"],
)
st.sidebar.divider()
st.sidebar.caption(f"Auto-refresh: every {REFRESH_SECONDS} seconds")
st.sidebar.caption("Read-only Firebase connection")

# Streamlit fragment reruns this data section on a timer in supported Streamlit versions.
@st.fragment(run_every=REFRESH_SECONDS)
def dashboard():
    with st.spinner("Connecting to Firebase and retrieving sensor data..."):
        data = safe_fetch_all()

    latest = data["latest"]
    history_df = records_to_frame(data["history"])
    latest_df = records_to_frame(latest)
    if isinstance(latest, dict):
        latest_record = latest
    elif not latest_df.empty:
        latest_record = latest_df.iloc[-1].to_dict()
    else:
        latest_record = {}

    connected = data["error"] is None
    risk = normalize_risk(value_or_none(latest_record, "environmental_risk"))
    dev_state, dev_detail = device_status(data["sensor_status"], data["system_status"], latest_record)
    updated = value_or_none(latest_record, "timestamp_ms") or value_or_none(latest_record, "timestamp")

    st.title("🌾 Machine Learning-Based Post-Harvest Loss Monitoring")
    st.caption("Monitor storage conditions, identify environmental risk signals, and support timely farmer action.")

    if connected:
        st.success("Firebase database reachable · Latest record request completed")
    else:
        st.error(f"Firebase connection failed: {data['error']}. Check the database URL, network access, and Firebase read rules.")
    if data["history_error"]:
        st.warning(f"Historical records could not be loaded: {data['history_error']}")

    if page == "Executive Overview":
        st.subheader("Executive Overview")
        c1, c2, c3, c4 = st.columns(4)
        metrics = [
            (c1, "Temperature", fmt(value_or_none(latest_record, "temperature"), " °C"), "🌡️"),
            (c2, "Relative humidity", fmt(value_or_none(latest_record, "humidity"), " %"), "💧"),
            (c3, "Soil moisture", fmt(value_or_none(latest_record, "soil_moisture_percent"), " %"), "🌱"),
            (c4, "Environmental risk", risk, "⚠️"),
        ]
        for col, label, value, icon in metrics:
            with col:
                st.markdown(f'<div class="metric-card"><div class="metric-label">{icon} {label}</div><div class="metric-value">{value}</div><div class="metric-sub">Latest Firebase record</div></div>', unsafe_allow_html=True)
        st.write("")
        a, b = st.columns([1, 1])
        with a:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown("#### Device status")
            color = "#22C55E" if dev_state == "ONLINE" else "#F87171" if dev_state == "OFFLINE" else "#F59E0B"
            st.markdown(f'<span class="pill" style="background:{color}22;color:{color}">{dev_state}</span>', unsafe_allow_html=True)
            st.caption(dev_detail)
            st.markdown(f"**Last data update:** {format_time(updated)}")
            st.markdown(f"**Soil condition:** {value_or_none(latest_record, 'soil_condition') or 'Not provided'}")
            st.markdown("</div>", unsafe_allow_html=True)
        with b:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown("#### Latest device recommendation")
            for item in risk_actions(risk, value_or_none(latest_record, "recommendation")):
                st.write("• " + item)
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("#### System data flow")
        x1, x2, x3 = st.columns([1, .25, 1])
        with x1: st.markdown('<div class="flow-node">📡 ESP8266 NodeMCU<br><small>DHT11 · YL-69</small></div>', unsafe_allow_html=True)
        with x2: st.markdown("<div style='text-align:center;padding-top:18px;color:#60A5FA;font-size:24px'>→</div>", unsafe_allow_html=True)
        with x3: st.markdown('<div class="flow-node">☁️ Firebase Realtime DB<br><small>REST API</small></div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align:center;color:#60A5FA;font-size:24px">↓</div><div class="flow-node">📊 Streamlit Dashboard · Live monitoring & analytics</div>', unsafe_allow_html=True)
        st.caption("Values are shown only when present in Firebase. A dash means the field is missing or unavailable.")

    elif page == "Live Sensor Monitoring":
        st.subheader("Live Sensor Monitoring")
        cols = st.columns(3)
        specs = [("Temperature", "temperature", "°C", "🌡️"),
                 ("Relative humidity", "humidity", "%", "💧"),
                 ("Soil moisture", "soil_moisture_percent", "%", "🌱")]
        for col, (label, key, unit, icon) in zip(cols, specs):
            val = value_or_none(latest_record, key)
            with col:
                st.markdown(f'<div class="metric-card"><div class="metric-label">{icon} {label}</div><div class="metric-value">{fmt(val, " "+unit)}</div><div class="metric-sub">Current reading</div></div>', unsafe_allow_html=True)
        st.write("")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Sensor values")
            for key, label in [("soil_moisture_raw", "Soil moisture raw"), ("soil_condition", "Soil condition")]:
                st.write(f"**{label}:** {value_or_none(latest_record, key) if value_or_none(latest_record, key) is not None else 'Not provided'}")
            st.write(f"**Timestamp:** {format_time(updated)}")
        with c2:
            st.markdown("#### Reading status")
            for key, label, low, high in [("temperature", "Temperature", 10, 35), ("humidity", "Humidity", 30, 85), ("soil_moisture_percent", "Soil moisture", 20, 80)]:
                v = value_or_none(latest_record, key)
                if v is None:
                    st.write(f"**{label}:** No reading")
                else:
                    st.write(f"**{label}:** {'Within illustrative display band' if low <= float(v) <= high else 'Outside illustrative display band'}")
            st.caption("Display bands are illustrative only, not crop-specific safety thresholds or diagnoses.")
        st.markdown("#### Recent trends")
        if history_df.empty:
            st.info("No historical records are available to plot yet.")
        else:
            c1, c2 = st.columns(2)
            with c1: line_chart(history_df, "temperature", "Temperature trend", "°C")
            with c2: line_chart(history_df, "humidity", "Humidity trend", "%")
            line_chart(history_df, "soil_moisture_percent", "Soil moisture trend", "%")

    elif page == "Environmental Risk":
        st.subheader("Environmental Risk Monitoring")
        color = RISK_COLORS.get(risk, "#94A3B8")
        st.markdown(f'<div class="panel"><div class="small-muted">CURRENT DEVICE-REPORTED ENVIRONMENTAL RISK</div><div style="font-size:2rem;font-weight:800;color:{color};margin-top:8px">{risk}</div><div class="small-muted">This classification is read from sensor_readings/latest.environmental_risk.</div></div>', unsafe_allow_html=True)
        st.markdown("#### Conditions received from device")
        values = {
            "Temperature": fmt(value_or_none(latest_record, "temperature"), " °C"),
            "Humidity": fmt(value_or_none(latest_record, "humidity"), " %"),
            "Soil moisture": fmt(value_or_none(latest_record, "soil_moisture_percent"), " %"),
            "Soil condition": value_or_none(latest_record, "soil_condition") or "Not provided",
        }
        st.table(pd.DataFrame([values]))
        st.markdown("#### Recommended corrective actions")
        for action in risk_actions(risk, value_or_none(latest_record, "recommendation")):
            st.write("• " + action)
        st.caption("Risk categories and recommendations are displayed as supplied by the ESP8266. They are not confirmation that produce has spoiled.")
        st.markdown("#### Previous risk records")
        if history_df.empty or "environmental_risk" not in history_df:
            st.info("No historical environmental-risk records are available.")
        else:
            risk_df = history_df[["timestamp", "environmental_risk"]].dropna().tail(100).sort_values("timestamp", ascending=False)
            st.dataframe(risk_df, use_container_width=True, hide_index=True)

    elif page == "Historical Analytics":
        st.subheader("Historical Data Analytics")
        if history_df.empty:
            st.warning("No historical sensor records were returned from sensor_readings/history.")
        else:
            dated = history_df.dropna(subset=["timestamp"])
            if not dated.empty:
                min_d, max_d = dated["timestamp"].min().date(), dated["timestamp"].max().date()
                d1, d2 = st.columns(2)
                start = d1.date_input("From date", value=min_d, min_value=min_d, max_value=max_d)
                end = d2.date_input("To date", value=max_d, min_value=min_d, max_value=max_d)
                filtered = dated[(dated["timestamp"].dt.date >= start) & (dated["timestamp"].dt.date <= end)] if start <= end else dated.iloc[0:0]
            else:
                st.warning("History records have no parseable timestamp; date filtering is unavailable.")
                filtered = history_df
            st.caption(f"{len(filtered)} records in selected period")
            c1, c2 = st.columns(2)
            with c1: line_chart(filtered, "temperature", "Temperature over time", "°C")
            with c2: line_chart(filtered, "humidity", "Humidity over time", "%")
            line_chart(filtered, "soil_moisture_percent", "Soil moisture over time", "%")
            if "environmental_risk" in filtered and filtered["environmental_risk"].notna().any():
                risk_counts = filtered["environmental_risk"].value_counts().rename_axis("Risk").reset_index(name="Records")
                fig = px.bar(risk_counts, x="Risk", y="Records", color="Risk",
                             color_discrete_map=RISK_COLORS, title="Environmental risk records")
                st.plotly_chart(chart_layout(fig), use_container_width=True)
            numeric_cols = [c for c in ["temperature", "humidity", "soil_moisture_percent"] if c in filtered]
            if len(numeric_cols) >= 2 and filtered[numeric_cols].notna().any().any():
                comp = filtered.melt(id_vars=["timestamp"], value_vars=numeric_cols, var_name="Sensor", value_name="Reading")
                fig = px.line(comp, x="timestamp", y="Reading", color="Sensor", title="Sensor comparison (native units)")
                st.plotly_chart(chart_layout(fig), use_container_width=True)
            st.markdown("#### Recent sensor readings")
            display_cols = [c for c in ["timestamp", "temperature", "humidity", "soil_moisture_percent", "soil_moisture_raw", "soil_condition", "environmental_risk"] if c in filtered]
            st.dataframe(filtered[display_cols].sort_values("timestamp", ascending=False).head(200), use_container_width=True, hide_index=True)
            csv = filtered.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download filtered history as CSV", csv, "post_harvest_sensor_history.csv", "text/csv")
        st.caption("Historical charts use records returned by Firebase; no synthetic readings are added.")

    elif page == "ML Prediction":
        st.subheader("Machine Learning Prediction")
        st.info("The Decision Tree model is not yet integrated. No prediction is generated until you provide a compatible trained model.")
        st.markdown("#### Model integration")
        st.write("Upload a scikit-learn compatible `.joblib` estimator trained to accept features in this order: temperature, humidity, soil_moisture_percent.")
        uploaded = st.file_uploader("Upload trained Decision Tree model (.joblib)", type=["joblib"])
        model = None
        if uploaded is not None:
            try:
                model = joblib.load(uploaded)
                st.success("Model file loaded in this session. Confirm that it is trusted and was trained with the expected feature order.")
            except Exception as exc:
                st.error(f"Could not load model: {exc}")
        elif latest_record and st.secrets.get("ml_model", {}).get("path"):
            st.caption("A model path is configured in Secrets, but Streamlit Cloud deployments should normally package the model securely with the app.")
        required = ["temperature", "humidity", "soil_moisture_percent"]
        inputs = [value_or_none(latest_record, k) for k in required]
        st.markdown("#### Current sensor inputs")
        st.write({k: v for k, v in zip(required, inputs)})
        if model is not None:
            if any(v is None for v in inputs):
                st.warning("Prediction unavailable: one or more required sensor inputs are missing.")
            elif st.button("Run model prediction", type="primary"):
                try:
                    X = pd.DataFrame([[float(v) for v in inputs]], columns=required)
                    prediction = model.predict(X)[0]
                    st.success(f"Model-predicted class: {prediction}")
                    st.write(f"**Prediction timestamp:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
                    st.write("**Explanation:** The class is the output of the uploaded estimator. Add a fitted Decision Tree and class mapping/feature metadata for a more detailed explanation.")
                    st.caption("This is a separate model output and does not replace the device-reported environmental_risk.")
                except Exception as exc:
                    st.error(f"Prediction failed. Check training feature names/order and model compatibility: {exc}")
        else:
            st.markdown('<div class="panel"><b>Integration status: Awaiting model file</b><br><span class="small-muted">Expected artifact: a trusted scikit-learn DecisionTreeClassifier saved with joblib. Ensure training preprocessing, feature order, and class labels match deployment.</span></div>', unsafe_allow_html=True)
        st.markdown("#### Recommended integration checklist")
        st.write("1. Train and validate the Decision Tree in Kaggle.  2. Save the fitted pipeline/estimator with joblib.  3. Confirm the three feature names and units.  4. Upload the artifact here to test inference.")

    elif page == "Alerts & Recommendations":
        st.subheader("Alerts and Recommendations")
        if risk in ["WARNING", "HIGH RISK"]:
            color = RISK_COLORS[risk]
            st.markdown(f'<div class="panel"><span class="pill" style="background:{color}22;color:{color}">CURRENT {risk}</span><h3>Environmental attention required</h3><p>Review the latest sensor conditions and device recommendation. This is an environmental alert, not a confirmed spoilage diagnosis.</p></div>', unsafe_allow_html=True)
        elif risk == "NORMAL":
            st.success("The latest device-reported risk is NORMAL.")
        else:
            st.warning("Current risk status is unavailable or not recognized.")
        st.markdown("#### Current recommendation")
        for action in risk_actions(risk, value_or_none(latest_record, "recommendation")):
            st.write("• " + action)
        st.markdown("#### Previous alerts")
        if history_df.empty or "environmental_risk" not in history_df:
            st.info("No historical alert records are available.")
        else:
            alerts = history_df[history_df["environmental_risk"].isin(["WARNING", "HIGH RISK"])].copy()
            if alerts.empty:
                st.write("No WARNING or HIGH RISK records found in available history.")
            else:
                st.dataframe(alerts.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)
        st.markdown("#### General farmer guidance")
        for item in [
            "Monitor storage temperature and compare it with crop-specific guidance.",
            "Check humidity conditions and investigate persistent changes.",
            "Improve ventilation where appropriate for the stored crop and storage design.",
            "Inspect unusually dry or wet storage environments and verify sensor placement.",
            "When alerts persist, inspect produce and consult local agricultural extension guidance.",
        ]:
            st.write("• " + item)

    elif page == "System Information":
        st.subheader("System Information")
        st.markdown("#### Project")
        st.write("**Title:** Machine Learning-Based System for Predicting and Reducing Post-Harvest Losses Among Farmers")
        st.write("**Objective:** Monitor environmental conditions affecting stored produce, identify potential risk signals, and support farmers with timely information.")
        st.markdown("#### Architecture")
        st.code("DHT11 + YL-69 → ESP8266 NodeMCU → Wi-Fi → Firebase Realtime Database → Streamlit dashboard", language="text")
        st.markdown("#### Hardware and software")
        st.write("- Controller: ESP8266 NodeMCU")
        st.write("- Sensors: DHT11 temperature/relative humidity; YL-69 soil moisture")
        st.write("- Cloud database: Firebase Realtime Database REST API")
        st.write("- Dashboard: Python, Streamlit, Pandas, Plotly, Requests")
        st.write("- Planned ML algorithm: Decision Tree Classification (scikit-learn/joblib)")
        st.markdown("#### Firebase configuration")
        db_url, _ = get_config()
        st.write(f"**Database URL:** `{db_url}`")
        st.write("**Read paths:** `sensor_readings/latest`, `sensor_readings/history`, `system_status`, `sensor_status`")
        st.write("**Connection:** " + ("Reachable" if connected else "Unavailable"))
        st.markdown("#### Integration status")
        st.write(f"- Latest sensor record: {'Available' if isinstance(latest, dict) else 'Missing'}")
        st.write(f"- Historical records: {'Available' if not history_df.empty else 'Missing/empty'}")
        st.write(f"- Device heartbeat: {dev_state}")
        st.write("- ML model: Not integrated unless uploaded on the ML Prediction page.")
        st.caption("The dashboard performs GET requests only. It does not write to Firebase.")

    st.divider()
    st.caption(f"Last dashboard refresh: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} · Refresh interval {REFRESH_SECONDS}s")

dashboard()
