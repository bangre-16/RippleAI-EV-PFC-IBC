from functools import lru_cache
from io import BytesIO
from datetime import datetime
import base64
import html

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from flask import Flask, jsonify, render_template, request, send_file, abort

BASE = Path(__file__).resolve().parent
PROJECT = BASE / "Major Project"
TEMPLATE_FOLDER = BASE / "app" / "templates"
STATIC_FOLDER = BASE / "app" / "static"
DOWNLOAD_FOLDER = STATIC_FOLDER / "downloads"
FRONTEND_DIST = BASE / "frontend" / "dist"

app = Flask(__name__, template_folder=str(TEMPLATE_FOLDER), static_folder=str(STATIC_FOLDER))

FEATURES = [
    "Switching_Freq_Hz", "Load_Resistance_Ohm", "Vin_RMS_V",
    "Duty_Cycle_Pct", "Inductance_uH", "Vout_DC_V"
]
LABELS = {
    "Switching_Freq_Hz": "Switching Frequency",
    "Load_Resistance_Ohm": "Load Resistance",
    "Vin_RMS_V": "Vin RMS",
    "Duty_Cycle_Pct": "Duty Cycle",
    "Inductance_uH": "Inductance",
    "Vout_DC_V": "DC Output Voltage",
}
UNITS = {
    "Switching_Freq_Hz": "Hz",
    "Load_Resistance_Ohm": "Ω",
    "Vin_RMS_V": "V RMS",
    "Duty_Cycle_Pct": "%",
    "Inductance_uH": "µH",
    "Vout_DC_V": "V",
}

MODEL_CONFIG = {
    "SVR": {
        "baseline": {"folder": "SVR", "model": "SVR_model.pkl", "scaler": "SVR_scaler.pkl", "graph": "svr_baseline.png"},
        "tuned": {"folder": "SVR Tuned", "model": "SVR_Tuned_model.pkl", "scaler": None, "graph": "svr_tuned.png"},
        "description": "Support Vector Regression learns a nonlinear mapping between converter operating conditions and output current ripple.",
    },
    "Gaussian Process": {
        "baseline": {"folder": "Gaussian Process", "model": "GPR_model.pkl", "scaler": "GPR_scaler.pkl", "graph": "gpr_baseline.png"},
        "tuned": {"folder": "Gaussian Process Tuned", "model": "GPR_Tuned_model.pkl", "scaler": "GPR_Tuned_scaler.pkl", "graph": "gpr_tuned.png"},
        "description": "Gaussian Process Regression models the ripple response with a kernel-based probabilistic function prior.",
    },
    "Random Forest": {
        "baseline": {"folder": "Random Forest", "model": "RandomForest_model.pkl", "scaler": None, "graph": "random_forest_baseline.png"},
        "tuned": {"folder": "RF Tuned", "model": "RandomForest_Tuned_model.pkl", "scaler": None, "graph": "random_forest_tuned.png"},
        "description": "Random Forest uses an ensemble of decision trees to capture nonlinear parameter interactions.",
    },
    "XGBoost": {
        "baseline": {"folder": "XGBoost", "model": "XGBoost_model.pkl", "scaler": None, "graph": "xgboost_baseline.png"},
        "tuned": {"folder": "XGBoost Tuned", "model": "XGBoost_Tuned_model.pkl", "scaler": None, "graph": "xgboost_tuned.png"},
        "description": "XGBoost builds boosted regression trees and learns nonlinear relationships through sequential residual correction.",
    },
    "KNN": {
        "baseline": {"folder": "KNN", "model": "KNN_model.pkl", "scaler": "KNN_scaler.pkl", "graph": "knn_baseline.png"},
        "tuned": {"folder": "KNN Tuned", "model": "KNN_Tuned_model.pkl", "scaler": None, "graph": "knn_tuned.png"},
        "description": "KNN estimates ripple from nearby operating points in the normalized feature space.",
    },
    "Stacked Ensemble": {
        "baseline": {"folder": "Stacked Meta Ensemble", "model": "Stacked_Ensemble_model.pkl", "scaler": None, "graph": "stacked_ensemble_baseline.png"},
        "tuned": {"folder": "Stacked Meta Ensemble Tuned", "model": "Stacked_Ensemble_model.pkl", "scaler": None, "graph": "stacked_ensemble_tuned.png"},
        "description": "A stacked meta-model combines SVR, GPR and Random Forest outputs through a learned ridge layer.",
    },
}

BASELINE_INFO = {
    "SVR": "RBF kernel · C=100 · gamma=scale · epsilon=0.001",
    "Gaussian Process": "Constant × RBF + WhiteKernel · 2 optimizer restarts",
    "Random Forest": "200 trees · unlimited depth · min split=2 · min leaf=1",
    "XGBoost": "200 trees · depth=5 · learning rate=0.05 · subsample=0.9",
    "KNN": "k=5 · distance weighting · Euclidean metric",
    "Stacked Ensemble": "SVR + GPR + Random Forest · Ridge alpha=1.0",
}

ALIAS = {
    "SVR": "SVR",
    "Gaussian Process": "GPR",
    "Random Forest": "Random Forest",
    "XGBoost": "XGBoost",
    "KNN": "KNN",
    "Stacked Ensemble": "Stacked Ensemble",
}

META_PATH = BASE / "website_data.json"
META = json.loads(META_PATH.read_text(encoding="utf-8"))
METRICS = META["metrics"]
HYPERPARAMETERS = META["hyper"]
IMPORTANCE = META["importance"]
MODEL_ZIPS = META["model_zips"]


def data_key(name):
    """Return the canonical key used by website_data.json."""
    return ALIAS.get(name, name)


def dataset_info():
    df = pd.read_csv(PROJECT / "Dataset" / "dataset.csv")
    ranges = {
        c: {"min": float(df[c].min()), "max": float(df[c].max())}
        for c in FEATURES + ["Current_Ripple_A"] if c in df.columns
    }
    return df, ranges


@lru_cache(maxsize=None)
def load_model(name, variant):
    cfg = MODEL_CONFIG[name][variant]
    model = joblib.load(PROJECT / cfg["folder"] / cfg["model"])
    scaler = None
    if cfg["scaler"]:
        scaler = joblib.load(PROJECT / cfg["folder"] / cfg["scaler"])
    return model, scaler


def predict_variant(name, variant, values):
    model, scaler = load_model(name, variant)
    raw = np.array([[values[f] for f in FEATURES]], dtype=float)

    if name == "Stacked Ensemble":
        bundle = model
        if "svr" in bundle:
            svr = bundle["svr"].predict(raw)
            gpr = bundle["gpr"].predict(bundle["gpr_scaler"].transform(raw))
            rf = bundle["random_forest"].predict(raw)
            return float(bundle["meta_model"].predict(np.column_stack([svr, gpr, rf]))[0])
        svr = bundle["SVR"].predict(raw)
        gpr = bundle["GPR"].predict(raw)
        rf = bundle["RandomForest"].predict(raw)
        return float(bundle["MetaModel"].predict(np.column_stack([svr, gpr, rf]))[0])

    x = scaler.transform(raw) if scaler is not None else raw
    return float(model.predict(x)[0])


def model_rows():
    rows = []
    for name in MODEL_CONFIG:
        key = data_key(name)
        metrics = METRICS[key]
        hyper = HYPERPARAMETERS[key]
        rows.append({
            "name": name,
            "short": ALIAS[key] if key in ALIAS else key,
            "description": MODEL_CONFIG[name]["description"],
            "baseline": metrics["baseline"],
            "tuned": metrics["tuned"],
            "baseline_graph": MODEL_CONFIG[name]["baseline"]["graph"],
            "tuned_graph": MODEL_CONFIG[name]["tuned"]["graph"],
            "baseline_download": f"/static/downloads/{MODEL_ZIPS[f'{key}|baseline']}",
            "tuned_download": f"/static/downloads/{MODEL_ZIPS[f'{key}|tuned']}",
            "baseline_info": BASELINE_INFO[name],
            "tuned_info": hyper["tuned"],
            "r2_delta": metrics["tuned"].get("R2", 0) - metrics["baseline"].get("R2", 0),
            "rmse_delta": metrics["tuned"].get("RMSE", 0) - metrics["baseline"].get("RMSE", 0),
        })
    return rows




def validate_values(values, ranges):
    """Return human-readable training-range warnings for one operating point."""
    warnings = []
    for feature, value in values.items():
        minimum = ranges[feature]["min"]
        maximum = ranges[feature]["max"]
        if value < minimum or value > maximum:
            warnings.append(
                f"{LABELS[feature]} is outside the training-data range "
                f"({minimum:.2f} - {maximum:.2f} {UNITS[feature]}). "
                f"Prediction reliability may be reduced."
            )
    return warnings


def recommendation_name():
    """Return the model selected from stored tuned-model R2 values."""
    ranked = []
    for name in MODEL_CONFIG:
        r2 = METRICS[data_key(name)]["tuned"].get("R2")
        if r2 is not None:
            ranked.append((name, r2))
    return max(ranked, key=lambda x: x[1])[0] if ranked else None


def predict_many(df):
    """Predict all rows in a dataframe using all saved tuned models."""
    result = df[FEATURES].copy()
    _, ranges = dataset_info()

    range_status = []
    out_of_range_cells = 0
    for _, row in df[FEATURES].iterrows():
        row_bad = False
        for feature in FEATURES:
            value = float(row[feature])
            if value < ranges[feature]["min"] or value > ranges[feature]["max"]:
                row_bad = True
                out_of_range_cells += 1
        range_status.append("Outside training range" if row_bad else "Within training range")

    result["Range_Status"] = range_status

    for name in MODEL_CONFIG:
        preds = []
        for _, row in df[FEATURES].iterrows():
            values = {f: float(row[f]) for f in FEATURES}
            try:
                preds.append(predict_variant(name, "tuned", values))
            except Exception:
                preds.append(np.nan)
        result[f"{name}_Prediction_A"] = preds

    return result, out_of_range_cells


def csv_bytes_from_records(records, columns):
    """Build a CSV payload for a batch result download."""
    buf = BytesIO()
    text = buf
    # csv requires text, so build separately and encode explicitly.
    import io
    sio = io.StringIO()
    writer = csv.DictWriter(sio, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in records:
        writer.writerow(row)
    return sio.getvalue().encode("utf-8")


def _pdf_table(data, col_widths=None, header=True):
    table = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d8dde1")),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e7f4f8")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111820")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
    table.setStyle(TableStyle(style))
    return table


def generate_prediction_report(payload):
    """Create a polished PDF report in memory and return its bytes."""
    values = payload.get("inputs", {})
    predictions = payload.get("predictions", {})
    warnings = payload.get("warnings", [])
    recommended = payload.get("recommended")

    # Build prediction comparison chart.
    chart_buf = BytesIO()
    names = [n for n in MODEL_CONFIG if predictions.get(n) is not None]
    vals = [float(predictions[n]) for n in names]
    fig, ax = plt.subplots(figsize=(8.4, 3.2))
    ax.bar(names, vals)
    ax.set_ylabel("Predicted current ripple (A)")
    ax.set_title("Model Prediction Comparison")
    ax.tick_params(axis="x", labelrotation=22, labelsize=8)
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(chart_buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    chart_buf.seek(0)

    # Styles.
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=19,
        textColor=colors.HexColor("#10161b"), spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle", parent=styles["Normal"], fontSize=8.5,
        textColor=colors.HexColor("#5d6972"), spaceAfter=10
    )
    h2 = ParagraphStyle(
        "H2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12,
        textColor=colors.HexColor("#111820"), spaceBefore=10, spaceAfter=7
    )
    body = ParagraphStyle(
        "Body", parent=styles["BodyText"], fontSize=8.5, leading=12,
        textColor=colors.HexColor("#3f4b53"), spaceAfter=6
    )
    note = ParagraphStyle(
        "Note", parent=body, fontSize=7.7, leading=10.5, textColor=colors.HexColor("#65727a")
    )

    story = []
    story.append(Paragraph("RIPPLEAI - Prediction Report", title_style))
    story.append(Paragraph(
        f"EV PFC-IBC Output Current Ripple Estimation | Generated {datetime.now().strftime('%d %b %Y, %H:%M')}",
        subtitle_style
    ))
    story.append(Paragraph(
        "This report uses the saved tuned machine-learning models and the operating point supplied in Prediction Studio. No model is retrained for this query.", body
    ))

    story.append(Paragraph("1. Converter Operating Point", h2))
    input_table = [["Parameter", "Value", "Unit"]]
    for f in FEATURES:
        input_table.append([LABELS[f], f"{float(values[f]):.4f}", UNITS[f]])
    story.append(_pdf_table(input_table, [82*mm, 35*mm, 35*mm]))

    story.append(Paragraph("2. Model Predictions", h2))
    pred_table = [["Model", "Predicted Current Ripple (A)"]]
    for n in MODEL_CONFIG:
        v = predictions.get(n)
        pred_table.append([n, "Unavailable" if v is None else f"{float(v):.6f}"])
    story.append(_pdf_table(pred_table, [95*mm, 57*mm]))

    if recommended:
        story.append(Spacer(1, 4))
        story.append(Paragraph(
            f"<b>Stored-performance recommendation:</b> {html.escape(recommended)}. This is based on the saved tuned-model evaluation performance and not on the unknown true ripple value.",
            body
        ))

    story.append(Paragraph("3. Training-Range Status", h2))
    if warnings:
        story.append(Paragraph("The following inputs are outside the dataset operating range:", body))
        warn_rows = [["Status / Warning"]] + [[w] for w in warnings]
        story.append(_pdf_table(warn_rows, [152*mm]))
    else:
        story.append(Paragraph(
            "All six inputs are within the training-data ranges available to the prediction system.", body
        ))

    story.append(Paragraph("4. Prediction Comparison", h2))
    story.append(RLImage(chart_buf, width=170*mm, height=65*mm))
    story.append(Paragraph(
        "The bar chart compares the predicted current-ripple values returned by the saved tuned models for this operating point.", note
    ))

    story.append(Paragraph("5. Feature Influence Context", h2))
    story.append(Paragraph(
        "The website's feature-influence analysis is model-based and indicates relative sensitivity of predictions to the six input parameters; it should not be interpreted as a direct physical percentage contribution to converter ripple.", note
    ))

    # Include tuned feature-influence top contributor for each model.
    fi_rows = [["Model", "Most influential input", "Relative influence"]]
    for name in MODEL_CONFIG:
        key = data_key(name)
        tuned = sorted(IMPORTANCE[key]["tuned"], key=lambda x: x["importance"], reverse=True)
        if tuned:
            fi_rows.append([name, tuned[0]["label"], f"{float(tuned[0]['importance']):.2f}%"])
    story.append(_pdf_table(fi_rows, [75*mm, 60*mm, 35*mm]))

    pdf_buf = BytesIO()
    doc = SimpleDocTemplate(
        pdf_buf, pagesize=A4, rightMargin=16*mm, leftMargin=16*mm,
        topMargin=15*mm, bottomMargin=15*mm,
        title="RippleAI Prediction Report", author="RippleAI"
    )
    doc.build(story)
    pdf_buf.seek(0)
    return pdf_buf.getvalue()


def generate_batch_prediction_report(payload):
    """Create a PDF containing the complete uploaded batch results."""
    csv_b64 = payload.get("csv_base64")
    if not csv_b64:
        raise ValueError("Batch report data is missing.")

    try:
        csv_text = base64.b64decode(csv_b64.encode("ascii")).decode("utf-8")
        result_df = pd.read_csv(BytesIO(csv_text.encode("utf-8")))
    except Exception as exc:
        raise ValueError(f"Could not reconstruct batch results for the report: {exc}") from exc

    if result_df.empty:
        raise ValueError("There are no batch results to include in the report.")

    recommended = payload.get("recommended")
    out_of_range_cells = int(payload.get("out_of_range_cells", 0))

    # Identify prediction columns in the same model order as the website.
    prediction_cols = [
        f"{name}_Prediction_A"
        for name in MODEL_CONFIG
        if f"{name}_Prediction_A" in result_df.columns
    ]

    # Summary chart: mean predicted ripple across the uploaded batch.
    chart_buf = BytesIO()
    chart_names = [c.replace("_Prediction_A", "") for c in prediction_cols]
    chart_values = [float(pd.to_numeric(result_df[c], errors="coerce").mean()) for c in prediction_cols]
    fig, ax = plt.subplots(figsize=(9.4, 3.4))
    ax.bar(chart_names, chart_values)
    ax.set_ylabel("Mean predicted current ripple (A)")
    ax.set_title("Batch Prediction Model Comparison")
    ax.tick_params(axis="x", labelrotation=20, labelsize=8)
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(chart_buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    chart_buf.seek(0)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "BatchReportTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=18,
        textColor=colors.HexColor("#10161b"), spaceAfter=5
    )
    subtitle_style = ParagraphStyle(
        "BatchReportSubtitle", parent=styles["Normal"], fontSize=8.2,
        textColor=colors.HexColor("#5d6972"), spaceAfter=9
    )
    h2 = ParagraphStyle(
        "BatchH2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=11.5,
        textColor=colors.HexColor("#111820"), spaceBefore=9, spaceAfter=6
    )
    body = ParagraphStyle(
        "BatchBody", parent=styles["BodyText"], fontSize=8.1, leading=11,
        textColor=colors.HexColor("#3f4b53"), spaceAfter=5
    )
    small = ParagraphStyle(
        "BatchSmall", parent=body, fontSize=6.8, leading=8.5,
        textColor=colors.HexColor("#65727a")
    )
    table_header = ParagraphStyle(
        "BatchTableHeader", parent=body, fontName="Helvetica-Bold", fontSize=5.8,
        leading=6.8, textColor=colors.HexColor("#111820")
    )
    table_cell = ParagraphStyle(
        "BatchTableCell", parent=body, fontSize=5.8, leading=6.7,
        textColor=colors.HexColor("#313a40")
    )

    story = []
    story.append(Paragraph("RIPPLEAI - Batch Prediction Report", title_style))
    story.append(Paragraph(
        f"EV PFC-IBC Output Current Ripple Estimation | Generated {datetime.now().strftime('%d %b %Y, %H:%M')} | {len(result_df)} operating points",
        subtitle_style
    ))
    story.append(Paragraph(
        "The report contains the complete batch output generated from the uploaded CSV using the saved tuned machine-learning models. No model is retrained for the uploaded batch.",
        body
    ))

    story.append(Paragraph("1. Batch Summary", h2))
    summary_rows = [
        ["Operating points processed", str(len(result_df))],
        ["Out-of-range cells", str(out_of_range_cells)],
        ["Saved-performance model", recommended or "N/A"],
    ]
    story.append(_pdf_table([["Item", "Value"]] + summary_rows, [75*mm, 45*mm]))

    story.append(Paragraph("2. Model Prediction Summary", h2))
    stat_rows = [["Model", "Mean (A)", "Minimum (A)", "Maximum (A)"]]
    for col in prediction_cols:
        model_name = col.replace("_Prediction_A", "")
        vals = pd.to_numeric(result_df[col], errors="coerce").dropna()
        if len(vals):
            stat_rows.append([model_name, f"{vals.mean():.6f}", f"{vals.min():.6f}", f"{vals.max():.6f}"])
    story.append(_pdf_table(stat_rows, [65*mm, 35*mm, 35*mm, 35*mm]))
    story.append(Spacer(1, 3))
    story.append(RLImage(chart_buf, width=245*mm, height=88*mm))
    story.append(Paragraph(
        "The chart shows the mean predicted current-ripple value across all uploaded operating points for each saved tuned model.", small
    ))

    story.append(Paragraph("3. Complete Batch Results", h2))
    label_map = {
        "Switching_Freq_Hz": "Switching Freq (Hz)",
        "Load_Resistance_Ohm": "Load R (Ω)",
        "Vin_RMS_V": "Vin RMS (V)",
        "Duty_Cycle_Pct": "Duty (%)",
        "Inductance_uH": "Inductance (µH)",
        "Vout_DC_V": "Vout DC (V)",
        "Range_Status": "Range status",
    }
    columns = [c for c in FEATURES + ["Range_Status"] if c in result_df.columns] + prediction_cols
    header = [Paragraph(label_map.get(c, c.replace("_Prediction_A", "").replace("_", " ")), table_header) for c in columns]
    table_data = [header]
    for idx, row in result_df.iterrows():
        rendered = []
        for c in columns:
            value = row[c]
            if pd.isna(value):
                text_value = "—"
            elif c in prediction_cols or c in FEATURES:
                text_value = f"{float(value):.6f}"
            else:
                text_value = str(value)
            rendered.append(Paragraph(html.escape(text_value), table_cell))
        table_data.append(rendered)

    # Landscape A4 gives enough width for all converter inputs + six model outputs.
    widths = [14*mm, 18*mm, 16*mm, 16*mm, 15*mm, 17*mm, 20*mm, 20*mm, 18*mm, 18*mm, 18*mm, 18*mm, 18*mm, 18*mm]
    widths = widths[:len(columns)]
    # Normalize widths if any unforeseen column count appears.
    total = sum(widths)
    if total > 258*mm:
        scale = (258*mm) / total
        widths = [w * scale for w in widths]
    full_table = Table(table_data, colWidths=widths, repeatRows=1, splitByRow=1)
    full_table.setStyle(TableStyle([
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 5.8),
        ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#cfd6db")),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e7f4f8")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor("#111820")),
        ("LEFTPADDING", (0,0), (-1,-1), 2),
        ("RIGHTPADDING", (0,0), (-1,-1), 2),
        ("TOPPADDING", (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
    ]))
    story.append(full_table)

    story.append(PageBreak())
    story.append(Paragraph("4. Feature Influence Context", h2))
    story.append(Paragraph(
        "Feature influence is model-based and describes relative influence on predictions; it should not be interpreted as a direct physical percentage contribution to converter ripple.",
        body
    ))
    fi_rows = [["Model", "Most influential input", "Relative influence"]]
    for name in MODEL_CONFIG:
        key = data_key(name)
        tuned = sorted(IMPORTANCE[key]["tuned"], key=lambda x: x["importance"], reverse=True)
        if tuned:
            fi_rows.append([name, tuned[0]["label"], f"{float(tuned[0]['importance']):.2f}%"])
    story.append(_pdf_table(fi_rows, [75*mm, 75*mm, 45*mm]))

    story.append(Paragraph("5. Interpretation", h2))
    story.append(Paragraph(
        "The batch module is intended to evaluate multiple converter operating points efficiently. Values marked outside the training range represent extrapolation relative to the dataset used by the saved models, so those predictions should be interpreted with additional caution.",
        body
    ))

    pdf_buf = BytesIO()
    doc = SimpleDocTemplate(
        pdf_buf, pagesize=landscape(A4), rightMargin=13*mm, leftMargin=13*mm,
        topMargin=12*mm, bottomMargin=12*mm,
        title="RippleAI Batch Prediction Report", author="RippleAI"
    )
    doc.build(story)
    pdf_buf.seek(0)
    return pdf_buf.getvalue()


@app.context_processor
def inject_common():
    return {"units": UNITS, "labels": LABELS}


@app.route("/api/site-data")
def api_site_data():
    """Expose the saved project metadata needed by the React UI."""
    df, ranges = dataset_info()
    image_map = {}
    for name in MODEL_CONFIG:
        image_map[f"{name}|baseline"] = MODEL_CONFIG[name]["baseline"]["graph"]
        image_map[f"{name}|tuned"] = MODEL_CONFIG[name]["tuned"]["graph"]
    return jsonify({
        "features": FEATURES,
        "labels": LABELS,
        "units": UNITS,
        "rows": len(df),
        "columns": list(df.columns),
        "ranges": ranges,
        "metrics": METRICS,
        "hyper": HYPERPARAMETERS,
        "importance": IMPORTANCE,
        "images": image_map,
        "model_zips": MODEL_ZIPS,
        "simulation": {
            "current": "/static/images/simulation_current.jpeg",
            "voltage": "/static/images/simulation_voltage.jpeg",
            "current_download": "/static/downloads/Simulation_Output_Current.jpeg",
            "voltage_download": "/static/downloads/Simulation_Output_Voltage.jpeg",
            "dataset_download": "/static/downloads/dataset.csv",
            "dataset_generation": "/static/downloads/dataset_generation.m",
            "slx": "/static/downloads/PFC_IBC_AC_Model.slx",
            "slxc": "/static/downloads/PFC_IBC_AC_Model.slxc",
        },
        "template_download": "/static/downloads/Batch_Prediction_Template.csv",
    })


@app.route("/assets/<path:filename>")
def react_assets(filename):
    assets_dir = FRONTEND_DIST / "assets"
    if not assets_dir.exists():
        return abort(404)
    return send_file(assets_dir / filename)


def serve_react(path: str = ""):
    index = FRONTEND_DIST / "index.html"
    if index.exists():
        return send_file(index)
    return jsonify({
        "error": "React UI is not built yet.",
        "next_step": "cd frontend && npm install && npm run build"
    }), 503


# React SPA routes.
for _route in ["/", "/dataset", "/models", "/feature-importance", "/results", "/prediction", "/project-files"]:
    app.add_url_rule(
        _route,
        endpoint=f"react_{_route.strip('/').replace('/', '_') or 'home'}",
        view_func=serve_react,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        payload = request.get_json(force=True) or {}
        missing = [f for f in FEATURES if f not in payload]
        if missing:
            return jsonify({"success": False, "error": "Missing input parameters", "missing": missing}), 400

        values = {f: float(payload[f]) for f in FEATURES}
        _, ranges = dataset_info()
        warnings = []
        for f, v in values.items():
            if v < ranges[f]["min"] or v > ranges[f]["max"]:
                warnings.append(
                    f"{LABELS[f]} is outside the training-data range "
                    f"({ranges[f]['min']:.2f} – {ranges[f]['max']:.2f}). "
                    f"Prediction reliability may be reduced."
                )

        predictions = {}
        errors = {}
        for name in MODEL_CONFIG:
            try:
                predictions[name] = predict_variant(name, "tuned", values)
            except Exception as exc:
                predictions[name] = None
                errors[name] = str(exc)

        ranked = [
            (n, METRICS[data_key(n)]["tuned"].get("R2"))
            for n, pred in predictions.items()
            if pred is not None and METRICS[data_key(n)]["tuned"].get("R2") is not None
        ]
        recommended = max(ranked, key=lambda x: x[1])[0] if ranked else None

        return jsonify({
            "success": True,
            "inputs": values,
            "predictions": predictions,
            "warnings": warnings,
            "recommended": recommended,
            "errors": errors,
        })
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500



@app.route("/api/batch-predict", methods=["POST"])
def api_batch_predict():
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "Please upload a CSV file."}), 400

        upload = request.files["file"]
        if not upload.filename or not upload.filename.lower().endswith(".csv"):
            return jsonify({"success": False, "error": "Only CSV files are supported."}), 400

        upload.stream.seek(0, 2)
        size = upload.stream.tell()
        upload.stream.seek(0)
        if size > 5 * 1024 * 1024:
            return jsonify({"success": False, "error": "CSV file is too large. Maximum size is 5 MB."}), 400

        try:
            df = pd.read_csv(upload)
        except Exception as exc:
            return jsonify({"success": False, "error": f"Could not read CSV: {exc}"}), 400

        if df.empty:
            return jsonify({"success": False, "error": "The uploaded CSV contains no rows."}), 400
        if len(df) > 5000:
            return jsonify({"success": False, "error": "Maximum 5,000 rows are supported per batch."}), 400

        missing = [f for f in FEATURES if f not in df.columns]
        if missing:
            return jsonify({
                "success": False,
                "error": "The CSV is missing required input columns.",
                "missing": missing
            }), 400

        work_df = df[FEATURES].copy()
        for feature in FEATURES:
            work_df[feature] = pd.to_numeric(work_df[feature], errors="coerce")

        invalid_counts = work_df[FEATURES].isna().sum().to_dict()
        invalid_rows = int(work_df[FEATURES].isna().any(axis=1).sum())
        if invalid_rows:
            bad_features = [f for f, c in invalid_counts.items() if c]
            return jsonify({
                "success": False,
                "error": "Some required input values are missing or non-numeric.",
                "invalid_rows": invalid_rows,
                "invalid_columns": bad_features,
                "invalid_counts": invalid_counts
            }), 400

        result_df, out_of_range_cells = predict_many(work_df)
        records = result_df.round(6).to_dict(orient="records")
        columns = list(result_df.columns)

        # Create a downloadable CSV payload entirely in memory.
        csv_text = result_df.to_csv(index=False)
        csv_b64 = base64.b64encode(csv_text.encode("utf-8")).decode("ascii")

        return jsonify({
            "success": True,
            "rows": records,
            "columns": columns,
            "row_count": len(records),
            "out_of_range_cells": out_of_range_cells,
            "recommended": recommendation_name(),
            "csv_base64": csv_b64,
            "download_name": "rippleai_batch_predictions.csv"
        })
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/generate-report", methods=["POST"])
def api_generate_report():
    try:
        payload = request.get_json(force=True) or {}
        report_type = payload.get("report_type", "single")

        if report_type == "batch":
            pdf_bytes = generate_batch_prediction_report(payload)
            filename = "RippleAI_Batch_Prediction_Report.pdf"
        else:
            required = ["inputs", "predictions", "warnings"]
            missing = [k for k in required if k not in payload]
            if missing:
                return jsonify({"success": False, "error": f"Missing report fields: {', '.join(missing)}"}), 400
            pdf_bytes = generate_prediction_report(payload)
            filename = "RippleAI_Prediction_Report.pdf"

        return send_file(
            BytesIO(pdf_bytes),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=filename
        )
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/metrics")
def api_metrics():
    return jsonify(METRICS)


@app.route("/<path:path>")
def react_catch_all(path):
    if path.startswith("api/") or path.startswith("static/"):
        return abort(404)
    return serve_react(path)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
