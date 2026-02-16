from __future__ import annotations

import io
import json
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

DB_PATH = Path("app.db")

app = FastAPI(title="Orders Processing Web App")
templates = Jinja2Templates(directory="templates")

REQUIRED_COLUMNS = ["ORDERS", "平台SKU", "TYPE", "COUNTRY", "QTY", "WEIGHT", "POST CODE"]
TYPE_PRIORITY = ["TEA", "FD", "LIQUID", "NO", "YES"]
SESSION_TTL_DAYS = 1


@dataclass
class CalculatorData:
    country_rates: pd.DataFrame
    zone_rates: pd.DataFrame
    postcode_zone: pd.DataFrame


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS calculator_data (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            country_rates_json TEXT NOT NULL,
            zone_rates_json TEXT NOT NULL,
            postcode_zone_json TEXT NOT NULL,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS processing_sessions (
            session_id TEXT PRIMARY KEY,
            dataframe_json TEXT NOT NULL,
            missing_json TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def cleanup_old_sessions() -> None:
    conn = get_conn()
    conn.execute(
        "DELETE FROM processing_sessions WHERE created_at < datetime('now', ?)",
        (f"-{SESSION_TTL_DAYS} day",),
    )
    conn.commit()
    conn.close()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def col_match(df: pd.DataFrame, options: list[str]) -> str:
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for opt in options:
        c = lower_map.get(opt.lower())
        if c:
            return c
    raise ValueError(f"Missing expected column. Tried: {options}")


def optional_col_match(df: pd.DataFrame, options: list[str]) -> str | None:
    try:
        return col_match(df, options)
    except ValueError:
        return None


def load_calculator_from_excel(file_bytes: bytes) -> CalculatorData:
    xl = pd.ExcelFile(io.BytesIO(file_bytes))
    sheet1 = normalize_columns(pd.read_excel(xl, sheet_name=0))
    sheet2 = normalize_columns(pd.read_excel(xl, sheet_name=1))
    sheet3 = normalize_columns(pd.read_excel(xl, sheet_name=2))

    country_rates = standardize_rates(sheet1, key="country")
    zone_rates = standardize_rates(sheet2, key="zone")
    postcode_zone = standardize_postcode_zone(sheet3)

    return CalculatorData(country_rates=country_rates, zone_rates=zone_rates, postcode_zone=postcode_zone)


def standardize_rates(df: pd.DataFrame, key: str) -> pd.DataFrame:
    key_col = col_match(df, [key, key.upper(), key.title()])
    type_col = col_match(df, ["type", "TYPE"])
    ship_col = col_match(df, ["shipping", "shipping rate", "price", "freight"])
    handle_col = col_match(df, ["handling", "handling fee", "service fee"])

    min_col = optional_col_match(df, ["min_weight", "min weight", "weight_from", "from", "min"])
    max_col = optional_col_match(df, ["max_weight", "max weight", "weight_to", "to", "max", "weight"])

    out = pd.DataFrame()
    out["key"] = df[key_col].astype(str).str.strip().str.upper()
    out["type"] = df[type_col].astype(str).str.strip().str.upper()
    out["shipping"] = pd.to_numeric(df[ship_col], errors="coerce")
    out["handling"] = pd.to_numeric(df[handle_col], errors="coerce")
    out["min_weight"] = pd.to_numeric(df[min_col], errors="coerce") if min_col else 0.0
    out["max_weight"] = pd.to_numeric(df[max_col], errors="coerce") if max_col else float("inf")

    out["min_weight"] = out["min_weight"].fillna(0.0)
    out["max_weight"] = out["max_weight"].fillna(float("inf"))
    out = out.dropna(subset=["shipping", "handling"])
    return out


def standardize_postcode_zone(df: pd.DataFrame) -> pd.DataFrame:
    pc_col = col_match(df, ["postcode", "post code", "zip", "postal code"])
    zone_col = col_match(df, ["zone", "shipping zone"])

    out = pd.DataFrame()
    out["postcode"] = (
        df[pc_col]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(" ", "", regex=False)
    )
    out["zone"] = df[zone_col].astype(str).str.strip().str.upper()
    out = out[out["postcode"].ne("")]
    return out.drop_duplicates(subset=["postcode"])


def save_calculator_data(calc: CalculatorData) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO calculator_data (id, country_rates_json, zone_rates_json, postcode_zone_json, updated_at)
        VALUES (1, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(id) DO UPDATE SET
            country_rates_json=excluded.country_rates_json,
            zone_rates_json=excluded.zone_rates_json,
            postcode_zone_json=excluded.postcode_zone_json,
            updated_at=CURRENT_TIMESTAMP
        """,
        (
            calc.country_rates.to_json(orient="split"),
            calc.zone_rates.to_json(orient="split"),
            calc.postcode_zone.to_json(orient="split"),
        ),
    )
    conn.commit()
    conn.close()


def load_calculator_data() -> CalculatorData:
    conn = get_conn()
    row = conn.execute("SELECT * FROM calculator_data WHERE id = 1").fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=400, detail="Calculator.xlsx has not been uploaded yet.")

    return CalculatorData(
        country_rates=pd.read_json(io.StringIO(row["country_rates_json"]), orient="split"),
        zone_rates=pd.read_json(io.StringIO(row["zone_rates_json"]), orient="split"),
        postcode_zone=pd.read_json(io.StringIO(row["postcode_zone_json"]), orient="split"),
    )


def preprocess_orders(df: pd.DataFrame, cutoff_order: str) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"orders.csv missing required columns: {missing}")

    cleaned = df.copy()
    cleaned = cleaned.sort_values("ORDERS", kind="stable")

    cutoff_order = str(cutoff_order).strip()
    if cutoff_order:
        mask = cleaned["ORDERS"].astype(str) >= cutoff_order
        cleaned = cleaned[mask]

    sku = cleaned["平台SKU"].astype(str)
    cleaned = cleaned[~sku.str.contains("-US", na=False)]
    sku = cleaned["平台SKU"].astype(str)
    cleaned = cleaned[sku != "WLN-INSERT1"]

    cleaned["TYPE"] = cleaned["TYPE"].astype(str).str.strip().str.upper()
    cleaned.loc[cleaned["TYPE"].isin(["NYLON", "POLYESTER"]), "TYPE"] = "YES"

    cleaned["QTY"] = pd.to_numeric(cleaned["QTY"], errors="coerce").fillna(0)
    cleaned["WEIGHT"] = pd.to_numeric(cleaned["WEIGHT"], errors="coerce").fillna(0)
    cleaned["COUNTRY"] = cleaned["COUNTRY"].astype(str).str.strip().str.upper()
    cleaned["POST CODE"] = cleaned["POST CODE"].astype(str).str.strip().str.upper().str.replace(" ", "", regex=False)
    cleaned["RowWeight"] = cleaned["QTY"] * cleaned["WEIGHT"]

    return cleaned


def pick_order_type(types: pd.Series) -> str:
    values = set(types.astype(str).str.upper())
    for t in TYPE_PRIORITY:
        if t in values:
            return t
    return "YES"


def find_rate(rates: pd.DataFrame, key: str, typ: str, weight: float) -> tuple[float, float]:
    subset = rates[rates["key"] == key]
    if subset.empty:
        return float("nan"), float("nan")

    typed = subset[subset["type"] == typ]
    if typed.empty:
        typed = subset

    in_range = typed[(typed["min_weight"] <= weight) & (typed["max_weight"] >= weight)]
    if in_range.empty:
        in_range = typed.sort_values("max_weight").tail(1)

    row = in_range.sort_values("max_weight").iloc[0]
    return float(row["shipping"]), float(row["handling"])


def calculate_shipping(cleaned: pd.DataFrame, calc: CalculatorData) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    df = cleaned.copy()
    df["Order Total Weight"] = ""
    df["Shipping"] = ""
    df["Handling"] = ""
    df["Zone"] = ""
    df["Total Value"] = ""

    order_weight_map = df.groupby("ORDERS")["RowWeight"].sum().to_dict()
    order_type_map = df.groupby("ORDERS")["TYPE"].apply(pick_order_type).to_dict()

    postcode_to_zone = dict(zip(calc.postcode_zone["postcode"], calc.postcode_zone["zone"]))

    missing_postcodes: list[dict[str, Any]] = []
    first_idx_by_order = df.groupby("ORDERS").head(1).index

    for idx in first_idx_by_order:
        order = df.at[idx, "ORDERS"]
        country = str(df.at[idx, "COUNTRY"]).upper()
        order_type = order_type_map[order]
        order_weight = float(order_weight_map[order])

        shipping = float("nan")
        handling = float("nan")
        zone = ""

        if country == "AUSTRALIA" and order_type in {"YES", "NO"}:
            postcode = str(df.at[idx, "POST CODE"]).strip().upper().replace(" ", "")
            zone = postcode_to_zone.get(postcode, "")
            if not zone:
                missing_postcodes.append({"index": int(idx), "order": str(order), "postcode": postcode})
                continue
            shipping, handling = find_rate(calc.zone_rates, zone, order_type, order_weight)
        else:
            key_country = country
            available_countries = set(calc.country_rates["key"])
            if key_country not in available_countries:
                key_country = "UNITED STATES"
            shipping, handling = find_rate(calc.country_rates, key_country, order_type, order_weight)

        if pd.isna(shipping) or pd.isna(handling):
            raise HTTPException(
                status_code=400,
                detail=f"No shipping rule found for order {order} (country/zone/type/weight combination missing in Calculator.xlsx).",
            )

        total_value = (((order_weight * shipping) / 1000) + handling) / 7 + 0.9
        df.at[idx, "Order Total Weight"] = round(order_weight, 4)
        df.at[idx, "Shipping"] = round(shipping, 4)
        df.at[idx, "Handling"] = round(handling, 4)
        df.at[idx, "Zone"] = zone
        df.at[idx, "Total Value"] = round(total_value, 4)

    df = df.drop(columns=["RowWeight"])
    return df, missing_postcodes


def save_processing_session(df: pd.DataFrame, missing: list[dict[str, Any]]) -> str:
    session_id = str(uuid.uuid4())
    conn = get_conn()
    conn.execute(
        "INSERT INTO processing_sessions (session_id, dataframe_json, missing_json) VALUES (?, ?, ?)",
        (session_id, df.to_json(orient="split"), json.dumps(missing)),
    )
    conn.commit()
    conn.close()
    return session_id


def load_processing_session(session_id: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    conn = get_conn()
    row = conn.execute("SELECT * FROM processing_sessions WHERE session_id = ?", (session_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Processing session expired or not found")

    df = pd.read_json(io.StringIO(row["dataframe_json"]), orient="split")
    missing = json.loads(row["missing_json"])
    return df, missing


def df_to_excel_response(df: pd.DataFrame) -> StreamingResponse:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="cleaned_orders")
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=cleaned_orders.xlsx"},
    )


@app.on_event("startup")
def startup() -> None:
    init_db()
    cleanup_old_sessions()


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request, "message": None, "missing": None, "session_id": None})


@app.post("/admin/upload-calculator", response_class=HTMLResponse)
async def upload_calculator(request: Request, calculator: UploadFile = File(...)) -> HTMLResponse:
    if not calculator.filename.lower().endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Please upload an Excel file")

    try:
        calc = load_calculator_from_excel(await calculator.read())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    save_calculator_data(calc)

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "message": "Calculator uploaded successfully.", "missing": None, "session_id": None},
    )


@app.post("/process-orders")
async def process_orders(request: Request, orders: UploadFile = File(...), cutoff_order: str = Form("")):
    if not orders.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file")

    calc = load_calculator_data()
    try:
        raw_df = pd.read_csv(io.BytesIO(await orders.read()))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Unable to parse orders.csv") from exc
    cleaned = preprocess_orders(raw_df, cutoff_order=cutoff_order)
    processed, missing = calculate_shipping(cleaned, calc)

    if missing:
        session_id = save_processing_session(cleaned, missing)
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "message": "Some AU postcodes were not found. Please correct them below and recalculate.",
                "missing": missing,
                "session_id": session_id,
            },
        )

    return df_to_excel_response(processed)


@app.post("/process-orders/recalculate")
async def recalculate_orders(request: Request, session_id: str = Form(...)):
    calc = load_calculator_data()
    df, missing = load_processing_session(session_id)

    form = await request.form()
    for entry in missing:
        idx = entry["index"]
        field_name = f"postcode_{idx}"
        corrected = str(form.get(field_name, "")).strip().upper().replace(" ", "")
        if corrected:
            df.at[idx, "POST CODE"] = corrected

    processed, remaining = calculate_shipping(df, calc)
    if remaining:
        new_session = save_processing_session(df, remaining)
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "message": "Some postcodes are still invalid. Please update and try again.",
                "missing": remaining,
                "session_id": new_session,
            },
        )

    return df_to_excel_response(processed)
