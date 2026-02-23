from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
import secrets
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware


# =========================================================
# Paths
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "app.db"
TEMPLATES_DIR = BASE_DIR / "templates"
DOWNLOADS_DIR = BASE_DIR / "downloads"
DOWNLOADS_DIR.mkdir(exist_ok=True)

# Where uploaded calculator will be stored (and used for calculations)
CALCULATOR_PATH = BASE_DIR / "Calculator.xlsx"


# =========================================================
# App & Templates
# =========================================================
app = FastAPI(title="Orders Processing App")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# =========================================================
# Sessions (cookie-based)
# =========================================================
SESSION_SECRET = os.environ.get("SESSION_SECRET") or secrets.token_urlsafe(32)
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    session_cookie="ops_session",
    https_only=False,  # set True if you want strict HTTPS-only cookies
    same_site="lax",
)


# =========================================================
# Database helpers
# =========================================================
def db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db() -> None:
    with db() as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                pass_salt TEXT NOT NULL,
                pass_hash TEXT NOT NULL,
                is_admin INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS processing_sessions (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                cleaned_json TEXT NOT NULL,
                missing_json TEXT NOT NULL
            )
            """
        )
        con.commit()

    # Ensure default admin exists (admin / admin123)
    if not get_user_by_username("admin"):
        create_user("admin", "admin123", is_admin=True)


# =========================================================
# Password hashing (PBKDF2)
# =========================================================
def hash_password(password: str, salt_hex: str) -> str:
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt_hex),
        120_000,
    )
    return dk.hex()


def verify_password(password: str, salt_hex: str, stored_hash_hex: str) -> bool:
    calc = hash_password(password, salt_hex)
    return hmac.compare_digest(calc, stored_hash_hex)


# =========================================================
# User CRUD
# =========================================================
def get_user_by_id(user_id: int) -> Optional[dict]:
    with db() as con:
        row = con.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    return dict(row) if row else None


def get_user_by_username(username: str) -> Optional[dict]:
    with db() as con:
        row = con.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    return dict(row) if row else None


def list_users() -> list[dict]:
    with db() as con:
        rows = con.execute("SELECT * FROM users ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]


def create_user(username: str, password: str, is_admin: bool = False) -> None:
    username = (username or "").strip()
    if not username:
        raise ValueError("Username cannot be blank.")
    if len(password) < 4:
        raise ValueError("Password must be at least 4 characters.")

    salt_hex = secrets.token_bytes(16).hex()
    pass_hash_hex = hash_password(password, salt_hex)

    with db() as con:
        con.execute(
            """
            INSERT INTO users(username, pass_salt, pass_hash, is_admin, created_at)
            VALUES(?,?,?,?,?)
            """,
            (
                username,
                salt_hex,
                pass_hash_hex,
                1 if is_admin else 0,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        con.commit()


def set_user_password(user_id: int, new_password: str) -> None:
    if len(new_password) < 4:
        raise ValueError("Password must be at least 4 characters.")
    salt_hex = secrets.token_bytes(16).hex()
    pass_hash_hex = hash_password(new_password, salt_hex)
    with db() as con:
        con.execute(
            "UPDATE users SET pass_salt=?, pass_hash=? WHERE id=?",
            (salt_hex, pass_hash_hex, user_id),
        )
        con.commit()


def delete_user(user_id: int) -> None:
    with db() as con:
        con.execute("DELETE FROM users WHERE id=?", (user_id,))
        con.commit()


# =========================================================
# Auth helpers
# =========================================================
def current_user(request: Request) -> Optional[dict]:
    uid = request.session.get("user_id")
    if not uid:
        return None
    try:
        return get_user_by_id(int(uid))
    except Exception:
        return None


def require_login(request: Request) -> dict:
    user = current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not logged in")
    return user


def require_admin(request: Request) -> dict:
    user = require_login(request)
    if not user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin only")
    return user


# =========================================================
# Data standardization helpers
# =========================================================
REQUIRED_ORDER_COLUMNS = [
    "DATE",
    "ORDERS",
    "ITEM",
    "SKU",
    "平台SKU",
    "NAME",
    "QTY",
    "COUNTRY",
    "DECLARED VALUE",
    "tax D. Value",
    "TYPE",
    "PRICE",
    "WEIGHT",
    "POST CODE",
]


def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def normalize_country(x: Any) -> str:
    return normalize_text(x).upper()


def normalize_type(x: Any) -> str:
    t = normalize_text(x).upper()
    # allow YES/NO/TEA only for this case
    if t in {"YES", "NO", "TEA"}:
        return t
    return t


def normalize_postcode(x: Any) -> str:
    s = normalize_text(x)
    # keep only digits (common for AU postcodes)
    digits = "".join(ch for ch in s if ch.isdigit())
    return digits


def sku_is_none(sku: Any) -> bool:
    s = normalize_text(sku).upper()
    return s.startswith("NONE")


# =========================================================
# Calculator loading & mapping
# =========================================================
@dataclass
class CalculatorData:
    rates: pd.DataFrame  # Sheet2
    postcode_zone: pd.DataFrame  # Sheet3


def read_calculator_or_fail() -> CalculatorData:
    if not CALCULATOR_PATH.exists():
        raise HTTPException(status_code=400, detail="Calculator.xlsx not found. Please upload it first.")

    try:
        xls = pd.ExcelFile(CALCULATOR_PATH)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read Calculator.xlsx: {e}")

    # Sheet2 = rates table
    try:
        rates_raw = pd.read_excel(xls, sheet_name="Sheet2")
    except Exception:
        # fallback to second sheet index
        rates_raw = pd.read_excel(xls, sheet_name=1)

    # Sheet3 = postcode->zone
    try:
        pc_raw = pd.read_excel(xls, sheet_name="Sheet3")
    except Exception:
        # fallback to third sheet index
        pc_raw = pd.read_excel(xls, sheet_name=2)

    rates = standardize_rates(rates_raw)
    pc = standardize_postcode_zone(pc_raw)

    return CalculatorData(rates=rates, postcode_zone=pc)


def standardize_rates(df: pd.DataFrame) -> pd.DataFrame:
    # expected headers: Zone TYPE Min Weight Max weight Shipping Handling
    cols = {c.strip(): c for c in df.columns.astype(str)}
    zone_col = cols.get("Zone") or cols.get("ZONE") or next((c for c in df.columns if "zone" in str(c).lower()), None)
    type_col = cols.get("TYPE") or next((c for c in df.columns if "type" in str(c).lower()), None)
    min_col = cols.get("Min Weight") or cols.get("Min Weigh") or next((c for c in df.columns if "min" in str(c).lower()), None)
    max_col = cols.get("Max weight") or cols.get("Max Weight") or next((c for c in df.columns if "max" in str(c).lower()), None)
    ship_col = cols.get("Shipping") or next((c for c in df.columns if "ship" in str(c).lower()), None)
    handle_col = cols.get("Handling") or next((c for c in df.columns if "hand" in str(c).lower()), None)

    out = pd.DataFrame()
    out["zone"] = pd.to_numeric(df[zone_col], errors="coerce")
    out["type"] = df[type_col].apply(normalize_type) if type_col is not None else ""
    out["shipping"] = pd.to_numeric(df[ship_col], errors="coerce") if ship_col is not None else None
    out["handling"] = pd.to_numeric(df[handle_col], errors="coerce") if handle_col is not None else None

    out["min_weight"] = pd.to_numeric(df[min_col], errors="coerce") if min_col is not None else 0.0
    out["max_weight"] = pd.to_numeric(df[max_col], errors="coerce") if max_col is not None else float("inf")

    out["min_weight"] = out["min_weight"].fillna(0.0)
    out["max_weight"] = out["max_weight"].fillna(float("inf"))

    out = out.dropna(subset=["zone", "shipping", "handling"])
    out["zone"] = out["zone"].astype(int)

    return out


def standardize_postcode_zone(df: pd.DataFrame) -> pd.DataFrame:
    # expected: col A postcode, col B zone (first two cols)
    if df.shape[1] < 2:
        raise HTTPException(status_code=400, detail="Sheet3 must have at least 2 columns: postcode and zone.")

    pc_col = df.columns[0]
    zone_col = df.columns[1]

    out = pd.DataFrame()
    out["postcode"] = df[pc_col].apply(normalize_postcode)
    out["zone"] = pd.to_numeric(df[zone_col], errors="coerce")

    out = out.dropna(subset=["postcode", "zone"])
    out["zone"] = out["zone"].astype(int)
    out = out[out["postcode"] != ""]

    return out


# =========================================================
# Processing session persistence (for missing AU postcodes)
# =========================================================
def save_processing_session(cleaned_df: pd.DataFrame, missing: list[dict]) -> str:
    session_id = str(uuid.uuid4())
    cleaned_json = cleaned_df.to_json(orient="records")
    missing_json = json.dumps(missing, ensure_ascii=False)

    with db() as con:
        con.execute(
            "INSERT INTO processing_sessions(id, created_at, cleaned_json, missing_json) VALUES(?,?,?,?)",
            (
                session_id,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                cleaned_json,
                missing_json,
            ),
        )
        con.commit()

    return session_id


def load_processing_session(session_id: str) -> tuple[pd.DataFrame, list[dict]]:
    with db() as con:
        row = con.execute("SELECT * FROM processing_sessions WHERE id=?", (session_id,)).fetchone()

    if not row:
        raise HTTPException(status_code=400, detail="Invalid or expired session_id.")

    cleaned = pd.read_json(row["cleaned_json"], orient="records")
    missing = json.loads(row["missing_json"])
    return cleaned, missing


def update_processing_session(session_id: str, cleaned_df: pd.DataFrame, missing: list[dict]) -> None:
    with db() as con:
        con.execute(
            "UPDATE processing_sessions SET cleaned_json=?, missing_json=? WHERE id=?",
            (cleaned_df.to_json(orient="records"), json.dumps(missing, ensure_ascii=False), session_id),
        )
        con.commit()


# =========================================================
# Core calculation
# =========================================================
def calculate_shipping(cleaned: pd.DataFrame, calc: CalculatorData) -> tuple[pd.DataFrame, list[dict]]:
    df = cleaned.copy()

    # Only Case 1 required by you:
    # AUSTRALIA + TYPE in {YES, NO, TEA}
    df["COUNTRY_N"] = df["COUNTRY"].apply(normalize_country)
    df["TYPE_N"] = df["TYPE"].apply(normalize_type)
    df["POSTCODE_N"] = df["POST CODE"].apply(normalize_postcode)
    df["WEIGHT_N"] = pd.to_numeric(df["WEIGHT"], errors="coerce")

    au_mask = (df["COUNTRY_N"] == "AUSTRALIA") & (df["TYPE_N"].isin({"YES", "NO", "TEA"}))

    # postcode->zone
    pc_map = dict(zip(calc.postcode_zone["postcode"], calc.postcode_zone["zone"]))
    df.loc[au_mask, "ZONE"] = df.loc[au_mask, "POSTCODE_N"].map(pc_map)

    # find missing postcodes (zone NaN)
    missing: list[dict] = []
    missing_rows = df[au_mask & (df["ZONE"].isna())].copy()

    if not missing_rows.empty:
        # group by invalid postcode (including blank)
        for pc, group in missing_rows.groupby("POSTCODE_N", dropna=False):
            orders_list = group["ORDERS"].astype(str).dropna().unique().tolist()
            missing.append(
                {
                    "postcode": pc if pc else "",
                    "orders": orders_list,
                }
            )
        return df.drop(columns=["COUNTRY_N", "TYPE_N", "POSTCODE_N", "WEIGHT_N"], errors="ignore"), missing

    # rate lookup
    rates = calc.rates

    def lookup_rate(zone: int, typ: str, w: float) -> tuple[float, float]:
        sub = rates[(rates["zone"] == int(zone)) & (rates["type"] == typ)]
        # weight range inclusive
        hit = sub[(sub["min_weight"] <= w) & (w <= sub["max_weight"])]
        if hit.empty:
            raise ValueError(f"No rate found for zone={zone}, type={typ}, weight={w}")
        row = hit.iloc[0]
        return float(row["shipping"]), float(row["handling"])

    shipping_vals = []
    handling_vals = []

    for _, r in df.iterrows():
        if not ((normalize_country(r["COUNTRY"]) == "AUSTRALIA") and (normalize_type(r["TYPE"]) in {"YES", "NO", "TEA"})):
            shipping_vals.append(None)
            handling_vals.append(None)
            continue

        zone = r.get("ZONE")
        w = r.get("WEIGHT")
        typ = normalize_type(r.get("TYPE"))

        w_num = float(pd.to_numeric(w, errors="coerce")) if pd.notna(w) else None
        if pd.isna(zone) or w_num is None:
            shipping_vals.append(None)
            handling_vals.append(None)
            continue

        try:
            s, h = lookup_rate(int(zone), typ, w_num)
            shipping_vals.append(s)
            handling_vals.append(h)
        except Exception:
            shipping_vals.append(None)
            handling_vals.append(None)

    df["SHIPPING"] = shipping_vals
    df["HANDLING"] = handling_vals

    # TOTAL VALUE (rounded to 2 decimals)
    df["TOTAL VALUE"] = (pd.to_numeric(df["SHIPPING"], errors="coerce").fillna(0.0) +
                         pd.to_numeric(df["HANDLING"], errors="coerce").fillna(0.0)).round(2)

    df = df.drop(columns=["COUNTRY_N", "TYPE_N", "POSTCODE_N", "WEIGHT_N"], errors="ignore")
    return df, []


def clean_orders_df(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure required columns exist
    missing_cols = [c for c in REQUIRED_ORDER_COLUMNS if c not in df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail=f"Orders CSV missing required columns: {missing_cols}",
        )

    # Keep only the required columns in the exact order you requested
    df = df[REQUIRED_ORDER_COLUMNS].copy()

    # Remove rows where SKU (not 平台SKU) starts with NONE (case-insensitive)
    df = df[~df["SKU"].apply(sku_is_none)].copy()

    return df


def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="cleaned_orders")
    return buf.getvalue()


def save_download_file(df: pd.DataFrame) -> str:
    file_id = str(uuid.uuid4())
    out_path = DOWNLOADS_DIR / f"cleaned_orders_{file_id}.xlsx"
    out_bytes = df_to_excel_bytes(df)
    out_path.write_bytes(out_bytes)
    return file_id


def get_download_path(file_id: str) -> Path:
    p = DOWNLOADS_DIR / f"cleaned_orders_{file_id}.xlsx"
    if not p.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return p


# =========================================================
# Routes: Auth
# =========================================================
@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("login.html", {"request": request, "message": None})


@app.post("/login", response_class=HTMLResponse)
async def login_submit(request: Request, username: str = Form(...), password: str = Form(...)) -> HTMLResponse:
    user = get_user_by_username(username.strip())
    if not user:
        return templates.TemplateResponse("login.html", {"request": request, "message": "Invalid username/password."})

    if not verify_password(password, user["pass_salt"], user["pass_hash"]):
        return templates.TemplateResponse("login.html", {"request": request, "message": "Invalid username/password."})

    request.session["user_id"] = int(user["id"])
    return RedirectResponse(url="/", status_code=303)


@app.get("/logout")
def logout(request: Request) -> RedirectResponse:
    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)


# =========================================================
# Routes: Main UI
# =========================================================
@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    user = current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "user": user,
            "message": None,
            "missing": None,
            "session_id": None,
            "total_sum": None,
            "download_id": None,
        },
    )


# =========================================================
# Upload Calculator (allowed for any logged-in user)
# =========================================================
@app.post("/admin/upload-calculator", response_class=HTMLResponse)
async def upload_calculator(request: Request, calculator: UploadFile = File(...)) -> HTMLResponse:
    user = require_login(request)

    if not calculator.filename.lower().endswith((".xlsx", ".xls")):
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "user": user,
                "message": "Please upload a valid .xlsx/.xls file.",
                "missing": None,
                "session_id": None,
                "total_sum": None,
                "download_id": None,
            },
            status_code=400,
        )

    content = await calculator.read()
    try:
        CALCULATOR_PATH.write_bytes(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save calculator: {e}")

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "user": user,
            "message": "Calculator uploaded successfully.",
            "missing": None,
            "session_id": None,
            "total_sum": None,
            "download_id": None,
        },
    )


# =========================================================
# Process orders
# =========================================================
@app.post("/process-orders", response_class=HTMLResponse)
async def process_orders(
    request: Request,
    orders: UploadFile = File(...),
    cutoff_order: str = Form(...),  # mandatory
) -> HTMLResponse:
    user = require_login(request)

    cutoff_order = (cutoff_order or "").strip()
    if not cutoff_order:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "user": user,
                "message": "Cutoff Order is required.",
                "missing": None,
                "session_id": None,
                "total_sum": None,
                "download_id": None,
            },
            status_code=400,
        )

    if not orders.filename.lower().endswith(".csv"):
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "user": user,
                "message": "Please upload a CSV file.",
                "missing": None,
                "session_id": None,
                "total_sum": None,
                "download_id": None,
            },
            status_code=400,
        )

    try:
        content = await orders.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    cleaned = clean_orders_df(df)

    # Apply cutoff (keep rows until ORDERS <= cutoff? depends on your scheme)
    # We'll keep rows with ORDERS <= cutoff if numeric; otherwise stop after first match.
    # If your ORDERS values are like N5504279, we will include rows up to cutoff in file order.
    if "ORDERS" in cleaned.columns:
        cutoff_idx = None
        for i, val in enumerate(cleaned["ORDERS"].astype(str).tolist()):
            if val.strip() == cutoff_order:
                cutoff_idx = i
                break
        if cutoff_idx is None:
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "user": user,
                    "message": f"Cutoff order '{cutoff_order}' not found in ORDERS column.",
                    "missing": None,
                    "session_id": None,
                    "total_sum": None,
                    "download_id": None,
                },
                status_code=400,
            )
        cleaned = cleaned.iloc[cutoff_idx:].copy()

    calc = read_calculator_or_fail()
    processed, missing = calculate_shipping(cleaned, calc)

    if missing:
        session_id = save_processing_session(processed, missing)
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "user": user,
                "message": "Some AU postcodes were not found. Please correct them below and recalculate.",
                "missing": missing,
                "session_id": session_id,
                "total_sum": None,
                "download_id": None,
            },
        )

    # Sum of TOTAL VALUE (rounded to 2 digits)
    total_sum = float(pd.to_numeric(processed.get("TOTAL VALUE"), errors="coerce").fillna(0.0).sum())
    total_sum = round(total_sum, 2)

    file_id = save_download_file(processed)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "user": user,
            "message": "Processing completed successfully.",
            "missing": None,
            "session_id": None,
            "total_sum": total_sum,
            "download_id": file_id,
        },
    )


@app.post("/process-orders/recalculate", response_class=HTMLResponse)
async def recalculate_orders(request: Request, session_id: str = Form(...)) -> HTMLResponse:
    user = require_login(request)

    cleaned, missing = load_processing_session(session_id)

    # Update postcodes from form fields postcode_0, postcode_1, ...
    # missing is list with {"postcode": "...", "orders": [...]}
    new_postcodes = []
    form = await request.form()
    for i in range(len(missing)):
        key = f"postcode_{i}"
        new_postcodes.append(normalize_postcode(form.get(key)))

    # Apply replacements: for each missing group, replace its postcode in cleaned
    for i, m in enumerate(missing):
        old_pc = normalize_postcode(m.get("postcode", ""))
        new_pc = new_postcodes[i]
        if old_pc == "":
            # blank postcode rows
            cleaned.loc[cleaned["POST CODE"].apply(normalize_postcode) == "", "POST CODE"] = new_pc
        else:
            cleaned.loc[cleaned["POST CODE"].apply(normalize_postcode) == old_pc, "POST CODE"] = new_pc

    # Re-run calculation
    calc = read_calculator_or_fail()
    processed, missing2 = calculate_shipping(cleaned, calc)

    if missing2:
        update_processing_session(session_id, processed, missing2)
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "user": user,
                "message": "Still some AU postcodes were not found. Fix them and try again.",
                "missing": missing2,
                "session_id": session_id,
                "total_sum": None,
                "download_id": None,
            },
            status_code=400,
        )

    total_sum = float(pd.to_numeric(processed.get("TOTAL VALUE"), errors="coerce").fillna(0.0).sum())
    total_sum = round(total_sum, 2)
    file_id = save_download_file(processed)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "user": user,
            "message": "Recalculation completed successfully.",
            "missing": None,
            "session_id": None,
            "total_sum": total_sum,
            "download_id": file_id,
        },
    )


# =========================================================
# Download button endpoint
# =========================================================
@app.get("/download/{file_id}")
def download_file(request: Request, file_id: str):
    require_login(request)
    path = get_download_path(file_id)
    return FileResponse(
        path=str(path),
        filename="cleaned_orders.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# =========================================================
# ADMIN: USERS (admin only)
# =========================================================
@app.get("/admin/users", response_class=HTMLResponse)
def admin_users_page(request: Request) -> HTMLResponse:
    user = require_admin(request)
    users = list_users()
    return templates.TemplateResponse(
        "admin_users.html",
        {"request": request, "user": user, "users": users, "message": None},
    )


@app.post("/admin/users/create", response_class=HTMLResponse)
async def admin_users_create(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    is_admin: str | None = Form(None),
):
    user = require_admin(request)
    try:
        create_user(username=username, password=password, is_admin=bool(is_admin))
        msg = "User created successfully."
    except Exception as e:
        msg = f"Failed to create user: {e}"

    users = list_users()
    return templates.TemplateResponse(
        "admin_users.html",
        {"request": request, "user": user, "users": users, "message": msg},
    )


@app.post("/admin/users/reset", response_class=HTMLResponse)
async def admin_users_reset(request: Request, user_id: int = Form(...), new_password: str = Form(...)):
    user = require_admin(request)
    try:
        set_user_password(user_id=user_id, new_password=new_password)
        msg = "Password updated."
    except Exception as e:
        msg = f"Failed to update password: {e}"

    users = list_users()
    return templates.TemplateResponse(
        "admin_users.html",
        {"request": request, "user": user, "users": users, "message": msg},
    )


@app.post("/admin/users/delete", response_class=HTMLResponse)
async def admin_users_delete(request: Request, user_id: int = Form(...)):
    user = require_admin(request)

    if int(user_id) == int(user["id"]):
        users = list_users()
        return templates.TemplateResponse(
            "admin_users.html",
            {"request": request, "user": user, "users": users, "message": "You cannot delete your own account."},
            status_code=400,
        )

    delete_user(user_id=user_id)
    users = list_users()
    return templates.TemplateResponse(
        "admin_users.html",
        {"request": request, "user": user, "users": users, "message": "User deleted."},
    )


# =========================================================
# Startup
# =========================================================
init_db()
