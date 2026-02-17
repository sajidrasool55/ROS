from __future__ import annotations

import base64
import hashlib
import hmac
import io
import json
import os
import secrets
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

# =========================
# PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "app.db"

DOWNLOAD_DIR = BASE_DIR / "downloads"
DOWNLOAD_DIR.mkdir(exist_ok=True)

# =========================
# APP
# =========================
app = FastAPI(title="Orders Processing Web App")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Session secret (REQUIRED for production)
SESSION_SECRET = os.environ.get("SESSION_SECRET", "").strip()
if not SESSION_SECRET:
    # Still runs locally, but you should set it for production (Railway/Render)
    SESSION_SECRET = "DEV_ONLY_CHANGE_ME_" + secrets.token_hex(16)

app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    session_cookie="ops_session",
    https_only=True,  # set True when behind HTTPS (Railway/Render will be HTTPS)
    same_site="lax",
)

# =========================
# CONFIG
# =========================
REQUIRED_COLUMNS = [
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

TYPE_PRIORITY = ["TEA", "FD", "LIQUID", "NO", "YES"]
SESSION_TTL_DAYS = 1
NBSP = "\u00A0"

DEFAULT_ADMIN_USER = os.environ.get("APP_ADMIN_USER", "admin")
DEFAULT_ADMIN_PASS = os.environ.get("APP_ADMIN_PASS", "admin123")  # change after first login!


# =========================
# DATA STRUCTURES
# =========================
@dataclass
class CalculatorData:
    country_rates: pd.DataFrame
    zone_rates: pd.DataFrame
    postcode_zone: pd.DataFrame


# =========================
# DB HELPERS
# =========================
def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_conn()
    cur = conn.cursor()

    # Calculator data
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

    # Sessions for postcode correction workflow
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

    # Users table (multi-user login)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_admin INTEGER NOT NULL DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    conn.commit()
    conn.close()

    ensure_default_admin()


def cleanup_old_sessions() -> None:
    conn = get_conn()
    conn.execute(
        "DELETE FROM processing_sessions WHERE created_at < datetime('now', ?)",
        (f"-{SESSION_TTL_DAYS} day",),
    )
    conn.commit()
    conn.close()


# =========================
# PASSWORD HASHING (PBKDF2)
# =========================
def _pbkdf2_hash(password: str, salt: bytes, iterations: int = 200_000) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations, dklen=32)


def make_password_hash(password: str) -> str:
    salt = secrets.token_bytes(16)
    iterations = 200_000
    dk = _pbkdf2_hash(password, salt, iterations)
    payload = b"%d$%s$%s" % (
        iterations,
        base64.b64encode(salt),
        base64.b64encode(dk),
    )
    return payload.decode("utf-8")


def verify_password(password: str, stored: str) -> bool:
    try:
        parts = stored.split("$")
        iterations = int(parts[0])
        salt = base64.b64decode(parts[1])
        expected = base64.b64decode(parts[2])
        actual = _pbkdf2_hash(password, salt, iterations)
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False


# =========================
# USER CRUD
# =========================
def ensure_default_admin() -> None:
    """Create an admin user if users table is empty."""
    conn = get_conn()
    row = conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()
    count = int(row["c"]) if row else 0

    if count == 0:
        conn.execute(
            "INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, 1)",
            (DEFAULT_ADMIN_USER, make_password_hash(DEFAULT_ADMIN_PASS)),
        )
        conn.commit()

    conn.close()


def get_user_by_username(username: str) -> sqlite3.Row | None:
    conn = get_conn()
    row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    return row


def list_users() -> list[sqlite3.Row]:
    conn = get_conn()
    rows = conn.execute("SELECT id, username, is_admin, created_at FROM users ORDER BY username").fetchall()
    conn.close()
    return list(rows)


def create_user(username: str, password: str, is_admin: bool) -> None:
    username = username.strip()
    if not username:
        raise HTTPException(status_code=400, detail="Username is required.")
    if len(password) < 4:
        raise HTTPException(status_code=400, detail="Password must be at least 4 characters.")

    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, ?)",
            (username, make_password_hash(password), 1 if is_admin else 0),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Username already exists.")
    finally:
        conn.close()


def set_user_password(user_id: int, new_password: str) -> None:
    if len(new_password) < 4:
        raise HTTPException(status_code=400, detail="Password must be at least 4 characters.")
    conn = get_conn()
    conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (make_password_hash(new_password), user_id))
    conn.commit()
    conn.close()


def delete_user(user_id: int) -> None:
    conn = get_conn()
    conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()


# =========================
# AUTH HELPERS
# =========================
def current_user(request: Request) -> dict[str, Any] | None:
    return request.session.get("user")


def require_login(request: Request) -> dict[str, Any]:
    user = current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not logged in.")
    return user


def require_admin(request: Request) -> dict[str, Any]:
    user = require_login(request)
    if not user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin access required.")
    return user


# =========================
# DOWNLOAD FILE HELPERS
# =========================
def save_excel_temp(df: pd.DataFrame) -> str:
    file_id = str(uuid.uuid4())
    file_path = DOWNLOAD_DIR / f"{file_id}.xlsx"
    df.to_excel(file_path, index=False, sheet_name="cleaned_orders")
    return file_id


# =========================
# NORMALIZATION UTILITIES
# =========================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).replace(NBSP, " ").strip() for col in df.columns]
    return df


def is_missing_text(value: Any) -> bool:
    text = str(value).replace(NBSP, "").strip().upper()
    return text in {"", "NAN", "NONE", "NULL"}


def col_match(df: pd.DataFrame, options: list[str]) -> str:
    lower_map = {str(c).replace(NBSP, " ").strip().lower(): c for c in df.columns}
    for opt in options:
        c = lower_map.get(str(opt).strip().lower())
        if c:
            return c
    raise ValueError(f"Missing expected column. Tried: {options}")


def optional_col_match(df: pd.DataFrame, options: list[str]) -> str | None:
    try:
        return col_match(df, options)
    except ValueError:
        return None


def normalize_postcode(value: Any) -> str:
    text = str(value).replace(NBSP, "").strip().upper().replace(" ", "")
    if text in {"", "NAN", "NONE", "NULL"}:
        return ""
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def postcode_numeric_key(postcode: str) -> str:
    if postcode.isdigit():
        key = postcode.lstrip("0")
        return key or "0"
    return postcode


def postcode_lookup_keys(value: Any) -> list[str]:
    pc = normalize_postcode(value)
    if not pc:
        return []

    keys: list[str] = [pc]

    numeric = postcode_numeric_key(pc)
    if numeric not in keys:
        keys.append(numeric)

    if "." in pc:
        try:
            as_float = float(pc)
            if as_float.is_integer():
                int_key = str(int(as_float))
                if int_key not in keys:
                    keys.append(int_key)
                int_numeric = postcode_numeric_key(int_key)
                if int_numeric not in keys:
                    keys.append(int_numeric)
        except ValueError:
            pass

    return keys


def normalize_zone_key(value: Any) -> str:
    text = str(value).replace(NBSP, "").strip().upper()
    if text in {"", "NAN", "NONE", "NULL"}:
        return ""
    try:
        num = float(text)
        if num.is_integer():
            return str(int(num))
    except ValueError:
        pass
    return text


def normalize_type(value: Any) -> str:
    return str(value).replace(NBSP, "").strip().upper()


# =========================
# CALCULATOR (EXCEL) LOADING
# =========================
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
    if key == "zone":
        key_col = col_match(df, ["zone", "Zone", "ZONE"])
    else:
        key_col = col_match(df, ["country", "Country", "COUNTRY"])

    type_col = col_match(df, ["type", "Type", "TYPE"])
    ship_col = col_match(df, ["shipping", "Shipping", "shipping rate", "price", "freight"])
    handle_col = col_match(df, ["handling", "Handling", "handling fee", "service fee"])

    min_col = optional_col_match(
        df,
        ["min_weight", "min weight", "Min Weight", "weight_from", "from", "min"],
    )
    max_col = optional_col_match(
        df,
        ["max_weight", "max weight", "Max Weight", "max weigh", "Max weigh", "weight_to", "to", "max"],
    )

    out = pd.DataFrame()

    if key == "zone":
        out["key"] = df[key_col].apply(normalize_zone_key)
    else:
        out["key"] = df[key_col].astype(str).str.replace(NBSP, "", regex=False).str.strip().str.upper()

    out["type"] = df[type_col].apply(normalize_type)
    out["shipping"] = pd.to_numeric(df[ship_col], errors="coerce")
    out["handling"] = pd.to_numeric(df[handle_col], errors="coerce")

    def clean_number(col):
        return (
            col.astype(str)
            .str.replace(",", "", regex=False)
            .str.replace(" ", "", regex=False)
            .str.replace(NBSP, "", regex=False)
            .str.strip()
            .replace("", "0")
        )

    out["min_weight"] = pd.to_numeric(clean_number(df[min_col]), errors="coerce") if min_col else 0.0
    out["max_weight"] = pd.to_numeric(clean_number(df[max_col]), errors="coerce") if max_col else float("inf")

    out["min_weight"] = out["min_weight"].fillna(0.0)
    out["max_weight"] = out["max_weight"].fillna(float("inf"))

    out = out.dropna(subset=["shipping", "handling"])
    return out


def standardize_postcode_zone(df: pd.DataFrame) -> pd.DataFrame:
    pc_col = col_match(df, ["postcode", "post code", "postal code", "Postal Code", "zip"])
    zone_col = col_match(df, ["zone", "Zone", "shipping zone"])

    out = pd.DataFrame()
    out["postcode"] = df[pc_col].apply(normalize_postcode)
    out["zone"] = df[zone_col].apply(normalize_zone_key)
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

    country_rates = pd.read_json(io.StringIO(row["country_rates_json"]), orient="split")
    zone_rates = pd.read_json(io.StringIO(row["zone_rates_json"]), orient="split")
    postcode_zone = pd.read_json(io.StringIO(row["postcode_zone_json"]), orient="split")

    # Force stable string keys after JSON load (avoid "2" -> 2 int issues)
    if "key" in country_rates.columns:
        country_rates["key"] = country_rates["key"].astype(str).str.replace(NBSP, "", regex=False).str.strip().str.upper()

    if "key" in zone_rates.columns:
        zone_rates["key"] = zone_rates["key"].astype(str).apply(normalize_zone_key)

    if "type" in zone_rates.columns:
        zone_rates["type"] = zone_rates["type"].astype(str).apply(normalize_type)

    if "postcode" in postcode_zone.columns:
        postcode_zone["postcode"] = postcode_zone["postcode"].astype(str).apply(normalize_postcode)

    if "zone" in postcode_zone.columns:
        postcode_zone["zone"] = postcode_zone["zone"].astype(str).apply(normalize_zone_key)

    return CalculatorData(country_rates=country_rates, zone_rates=zone_rates, postcode_zone=postcode_zone)


# =========================
# ORDERS PREPROCESSING
# =========================
def preprocess_orders(df: pd.DataFrame, cutoff_order: str) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"orders.csv missing required columns: {missing}")

    cleaned = df.copy()
    cleaned = cleaned.sort_values("ORDERS", kind="stable")

    cutoff_order = str(cutoff_order).replace(NBSP, "").strip()
    if not cutoff_order:
        raise HTTPException(status_code=400, detail="Cutoff Order is required.")

    order_series = cleaned["ORDERS"].astype(str).str.replace(NBSP, "", regex=False).str.strip()
    matches = cleaned.index[order_series == cutoff_order].tolist()
    if matches:
        cleaned = cleaned.loc[matches[0] :]

    # Remove rows where 平台SKU contains "-US"
    platform_sku = cleaned["平台SKU"].astype(str).str.replace(NBSP, "", regex=False).str.strip()
    cleaned = cleaned[~platform_sku.str.upper().str.contains("-US", na=False)]

    # Remove rows where 平台SKU == "WLN-INSERT1"
    platform_sku = cleaned["平台SKU"].astype(str).str.replace(NBSP, "", regex=False).str.strip()
    cleaned = cleaned[~platform_sku.str.upper().eq("WLN-INSERT1")]

    # Remove rows where SKU (NOT 平台SKU) starts with "NONE"
    sku = cleaned["SKU"].astype(str).str.replace(NBSP, "", regex=False).str.strip().str.upper()
    cleaned = cleaned[~sku.str.startswith("NONE", na=False)]

    # If both 平台SKU and WEIGHT are missing, skip those rows
    weight_missing = cleaned["WEIGHT"].apply(is_missing_text)
    platform_sku_missing = cleaned["平台SKU"].apply(is_missing_text)
    cleaned = cleaned[~(platform_sku_missing & weight_missing)]

    # Normalize TYPE robustly
    cleaned["TYPE"] = cleaned["TYPE"].apply(normalize_type)
    cleaned.loc[cleaned["TYPE"].isin(["NYLON", "POLYESTER"]), "TYPE"] = "YES"

    cleaned["QTY"] = pd.to_numeric(cleaned["QTY"], errors="coerce").fillna(0)
    cleaned["WEIGHT"] = pd.to_numeric(cleaned["WEIGHT"], errors="coerce").fillna(0)

    cleaned["COUNTRY"] = cleaned["COUNTRY"].astype(str).str.replace(NBSP, "", regex=False).str.strip().str.upper()
    cleaned["POST CODE"] = cleaned["POST CODE"].apply(normalize_postcode)

    cleaned["RowWeight"] = cleaned["QTY"] * cleaned["WEIGHT"]
    return cleaned


# =========================
# SHIPPING CALCULATION
# =========================
def pick_order_type(types: pd.Series) -> str:
    values = set(types.astype(str).str.replace(NBSP, "", regex=False).str.strip().str.upper())
    for t in TYPE_PRIORITY:
        if t in values:
            return t
    return "YES"


def find_rate(rates: pd.DataFrame, key: str, typ: str, weight: float) -> tuple[float, float]:
    key_str = str(key).strip()
    subset = rates[rates["key"].astype(str).str.strip() == key_str]
    if subset.empty:
        return float("nan"), float("nan")

    typ = normalize_type(typ)
    typed = subset[subset["type"] == typ]
    if typed.empty:
        typed = subset

    in_range = typed[(typed["min_weight"] <= weight) & (typed["max_weight"] >= weight)]
    if in_range.empty:
        in_range = typed.sort_values("max_weight").tail(1)

    if in_range.empty:
        return float("nan"), float("nan")

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

    # Build postcode->zone map from Sheet3
    postcode_to_zone: dict[str, str] = {}
    for _, row in calc.postcode_zone.iterrows():
        zone = normalize_zone_key(row["zone"])
        for k in postcode_lookup_keys(row["postcode"]):
            postcode_to_zone.setdefault(k, zone)

    missing_map: dict[str, dict[str, Any]] = {}
    first_idx_by_order = df.groupby("ORDERS").head(1).index

    for idx in first_idx_by_order:
        order = df.at[idx, "ORDERS"]
        country = str(df.at[idx, "COUNTRY"]).replace(NBSP, "").strip().upper()
        order_type = normalize_type(order_type_map[order])
        order_weight = float(order_weight_map[order])

        shipping = float("nan")
        handling = float("nan")
        zone = ""
        rule_hint = "Sheet1 (country rates)"

        # AUSTRALIA + TYPE in {YES, NO, TEA}
        if country == "AUSTRALIA" and order_type in {"YES", "NO", "TEA"}:
            postcode = normalize_postcode(df.at[idx, "POST CODE"])
            for k in postcode_lookup_keys(postcode):
                zone = postcode_to_zone.get(k, "")
                if zone:
                    break

            if not zone:
                map_key = postcode or "<BLANK>"
                missing_map.setdefault(map_key, {"postcode": postcode, "orders": [], "indexes": []})
                missing_map[map_key]["orders"].append(str(order))
                missing_map[map_key]["indexes"].append(int(idx))
                continue

            rule_hint = "Sheet3 (postcode->zone) + Sheet2 (zone rates)"
            shipping, handling = find_rate(calc.zone_rates, zone, order_type, order_weight)

        else:
            key_country = country
            if key_country not in set(calc.country_rates["key"]):
                key_country = "UNITED STATES"
            rule_hint = f"Sheet1 (country rates, country used: {key_country})"
            shipping, handling = find_rate(calc.country_rates, key_country, order_type, order_weight)

        if pd.isna(shipping) or pd.isna(handling):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"No shipping rule found for order {order}. "
                    f"country={country}, zone={zone or '-'}, type={order_type}, weight={order_weight}. "
                    f"Checked: {rule_hint}."
                ),
            )

        total_value = (((order_weight * shipping) / 1000) + handling) / 7 + 0.9

        df.at[idx, "Order Total Weight"] = round(order_weight, 4)
        df.at[idx, "Shipping"] = round(shipping, 4)
        df.at[idx, "Handling"] = round(handling, 4)
        df.at[idx, "Zone"] = zone
        df.at[idx, "Total Value"] = round(total_value, 2)

    missing_postcodes = sorted(missing_map.values(), key=lambda item: (item["postcode"] == "", item["postcode"]))
    df = df.drop(columns=["RowWeight"])
    return df, missing_postcodes


# =========================
# SESSIONS FOR POSTCODE FIX
# =========================
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


# =========================
# STARTUP
# =========================
@app.on_event("startup")
def startup() -> None:
    init_db()
    cleanup_old_sessions()


# =========================
# AUTH ROUTES
# =========================
@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request) -> HTMLResponse:
    if current_user(request):
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse("login.html", {"request": request, "message": None})


@app.post("/login")
async def login_action(request: Request, username: str = Form(...), password: str = Form(...)):
    row = get_user_by_username(username.strip())
    if not row or not verify_password(password, row["password_hash"]):
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "message": "Invalid username or password."},
            status_code=401,
        )

    request.session["user"] = {"id": int(row["id"]), "username": row["username"], "is_admin": bool(row["is_admin"])}
    return RedirectResponse(url="/", status_code=303)


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)


# =========================
# MAIN UI
# =========================
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


@app.get("/download/{file_id}")
def download_file(request: Request, file_id: str):
    require_login(request)
    file_path = DOWNLOAD_DIR / f"{file_id}.xlsx"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File expired or not found")
    return FileResponse(file_path, filename="cleaned_orders.xlsx")


# =========================
# ADMIN: USERS
# =========================
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
    create_user(username=username, password=password, is_admin=bool(is_admin))
    users = list_users()
    return templates.TemplateResponse(
        "admin_users.html",
        {"request": request, "user": user, "users": users, "message": "User created successfully."},
    )


@app.post("/admin/users/reset", response_class=HTMLResponse)
async def admin_users_reset(request: Request, user_id: int = Form(...), new_password: str = Form(...)):
    user = require_admin(request)
    set_user_password(user_id=user_id, new_password=new_password)
    users = list_users()
    return templates.TemplateResponse(
        "admin_users.html",
        {"request": request, "user": user, "users": users, "message": "Password updated."},
    )


@app.post("/admin/users/delete", response_class=HTMLResponse)
async def admin_users_delete(request: Request, user_id: int = Form(...)):
    user = require_admin(request)
    # Prevent deleting yourself
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


# =========================
# ADMIN: UPLOAD CALCULATOR
# =========================
@app.post("/admin/upload-calculator", response_class=HTMLResponse)
async def upload_calculator(request: Request, calculator: UploadFile = File(...)) -> HTMLResponse:
    user = require_admin(request)

    if not calculator.filename.lower().endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Please upload an Excel file")

    try:
        calc = load_calculator_from_excel(await calculator.read())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    save_calculator_data(calc)

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


# =========================
# USER: PROCESS ORDERS
# =========================
@app.post("/process-orders", response_class=HTMLResponse)
async def process_orders(
    request: Request,
    orders: UploadFile = File(...),
    cutoff_order: str = Form(...),  # mandatory
):
    user = require_login(request)

    if not orders.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file")

    calc = load_calculator_data()

    try:
        raw_df = pd.read_csv(io.BytesIO(await orders.read()))
        raw_df = normalize_columns(raw_df)
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
                "user": user,
                "message": "Some AU postcodes were not found. Please correct them below and recalculate.",
                "missing": missing,
                "session_id": session_id,
                "total_sum": None,
                "download_id": None,
            },
        )

    total_sum = round(pd.to_numeric(processed["Total Value"], errors="coerce").fillna(0).sum(), 2)
    file_id = save_excel_temp(processed)

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
async def recalculate_orders(request: Request, session_id: str = Form(...)):
    user = require_login(request)

    calc = load_calculator_data()
    df, missing = load_processing_session(session_id)

    form = await request.form()
    for i, entry in enumerate(missing):
        field_name = f"postcode_{i}"
        corrected = normalize_postcode(form.get(field_name, ""))
        if corrected:
            for idx in entry.get("indexes", []):
                df.at[idx, "POST CODE"] = corrected

    processed, remaining = calculate_shipping(df, calc)
    if remaining:
        new_session = save_processing_session(df, remaining)
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "user": user,
                "message": "Some postcodes are still invalid. Please update and try again.",
                "missing": remaining,
                "session_id": new_session,
                "total_sum": None,
                "download_id": None,
            },
        )

    total_sum = round(pd.to_numeric(processed["Total Value"], errors="coerce").fillna(0).sum(), 2)
    file_id = save_excel_temp(processed)

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
# =========================
# ADMIN: USERS
# =========================
@app.get("/admin/users", response_class=HTMLResponse)
def admin_users_page(request: Request) -> HTMLResponse:
    user = require_admin(request)
    users = list_users()
    return templates.TemplateResponse(
        "admin_users.html",
        {"request": request, "user": user, "users": users, "message": None},
    )
