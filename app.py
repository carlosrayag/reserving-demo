import os
import sys
import subprocess
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine, text

# =============================
# Configuración general
# =============================
st.set_page_config(
    page_title="Simulador de Reservas – Chain Ladder",
    page_icon="📊",
    layout="wide"
)

DEFAULT_DB = "reserving_demo.db"

# =============================
# Garantizar existencia y consistencia de BD
# =============================
def ensure_db(db_path: str):
    # 1) Si no existe el archivo, créalo
    if not os.path.exists(db_path):
        subprocess.check_call([
            sys.executable, "create_db.py",
            "--db", db_path, "--seed", "42", "--rebuild"
        ])
        return

    # 2) Si existe, verifica que tenga tabla y datos; si falla, reconstruye
    try:
        engine = create_engine(f"sqlite:///{db_path}", future=True)
        with engine.begin() as conn:
            exists = conn.execute(text("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='claims_triangle'
            """)).fetchone()

            if not exists:
                raise RuntimeError("Tabla claims_triangle no existe")

            n = conn.execute(text("SELECT COUNT(*) FROM claims_triangle")).scalar()
            if n == 0:
                raise RuntimeError("Tabla claims_triangle está vacía")

    except Exception:
        subprocess.check_call([
            sys.executable, "create_db.py",
            "--db", db_path, "--seed", "42", "--rebuild"
        ])

# =============================
# Caching
# =============================
@st.cache_resource(show_spinner=False)
def get_engine(db_path: str):
    return create_engine(f"sqlite:///{db_path}", future=True)

@st.cache_data(show_spinner=False)
def fetch_distinct(db_path: str, column: str):
    engine = get_engine(db_path)
    with engine.begin() as conn:
        q = text(f"SELECT DISTINCT {column} FROM claims_triangle ORDER BY {column}")
        return [r[0] for r in conn.execute(q).all()]

@st.cache_data(show_spinner=False)
def load_triangle(db_path: str, measure: str, lob: str, product: str, segment: str, region: str, currency: str):
    engine = get_engine(db_path)
    col = "incurred" if measure == "Incurred" else "paid"
    q = text(f"""
        SELECT accident_year, dev_month, {col} AS value
        FROM claims_triangle
        WHERE lob_code=:lob
          AND product_code=:product
          AND segment_code=:segment
          AND region_code=:region
          AND currency=:currency
        ORDER BY accident_year, dev_month
    """)
    df = pd.read_sql(q, engine, params={
        "lob": lob,
        "product": product,
        "segment": segment,
        "region": region,
        "currency": currency
    })
    tri = df.pivot(index="accident_year", columns="dev_month", values="value")
    tri = tri.sort_index().sort_index(axis=1)
    return tri

# =============================
# Modelo Chain Ladder
# =============================
def chain_ladder_factors(triangle: pd.DataFrame) -> pd.Series:
    devs = list(triangle.columns)
    f = []
    for i in range(len(devs) - 1):
        c0, c1 = devs[i], devs[i + 1]
        mask = triangle[c0].notna() & triangle[c1].notna()
        denom = triangle.loc[mask, c0].sum()
        num = triangle.loc[mask, c1].sum()
        factor = (num / denom) if denom and denom > 0 else np.nan
        f.append(factor)
    return pd.Series(f, index=[f"{devs[i]}->{devs[i+1]}" for i in range(len(devs)-1)])

def project_ultimate(triangle: pd.DataFrame, factors: pd.Series) -> pd.DataFrame:
    devs = list(triangle.columns)
    fvals = factors.values

    rows = []
    for ay in triangle.index:
        row = triangle.loc[ay]
        last_dev = row.last_valid_index()
        if last_dev is None:
            continue
        last_val = row[last_dev]
        idx = devs.index(last_dev)

        # Producto de factores desde el último dev hacia el final
        if idx < len(fvals):
            tail_prod = np.prod([x for x in fvals[idx:] if pd.notna(x)])
        else:
            tail_prod = 1.0

        ultimate = float(last_val) * float(tail_prod)
        rows.append({
            "Accident Year": int(ay),
            "Latest": float(last_val),
            "Ultimate": float(ultimate),
            "IBNR": float(ultimate - last_val)
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("Accident Year")
    return out

# =============================
# Sidebar (Parámetros)
# =============================
st.sidebar.title("Parámetros")

db_path = st.sidebar.text_input("Base de datos", DEFAULT_DB).strip() or DEFAULT_DB

# Garantiza BD válida antes de consultar catálogos
ensure_db(db_path)

measure = st.sidebar.selectbox("Medida", ["Incurred", "Paid"])

# Cargar catálogos con guardas
try:
    lobs = fetch_distinct(db_path, "lob_code")
    products = fetch_distinct(db_path, "product_code")
    segments = fetch_distinct(db_path, "segment_code")
    regions = fetch_distinct(db_path, "region_code")
    currencies = fetch_distinct(db_path, "currency")
except Exception:
    st.error("No se pudo cargar la BD. Revisa que 'create_db.py' exista en el repo y que la BD se pueda generar.")
    st.stop()

if not lobs:
    st.error("La base no contiene datos. Intenta recargar la página; el sistema intentará regenerar la BD.")
    st.stop()

lob = st.sidebar.selectbox("Línea", lobs)
product = st.sidebar.selectbox("Producto", products)
segment = st.sidebar.selectbox("Segmento", segments)
region = st.sidebar.selectbox("Región", regions)
currency = st.sidebar.selectbox("Moneda", currencies)

run = st.sidebar.button("Ejecutar simulación")

# =============================
# Main
# =============================
st.title("📊 Simulador de Reservas – Chain Ladder")
st.caption("Los datos mostrados son simulados y utilizados únicamente con fines ilustrativos.")

triangle = load_triangle(db_path, measure, lob, product, segment, region, currency)

st.subheader("Triángulo acumulado")
if triangle.empty:
    st.info("No hay datos para la combinación seleccionada. Cambia los filtros o regenera la base.")
else:
    st.dataframe(triangle.style.format("{:,.0f}"), use_container_width=True)

if run:
    if triangle.empty:
        st.warning("No se puede ejecutar simulación: el triángulo está vacío para el slice seleccionado.")
        st.stop()

    factors = chain_ladder_factors(triangle)

    # Si hay factores NaN por falta de datos, avisar (sin crashear)
    if factors.isna().any():
        st.warning("Algunos factores no pudieron calcularse (falta de datos en ciertas diagonales). Se usará lo disponible.")

    results = project_ultimate(triangle, factors)

    if results.empty or not all(col in results.columns for col in ["Latest", "Ultimate", "IBNR"]):
        st.error("No se pudieron generar resultados para este triángulo. Prueba con otro slice.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Latest total", f"{results['Latest'].sum():,.0f}")
    c2.metric("Ultimate total", f"{results['Ultimate'].sum():,.0f}")
    c3.metric("IBNR total", f"{results['IBNR'].sum():,.0f}")

    st.subheader("Resultados por año de ocurrencia")
    st.dataframe(results.style.format("{:,.0f}"), use_container_width=True)

    fig = px.bar(results, x="Accident Year", y="IBNR", title="IBNR por Accident Year")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Factores Chain Ladder")
    factors_df = factors.reset_index()
    factors_df.columns = ["Desarrollo", "Factor"]
    st.dataframe(factors_df.style.format({"Factor": "{:.4f}"}), use_container_width=True)
