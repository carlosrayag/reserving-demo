import os
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

DDL = """
CREATE TABLE IF NOT EXISTS claims_triangle (
    lob_code TEXT NOT NULL,
    product_code TEXT NOT NULL,
    segment_code TEXT NOT NULL,
    region_code TEXT NOT NULL,
    currency TEXT NOT NULL,
    accident_year INTEGER NOT NULL,
    dev_month INTEGER NOT NULL,
    paid REAL NOT NULL,
    incurred REAL NOT NULL,
    PRIMARY KEY (lob_code, product_code, segment_code, region_code, currency, accident_year, dev_month)
);
"""

# =============================
# Caching
# =============================
@st.cache_resource(show_spinner=False)
def get_engine(db_path: str):
    # Streamlit Cloud permite escribir en el filesystem del contenedor (efímero)
    # SQLite en archivo funciona bien para demo.
    return create_engine(f"sqlite:///{db_path}", future=True)

def _generate_synthetic_triangle(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    devs = [12, 24, 36, 48, 60]
    ays = list(range(2018, 2026))  # 2018–2025

    incurred_pattern = np.array([0.55, 0.78, 0.90, 0.97, 1.00])
    paid_pattern = np.array([0.45, 0.70, 0.85, 0.95, 1.00])

    slices = [
        ("AUTO", "AUTO_STD", "RETAIL", "CENTRO", "MXN"),
        ("AUTO", "AUTO_STD", "RETAIL", "NORTE",  "MXN"),
        ("AUTO", "AUTO_STD", "CORP",   "CENTRO", "MXN"),
        ("GMM",  "GMM_STD",  "RETAIL", "CENTRO", "MXN"),
    ]

    rows = []
    base_ultimate = 120_000_000
    trend = 0.06

    for (lob, prod, seg, reg, ccy) in slices:
        for i, ay in enumerate(ays):
            ult_mean = base_ultimate * ((1 + trend) ** (i - (len(ays)-1)))
            ultimate = ult_mean * rng.lognormal(mean=0.0, sigma=0.10)

            incurred_cum = ultimate * incurred_pattern * rng.lognormal(0.0, 0.03, size=len(devs))
            incurred_cum = np.maximum.accumulate(incurred_cum)

            paid_cum = ultimate * paid_pattern * rng.lognormal(0.0, 0.05, size=len(devs))
            paid_cum = np.maximum.accumulate(paid_cum)

            # AY recientes con menos desarrollo
            max_dev_idx = len(devs) - 1 - max(0, (ay - 2021))
            max_dev_idx = int(np.clip(max_dev_idx, 0, len(devs)-1))

            for j, d in enumerate(devs):
                if j <= max_dev_idx:
                    rows.append({
                        "lob_code": lob,
                        "product_code": prod,
                        "segment_code": seg,
                        "region_code": reg,
                        "currency": ccy,
                        "accident_year": int(ay),
                        "dev_month": int(d),
                        "paid": float(paid_cum[j]),
                        "incurred": float(incurred_cum[j]),
                    })

    return pd.DataFrame(rows)

def ensure_db(db_path: str, seed: int = 42):
    """
    A prueba de Streamlit Cloud:
    - crea tabla si no existe
    - si no hay filas, inserta datos simulados
    """
    engine = get_engine(db_path)

    with engine.begin() as conn:
        conn.execute(text(DDL))

        n = conn.execute(text("SELECT COUNT(*) FROM claims_triangle")).scalar()
        if n is None or int(n) == 0:
            df = _generate_synthetic_triangle(seed=seed)
            # Insertar con pandas (rápido y estable)
            df.to_sql("claims_triangle", engine, if_exists="append", index=False)

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
        "lob": lob, "product": product, "segment": segment, "region": region, "currency": currency
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
        last_val = float(row[last_dev])
        idx = devs.index(last_dev)

        if idx < len(fvals):
            tail_prod = np.prod([x for x in fvals[idx:] if pd.notna(x)])
        else:
            tail_prod = 1.0

        ultimate = last_val * float(tail_prod)
        rows.append({
            "Accident Year": int(ay),
            "Latest": last_val,
            "Ultimate": ultimate,
            "IBNR": ultimate - last_val
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("Accident Year")
    return out

# =============================
# Sidebar
# =============================
st.sidebar.title("Parámetros")

db_path = st.sidebar.text_input("Base de datos", DEFAULT_DB).strip() or DEFAULT_DB

# Asegurar BD y datos
ensure_db(db_path, seed=42)

measure = st.sidebar.selectbox("Medida", ["Incurred", "Paid"])

# Catálogos
lobs = fetch_distinct(db_path, "lob_code")
products = fetch_distinct(db_path, "product_code")
segments = fetch_distinct(db_path, "segment_code")
regions = fetch_distinct(db_path, "region_code")
currencies = fetch_distinct(db_path, "currency")

if not lobs:
    st.error("La base no tiene datos (inesperado). Recarga la página.")
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
    st.info("No hay datos para la combinación seleccionada. Cambia los filtros.")
else:
    st.dataframe(triangle.style.format("{:,.0f}"), use_container_width=True)

if run:
    if triangle.empty:
        st.warning("No se puede ejecutar simulación: el triángulo está vacío.")
        st.stop()

    factors = chain_ladder_factors(triangle)
    results = project_ultimate(triangle, factors)

    if results.empty:
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
    fdf = factors.reset_index()
    fdf.columns = ["Desarrollo", "Factor"]
    st.dataframe(fdf.style.format({"Factor": "{:.4f}"}), use_container_width=True)
