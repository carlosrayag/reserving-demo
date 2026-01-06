import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine, text

# =============================
# Configuración general
# =============================
st.set_page_config(
    page_title="Simulador de Reservas – Chain Ladder (Laboratorio)",
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
# Engine / BD
# =============================
@st.cache_resource(show_spinner=False)
def get_engine(db_path: str):
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
    engine = get_engine(db_path)
    with engine.begin() as conn:
        conn.execute(text(DDL))
        n = conn.execute(text("SELECT COUNT(*) FROM claims_triangle")).scalar()
        if n is None or int(n) == 0:
            df = _generate_synthetic_triangle(seed=seed)
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
# Chain Ladder (métodos de factores)
# =============================
def _ata_factor(triangle: pd.DataFrame, c0: int, c1: int, method: str) -> float:
    """Calcula factor c0->c1 según método."""
    mask = triangle[c0].notna() & triangle[c1].notna()
    x = triangle.loc[mask, c0].astype(float)
    y = triangle.loc[mask, c1].astype(float)

    if len(x) == 0:
        return np.nan

    ratios = (y / x).replace([np.inf, -np.inf], np.nan).dropna()
    if len(ratios) == 0:
        return np.nan

    if method == "Ponderado (volumen)":
        denom = x.sum()
        num = y.sum()
        return (num / denom) if denom > 0 else np.nan
    elif method == "Promedio simple":
        return float(ratios.mean())
    elif method == "Mediana":
        return float(ratios.median())
    else:
        return float(ratios.mean())

def chain_ladder_factors(triangle: pd.DataFrame, method: str) -> pd.Series:
    devs = list(triangle.columns)
    f = []
    for i in range(len(devs) - 1):
        c0, c1 = devs[i], devs[i + 1]
        f.append(_ata_factor(triangle, c0, c1, method))
    return pd.Series(f, index=[f"{devs[i]}->{devs[i+1]}" for i in range(len(devs)-1)])

def project_ultimate(triangle: pd.DataFrame, factors: pd.Series, tail_factor: float = 1.0) -> pd.DataFrame:
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
            base_ldf = np.prod([x for x in fvals[idx:] if pd.notna(x)])
        else:
            base_ldf = 1.0

        ldf_total = float(base_ldf) * float(tail_factor)
        ultimate = last_val * ldf_total

        rows.append({
            "Accident Year": int(ay),
            "Latest": last_val,
            "LDF_total": ldf_total,
            "Ultimate": ultimate,
            "IBNR": ultimate - last_val
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("Accident Year")
    return out

# =============================
# Escenario de inflación (stress)
# =============================
def apply_inflation_stress(results: pd.DataFrame, annual_inflation: float, horizon_years: float = 1.0) -> pd.DataFrame:
    """
    Ajuste simple y defendible para demo:
    - Inflación afecta principalmente la parte no observada: IBNR.
    - Escalamos IBNR por (1+infl)^horizon.
    """
    out = results.copy()
    factor = (1.0 + annual_inflation) ** horizon_years
    out["IBNR_stress"] = out["IBNR"] * factor
    out["Ultimate_stress"] = out["Latest"] + out["IBNR_stress"]
    out["Stress_factor"] = factor
    return out

# =============================
# Sidebar
# =============================
st.sidebar.title("Parámetros")

db_path = st.sidebar.text_input("Base de datos", DEFAULT_DB).strip() or DEFAULT_DB
ensure_db(db_path, seed=42)

measure = st.sidebar.selectbox("Medida", ["Incurred", "Paid"])

factor_method = st.sidebar.selectbox(
    "Método de factores",
    ["Ponderado (volumen)", "Promedio simple", "Mediana"]
)

tail_factor = st.sidebar.slider("Tail factor", min_value=1.00, max_value=1.10, value=1.00, step=0.005)

infl = st.sidebar.slider("Inflación anual (stress)", min_value=0.0, max_value=0.20, value=0.0, step=0.01)
horizon = st.sidebar.slider("Horizonte (años) para stress", min_value=0.5, max_value=3.0, value=1.0, step=0.5)

# Catálogos
lobs = fetch_distinct(db_path, "lob_code")
products = fetch_distinct(db_path, "product_code")
segments = fetch_distinct(db_path, "segment_code")
regions = fetch_distinct(db_path, "region_code")
currencies = fetch_distinct(db_path, "currency")

lob = st.sidebar.selectbox("Línea", lobs)
product = st.sidebar.selectbox("Producto", products)
segment = st.sidebar.selectbox("Segmento", segments)
region = st.sidebar.selectbox("Región", regions)
currency = st.sidebar.selectbox("Moneda", currencies)

if st.sidebar.button("Reset (demo)"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

run = st.sidebar.button("Ejecutar simulación")

# =============================
# Main
# =============================
st.title("📊 Laboratorio de Reservas – Chain Ladder")
st.caption("Datos simulados con fines ilustrativos. El objetivo es mostrar sensibilidad a supuestos y escenarios.")

triangle = load_triangle(db_path, measure, lob, product, segment, region, currency)

# Triángulo (performance: preview)
st.subheader("Triángulo acumulado (preview)")
if triangle.empty:
    st.info("No hay datos para la combinación seleccionada. Cambia los filtros.")
    st.stop()
else:
    st.dataframe(triangle.round(0).head(10), use_container_width=True)

    heat = px.imshow(triangle, aspect="auto", title="Heatmap del triángulo (valores acumulados)")
    st.plotly_chart(heat, use_container_width=True)

if run:
    factors = chain_ladder_factors(triangle, factor_method)
    results = project_ultimate(triangle, factors, tail_factor=tail_factor)

    if results.empty:
        st.error("No se pudieron generar resultados. Prueba otra combinación.")
        st.stop()

    stress = apply_inflation_stress(results, annual_inflation=infl, horizon_years=horizon)

    # KPIs base
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest total", f"{results['Latest'].sum():,.0f}")
    c2.metric("IBNR total", f"{results['IBNR'].sum():,.0f}")
    c3.metric("Ultimate total", f"{results['Ultimate'].sum():,.0f}")
    c4.metric("Ultimate (stress)", f"{stress['Ultimate_stress'].sum():,.0f}")

    # Waterfall base
    water = pd.DataFrame({
        "Concepto": ["Latest", "IBNR", "Ultimate"],
        "Valor": [results["Latest"].sum(), results["IBNR"].sum(), results["Ultimate"].sum()]
    })
    wf = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "total"],
        x=water["Concepto"],
        y=water["Valor"],
    ))
    wf.update_layout(title="Puente (base): Latest → IBNR → Ultimate", showlegend=False)
    st.plotly_chart(wf, use_container_width=True)

    # Waterfall stress (Latest + IBNR_stress)
    water_s = pd.DataFrame({
        "Concepto": ["Latest", "IBNR (stress)", "Ultimate (stress)"],
        "Valor": [stress["Latest"].sum(), stress["IBNR_stress"].sum(), stress["Ultimate_stress"].sum()]
    })
    wf2 = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "total"],
        x=water_s["Concepto"],
        y=water_s["Valor"],
    ))
    wf2.update_layout(title="Puente (stress): Latest → IBNR_stress → Ultimate_stress", showlegend=False)
    st.plotly_chart(wf2, use_container_width=True)

    st.subheader("Resultados por Accident Year (base vs stress)")
    show = stress[["Accident Year", "Latest", "LDF_total", "Ultimate", "IBNR", "Ultimate_stress", "IBNR_stress"]].copy()
    st.dataframe(show.round(0), use_container_width=True)

    # IBNR por AY (base)
    fig = px.bar(results, x="Accident Year", y="IBNR", title="IBNR por Accident Year (base)")
    st.plotly_chart(fig, use_container_width=True)

    # IBNR por AY (stress)
    fig2 = px.bar(stress, x="Accident Year", y="IBNR_stress", title="IBNR por Accident Year (stress)")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Factores Chain Ladder")
    fdf = factors.reset_index()
    fdf.columns = ["Desarrollo", "Factor"]
    st.dataframe(fdf.style.format({"Factor": "{:.4f}"}), use_container_width=True)
