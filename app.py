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
    page_title="Laboratorio de Reservas – Chain Ladder (Escenarios + Bootstrap)",
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
# Stress por inflación
# =============================
def apply_inflation_stress(results: pd.DataFrame, annual_inflation: float, horizon_years: float = 1.0) -> pd.DataFrame:
    out = results.copy()
    factor = (1.0 + annual_inflation) ** horizon_years
    out["IBNR_stress"] = out["IBNR"] * factor
    out["Ultimate_stress"] = out["Latest"] + out["IBNR_stress"]
    out["Stress_factor"] = factor
    return out

# =============================
# Bootstrap (Incremento 3)
# =============================
def _ratios_by_dev(triangle: pd.DataFrame):
    """
    Devuelve diccionario: (c0,c1) -> array de ratios y/x, para resampling.
    """
    devs = list(triangle.columns)
    ratios = {}
    for i in range(len(devs) - 1):
        c0, c1 = devs[i], devs[i + 1]
        mask = triangle[c0].notna() & triangle[c1].notna()
        x = triangle.loc[mask, c0].astype(float)
        y = triangle.loc[mask, c1].astype(float)
        r = (y / x).replace([np.inf, -np.inf], np.nan).dropna().values
        ratios[(c0, c1)] = r
    return ratios

def _bootstrap_factor_from_ratios(rng: np.random.Generator, r: np.ndarray, method: str) -> float:
    """
    Genera un factor bootstrap re-muestreando ratios (con reemplazo).
    """
    if r is None or len(r) == 0:
        return np.nan
    sample = rng.choice(r, size=len(r), replace=True)

    if method == "Ponderado (volumen)":
        # En demo, aproximamos ponderado con media de ratios (simplemente).
        # Si quisieras exacto ponderado, habría que re-muestrear pares (x,y).
        return float(np.nanmean(sample))
    elif method == "Promedio simple":
        return float(np.nanmean(sample))
    elif method == "Mediana":
        return float(np.nanmedian(sample))
    else:
        return float(np.nanmean(sample))

@st.cache_data(show_spinner=False)
def bootstrap_distribution(
    triangle: pd.DataFrame,
    method: str,
    tail_factor: float,
    annual_inflation: float,
    horizon_years: float,
    n_sims: int,
    seed: int
):
    """
    Retorna DataFrame con distribuciones de IBNR_total y Ultimate_total (base y stress).
    Cacheado para que sea usable en Cloud.
    """
    rng = np.random.default_rng(seed)
    devs = list(triangle.columns)
    ratios = _ratios_by_dev(triangle)

    # Precomputar Latest por AY (determinista)
    latest_by_ay = {}
    last_dev_by_ay = {}
    for ay in triangle.index:
        row = triangle.loc[ay]
        ld = row.last_valid_index()
        if ld is None:
            continue
        latest_by_ay[int(ay)] = float(row[ld])
        last_dev_by_ay[int(ay)] = int(ld)

    ays = sorted(latest_by_ay.keys())
    latest_total = sum(latest_by_ay.values())

    infl_factor = (1.0 + annual_inflation) ** horizon_years

    out = []
    for s in range(n_sims):
        # 1) Simular factores por edad (bootstrap)
        f_sim = []
        for i in range(len(devs) - 1):
            c0, c1 = devs[i], devs[i + 1]
            f_sim.append(_bootstrap_factor_from_ratios(rng, ratios.get((c0, c1), np.array([])), method))
        f_sim = np.array(f_sim, dtype=float)

        # Reemplazar NaNs por 1.0 para estabilidad (si falta data)
        f_sim = np.where(np.isnan(f_sim), 1.0, f_sim)

        # 2) Calcular Ultimate e IBNR total simulados
        ultimate_total = 0.0
        for ay in ays:
            ld = last_dev_by_ay[ay]
            idx = devs.index(ld)
            base_ldf = np.prod(f_sim[idx:]) if idx < len(f_sim) else 1.0
            ldf_total = float(base_ldf) * float(tail_factor)
            ultimate_total += latest_by_ay[ay] * ldf_total

        ibnr_total = ultimate_total - latest_total

        # 3) Stress inflación sobre IBNR
        ibnr_stress = ibnr_total * infl_factor
        ultimate_stress = latest_total + ibnr_stress

        out.append({
            "sim": s + 1,
            "Latest_total": latest_total,
            "Ultimate_total": ultimate_total,
            "IBNR_total": ibnr_total,
            "Ultimate_stress": ultimate_stress,
            "IBNR_stress": ibnr_stress
        })

    dist = pd.DataFrame(out)
    return dist

def percentile_summary(dist: pd.DataFrame, col: str):
    qs = [0.50, 0.75, 0.90, 0.95]
    vals = dist[col].quantile(qs).to_dict()
    return {
        "P50": vals[0.50],
        "P75": vals[0.75],
        "P90": vals[0.90],
        "P95": vals[0.95],
        "Media": dist[col].mean(),
        "Std": dist[col].std(ddof=1)
    }

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

st.sidebar.markdown("---")
st.sidebar.subheader("Bootstrap (incertidumbre)")

do_bootstrap = st.sidebar.checkbox("Activar bootstrap", value=True)
n_sims = st.sidebar.slider("Número de simulaciones", 100, 2000, 400, 100)
boot_seed = st.sidebar.number_input("Seed", value=123, step=1)

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
st.caption("Datos simulados con fines ilustrativos. Objetivo: sensibilidad a supuestos + incertidumbre (bootstrap).")

triangle = load_triangle(db_path, measure, lob, product, segment, region, currency)

st.subheader("Triángulo acumulado (preview)")
if triangle.empty:
    st.info("No hay datos para la combinación seleccionada. Cambia los filtros.")
    st.stop()

st.dataframe(triangle.round(0).head(10), use_container_width=True)
heat = px.imshow(triangle, aspect="auto", title="Heatmap del triángulo (valores acumulados)")
st.plotly_chart(heat, use_container_width=True)

if run:
    # Base determinista
    factors = chain_ladder_factors(triangle, factor_method)
    results = project_ultimate(triangle, factors, tail_factor=tail_factor)
    stress = apply_inflation_stress(results, annual_inflation=infl, horizon_years=horizon)

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest total", f"{results['Latest'].sum():,.0f}")
    c2.metric("IBNR total (base)", f"{results['IBNR'].sum():,.0f}")
    c3.metric("Ultimate total (base)", f"{results['Ultimate'].sum():,.0f}")
    c4.metric("Ultimate total (stress)", f"{stress['Ultimate_stress'].sum():,.0f}")

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

    # Waterfall stress
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

    fig = px.bar(results, x="Accident Year", y="IBNR", title="IBNR por Accident Year (base)")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.bar(stress, x="Accident Year", y="IBNR_stress", title="IBNR por Accident Year (stress)")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Factores Chain Ladder")
    fdf = factors.reset_index()
    fdf.columns = ["Desarrollo", "Factor"]
    st.dataframe(fdf.style.format({"Factor": "{:.4f}"}), use_container_width=True)

    # =============================
    # Bootstrap outputs (Incremento 3)
    # =============================
    if do_bootstrap:
        st.subheader("Incertidumbre (Bootstrap)")
        with st.spinner("Corriendo bootstrap…"):
            dist = bootstrap_distribution(
                triangle=triangle,
                method=factor_method,
                tail_factor=tail_factor,
                annual_inflation=infl,
                horizon_years=horizon,
                n_sims=int(n_sims),
                seed=int(boot_seed),
            )

        # Percentiles
        sum_u = percentile_summary(dist, "Ultimate_total")
        sum_i = percentile_summary(dist, "IBNR_total")
        sum_us = percentile_summary(dist, "Ultimate_stress")
        sum_is = percentile_summary(dist, "IBNR_stress")

        st.markdown("**Percentiles (Base):**")
        ptab = pd.DataFrame({
            "Ultimate_total": sum_u,
            "IBNR_total": sum_i,
        })
        st.dataframe(ptab.applymap(lambda x: f"{x:,.0f}"), use_container_width=True)

        st.markdown("**Percentiles (Stress):**")
        ptab2 = pd.DataFrame({
            "Ultimate_stress": sum_us,
            "IBNR_stress": sum_is,
        })
        st.dataframe(ptab2.applymap(lambda x: f"{x:,.0f}"), use_container_width=True)

        # Histogramas
        h1 = px.histogram(dist, x="IBNR_total", nbins=40, title="Distribución IBNR total (base)")
        st.plotly_chart(h1, use_container_width=True)

        h2 = px.histogram(dist, x="Ultimate_total", nbins=40, title="Distribución Ultimate total (base)")
        st.plotly_chart(h2, use_container_width=True)

        h3 = px.histogram(dist, x="IBNR_stress", nbins=40, title="Distribución IBNR total (stress)")
        st.plotly_chart(h3, use_container_width=True)

        # Línea de percentiles sobre hist (marcas)
        for label, q in [("P50", 0.50), ("P75", 0.75), ("P90", 0.90), ("P95", 0.95)]:
            v = dist["IBNR_total"].quantile(q)
            h1.add_vline(x=float(v), annotation_text=label, annotation_position="top")

        # Mostrar una muestra
        st.markdown("**Muestra de simulaciones (primeras 20):**")
        st.dataframe(dist.head(20).round(0), use_container_width=True)

    else:
        st.info("Bootstrap desactivado. Actívalo en el sidebar para ver percentiles y distribuciones.")
