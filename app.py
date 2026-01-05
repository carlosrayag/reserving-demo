import os
import sys
import subprocess
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine, text

st.set_page_config(
    page_title="Simulador de Reservas – Chain Ladder",
    page_icon="📊",
    layout="wide"
)

DEFAULT_DB = "reserving_demo.db"

def ensure_db(db_path: str):
    if not os.path.exists(db_path):
        subprocess.check_call([
            sys.executable,
            "create_db.py",
            "--db", db_path,
            "--seed", "42",
            "--rebuild"
        ])

@st.cache_resource
def get_engine(db_path):
    return create_engine(f"sqlite:///{db_path}")

@st.cache_data
def fetch_distinct(db_path, column):
    engine = get_engine(db_path)
    with engine.begin() as conn:
        q = text(f"SELECT DISTINCT {column} FROM claims_triangle ORDER BY {column}")
        return [r[0] for r in conn.execute(q)]

@st.cache_data
def load_triangle(db_path, measure, lob, product, segment, region, currency):
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
    return df.pivot(index="accident_year", columns="dev_month", values="value")

def chain_ladder_factors(triangle):
    factors = {}
    devs = list(triangle.columns)
    for i in range(len(devs) - 1):
        c0, c1 = devs[i], devs[i + 1]
        mask = triangle[c0].notna() & triangle[c1].notna()
        factors[f"{c0}->{c1}"] = triangle.loc[mask, c1].sum() / triangle.loc[mask, c0].sum()
    return pd.Series(factors)

def project_ultimate(triangle, factors):
    devs = list(triangle.columns)
    rows = []
    for ay in triangle.index:
        row = triangle.loc[ay]
        last_dev = row.last_valid_index()
        last_val = row[last_dev]
        idx = devs.index(last_dev)
        ldf = np.prod(factors.iloc[idx:]) if idx < len(factors) else 1.0
        ultimate = last_val * ldf
        rows.append({
            "Accident Year": ay,
            "Latest": last_val,
            "Ultimate": ultimate,
            "IBNR": ultimate - last_val
        })
    return pd.DataFrame(rows)

st.sidebar.title("Parámetros")

db_path = st.sidebar.text_input("Base de datos", DEFAULT_DB)
ensure_db(db_path)

measure = st.sidebar.selectbox("Medida", ["Incurred", "Paid"])

lob = st.sidebar.selectbox("Línea", fetch_distinct(db_path, "lob_code"))
product = st.sidebar.selectbox("Producto", fetch_distinct(db_path, "product_code"))
segment = st.sidebar.selectbox("Segmento", fetch_distinct(db_path, "segment_code"))
region = st.sidebar.selectbox("Región", fetch_distinct(db_path, "region_code"))
currency = st.sidebar.selectbox("Moneda", fetch_distinct(db_path, "currency"))

run = st.sidebar.button("Ejecutar simulación")

st.title("📊 Simulador de Reservas – Chain Ladder")
st.caption(
    "Los datos mostrados son simulados y utilizados únicamente con fines ilustrativos."
)

triangle = load_triangle(db_path, measure, lob, product, segment, region, currency)

st.subheader("Triángulo acumulado")
st.dataframe(triangle.style.format("{:,.0f}"), use_container_width=True)

if run:
    factors = chain_ladder_factors(triangle)
    results = project_ultimate(triangle, factors)

    c1, c2, c3 = st.columns(3)
    c1.metric("Latest total", f"{results['Latest'].sum():,.0f}")
    c2.metric("Ultimate total", f"{results['Ultimate'].sum():,.0f}")
    c3.metric("IBNR total", f"{results['IBNR'].sum():,.0f}")

    st.subheader("Resultados por año de ocurrencia")
    st.dataframe(results.style.format("{:,.0f}"), use_container_width=True)

    fig = px.bar(results, x="Accident Year", y="IBNR", title="IBNR por Accident Year")
    st.plotly_chart(fig, use_container_width=True)
