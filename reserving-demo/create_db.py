import argparse
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

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

def generate_synthetic_triangle(seed: int = 42):
    rng = np.random.default_rng(seed)

    devs = [12, 24, 36, 48, 60]
    ays = list(range(2018, 2026))  # 2018–2025

    # Patrones acumulados típicos (emergen hacia 1.0)
    incurred_pattern = np.array([0.55, 0.78, 0.90, 0.97, 1.00])
    paid_pattern = np.array([0.45, 0.70, 0.85, 0.95, 1.00])

    # Slices (para que el dashboard tenga filtros)
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
            # Ultimate con tendencia y ruido lognormal
            ult_mean = base_ultimate * ((1 + trend) ** (i - (len(ays)-1)))
            ultimate = ult_mean * rng.lognormal(mean=0.0, sigma=0.10)

            incurred_cum = ultimate * incurred_pattern * rng.lognormal(0.0, 0.03, size=len(devs))
            incurred_cum = np.maximum.accumulate(incurred_cum)

            paid_cum = ultimate * paid_pattern * rng.lognormal(0.0, 0.05, size=len(devs))
            paid_cum = np.maximum.accumulate(paid_cum)

            # Triángulo: AY recientes no tienen todo el desarrollo
            # (entre más reciente, menos columnas)
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

def build_db(db_path: str, seed: int = 42, rebuild: bool = False):
    engine = create_engine(f"sqlite:///{db_path}")

    with engine.begin() as conn:
        conn.execute(text(DDL))
        if rebuild:
            conn.execute(text("DELETE FROM claims_triangle"))

        # Insertar datos
        df = generate_synthetic_triangle(seed=seed)
        df.to_sql("claims_triangle", engine, if_exists="append", index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="reserving_demo.db")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    build_db(db_path=args.db, seed=args.seed, rebuild=True if args.rebuild else False)
    print(f"[OK] BD lista: {args.db}")

if __name__ == "__main__":
    main()
