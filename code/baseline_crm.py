"""
P2.2 CRM 基线复刻 (simplified Capacitance Resistance Model — Producer-based, CRMP)

参考: Sayarpour et al. 2009 SPE 110081 / Nguyen et al. 2011 SPE 147344

模型 (CRMP, 离散日步长):
  q_j(t) = q_j(t-1) * exp(-Δt/τ_j)
         + (1 - exp(-Δt/τ_j)) * Σ_i λ_{ij} * I_i(t)

其中:
  q_j(t)   = producer j 当日产油 (m³)
  I_i(t)   = injector i 当日注水 (m³)
  τ_j      = producer j 时间常数 (days)
  λ_{ij}   = 注水井 i → 产油井 j 连通系数
            Σ_j λ_{ij} ≤ 1 (物质平衡约束)
            λ_{ij} ≥ 0      (非负约束)

实现:
  - 把 τ_j 当超参数（grid 搜 5-100 天），每个 τ 下解非负最小二乘 (NNLS)
  - 输出每对 (i,j) 的 λ 矩阵 + 每口 producer 的 R²

输入: data/processed/volve_wide_oil.parquet, volve_wide_inj.parquet
输出: data/processed/crm_lambda.csv, crm_metrics.json
"""
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.optimize import nnls
import json

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / 'data/processed'


def fit_crmp_producer(q_series: pd.Series, inj_df: pd.DataFrame,
                      tau_grid=(5, 10, 20, 30, 45, 60, 90)):
    """对单口 producer 网格搜 τ，每个 τ 下解 NNLS λ_ij。返回最优 (τ, λ, R²)."""
    mask = q_series.notna() & inj_df.notna().all(axis=1)
    if mask.sum() < 100:
        return None
    q = q_series[mask].values
    I = inj_df.loc[mask].values  # shape (T, n_inj)
    T = len(q)
    best = None
    for tau in tau_grid:
        a = np.exp(-1.0 / tau)
        # 构造 CRMP 残差 = q(t) - a*q(t-1) = (1-a) * Σ λ_i * I_i(t)
        # 数值上等价于 NNLS: y = X λ, y = q(t) - a*q(t-1), X = (1-a) * I(t)
        if T < 2:
            continue
        y = q[1:] - a * q[:-1]
        X = (1.0 - a) * I[1:]
        lam, _ = nnls(X, y)
        y_hat = X @ lam
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        if best is None or r2 > best['r2']:
            best = {'tau': tau, 'lambda': lam, 'r2': r2, 'n_obs': T}
    return best


def main():
    oil = pd.read_parquet(PROC / 'volve_wide_oil.parquet')
    inj = pd.read_parquet(PROC / 'volve_wide_inj.parquet')
    # 缺失对齐: 仅在产油+注水同时有数的日子拟合
    print(f'oil cols: {list(oil.columns)}')
    print(f'inj cols: {list(inj.columns)}')

    results = {}
    lam_rows = []
    for prod in oil.columns:
        r = fit_crmp_producer(oil[prod], inj)
        if r is None:
            print(f'  {prod}: SKIP (<100 obs)')
            continue
        results[prod] = {'tau_days': r['tau'], 'r2': float(r['r2']), 'n_obs': r['n_obs']}
        for inj_well, lv in zip(inj.columns, r['lambda']):
            lam_rows.append({'injector': inj_well, 'producer': prod, 'lambda': float(lv)})
        print(f'  {prod}: τ={r["tau"]:>3}d, R²={r["r2"]:.3f}, n={r["n_obs"]}, '
              f'λ=[{", ".join(f"{x:.3f}" for x in r["lambda"])}]')

    lam_df = pd.DataFrame(lam_rows)
    pivot = lam_df.pivot(index='injector', columns='producer', values='lambda')
    print('\n=== CRM λ matrix (injector × producer) ===')
    print(pivot.round(3))

    pivot.to_csv(PROC / 'crm_lambda.csv')
    with open(PROC / 'crm_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: {PROC}/crm_lambda.csv + crm_metrics.json')


if __name__ == '__main__':
    main()
