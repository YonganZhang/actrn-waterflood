"""
P2.3 复杂网络/相关图基线 (Mayoral-Villa et al. 2025 Fluids 风格)

核心: 把每口井当节点, 边权 = 时序相关性 / 同步指数。
本实现:
  - Pearson 相关 (无方向)
  - 滞后 Cross-Correlation 取峰值 (含方向, 但仅基于相关性)
  - 时间窗 sliding (查看连通是否稳定)

注: 该基线无干预语义, 无反事实能力 — 这正是我们论文要打的点。

输入: data/processed/volve_wide_oil.parquet, volve_wide_inj.parquet
输出: data/processed/network_corr.csv, network_lagcorr.csv
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / 'data/processed'

MAX_LAG = 30  # 天


def pearson_matrix(inj: pd.DataFrame, oil: pd.DataFrame) -> pd.DataFrame:
    """注水量 vs 产油量 同期 Pearson"""
    out = pd.DataFrame(index=inj.columns, columns=oil.columns, dtype=float)
    for i in inj.columns:
        for j in oil.columns:
            mask = inj[i].notna() & oil[j].notna()
            if mask.sum() < 100:
                continue
            out.loc[i, j] = inj[i][mask].corr(oil[j][mask])
    return out


def lagcorr_peak(inj: pd.DataFrame, oil: pd.DataFrame, max_lag=MAX_LAG):
    """对每对 (inj, oil), 让 inj 提前 0..max_lag 天, 取最大相关 + 对应 lag."""
    peak_r = pd.DataFrame(index=inj.columns, columns=oil.columns, dtype=float)
    peak_lag = pd.DataFrame(index=inj.columns, columns=oil.columns, dtype=float)
    for i in inj.columns:
        for j in oil.columns:
            r_list = []
            for lag in range(0, max_lag + 1):
                a = inj[i].shift(lag)
                b = oil[j]
                mask = a.notna() & b.notna()
                if mask.sum() < 100:
                    r_list.append(np.nan)
                    continue
                r_list.append(a[mask].corr(b[mask]))
            arr = np.array(r_list)
            if np.all(np.isnan(arr)):
                continue
            k = int(np.nanargmax(arr))
            peak_r.loc[i, j] = arr[k]
            peak_lag.loc[i, j] = k
    return peak_r, peak_lag


def main():
    oil = pd.read_parquet(PROC / 'volve_wide_oil.parquet')
    inj = pd.read_parquet(PROC / 'volve_wide_inj.parquet')

    corr = pearson_matrix(inj, oil)
    print('=== Pearson (no lag) ===')
    print(corr.round(3))

    peak_r, peak_lag = lagcorr_peak(inj, oil)
    print(f'\n=== Lag-cross-correlation peak r (0-{MAX_LAG}d) ===')
    print(peak_r.round(3))
    print(f'\n=== Argmax lag (days) ===')
    print(peak_lag.astype('Int64'))

    corr.to_csv(PROC / 'network_corr.csv')
    peak_r.to_csv(PROC / 'network_lagcorr_r.csv')
    peak_lag.to_csv(PROC / 'network_lagcorr_lag.csv')
    print(f'\nSaved: network_corr.csv, network_lagcorr_{{r,lag}}.csv')


if __name__ == '__main__':
    main()
