"""
P2.1 Volve 数据预处理

输入: data/raw/volve/Volve production data.xlsx (Daily Production Data sheet)
输出: data/processed/volve_panel.parquet  — 注采井日级长表，已清洗对齐
      data/processed/volve_wide_oil.parquet — 产油井宽表（cols = 5 producers）
      data/processed/volve_wide_inj.parquet — 注水井宽表（cols = 2 injectors）

清洗约定:
  - 负值/异常值设为 NaN
  - 按井 ON_STREAM_HRS>0 切"在线时段"
  - 日级别 reindex 全时段，缺失保留 NaN（不前向填补，留给下游模型决定）
  - F-5 既注又采 → 拆成两条逻辑井 F-5_inj / F-5_op，按 FLOW_KIND 区分
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/volve/Volve production data.xlsx'
OUT = ROOT / 'data/processed'
OUT.mkdir(parents=True, exist_ok=True)

PRODUCERS = ['15/9-F-1 C', '15/9-F-11', '15/9-F-12', '15/9-F-14', '15/9-F-15 D']
INJECTORS = ['15/9-F-4', '15/9-F-5']  # F-5 主要是注水，少量产油子时段单独处理


def load_raw() -> pd.DataFrame:
    df = pd.read_excel(RAW, sheet_name='Daily Production Data')
    df['DATEPRD'] = pd.to_datetime(df['DATEPRD'])
    df = df.rename(columns={'NPD_WELL_BORE_NAME': 'well'})
    return df


def clean_volumes(df: pd.DataFrame) -> pd.DataFrame:
    for col in ['BORE_OIL_VOL', 'BORE_GAS_VOL', 'BORE_WAT_VOL', 'BORE_WI_VOL', 'ON_STREAM_HRS']:
        df.loc[df[col] < 0, col] = np.nan
    # 离线日（hrs=0 或 NaN）的产量也设 NaN
    offline = df['ON_STREAM_HRS'].isna() | (df['ON_STREAM_HRS'] == 0)
    for col in ['BORE_OIL_VOL', 'BORE_GAS_VOL', 'BORE_WAT_VOL', 'BORE_WI_VOL']:
        df.loc[offline, col] = np.nan
    return df


def build_panel(df: pd.DataFrame) -> pd.DataFrame:
    """日级长表，每行 (date, well_id, role, vars...)"""
    rows = []
    # 产油井
    for w in PRODUCERS:
        sub = df[(df['well'] == w) & (df['FLOW_KIND'] == 'production')].copy()
        sub['well_id'] = w
        sub['role'] = 'producer'
        rows.append(sub)
    # 注水井（F-4 全部, F-5 仅注水段）
    for w in INJECTORS:
        sub = df[(df['well'] == w) & (df['FLOW_KIND'] == 'injection')].copy()
        sub['well_id'] = w + ('_inj' if w == '15/9-F-5' else '')
        sub['role'] = 'injector'
        rows.append(sub)
    # F-5 产油段（少量，留作后续敏感性分析用）
    f5_op = df[(df['well'] == '15/9-F-5') & (df['FLOW_KIND'] == 'production')].copy()
    f5_op['well_id'] = '15/9-F-5_op'
    f5_op['role'] = 'producer_minor'
    rows.append(f5_op)

    panel = pd.concat(rows, ignore_index=True)
    keep = ['DATEPRD', 'well_id', 'role', 'ON_STREAM_HRS',
            'BORE_OIL_VOL', 'BORE_GAS_VOL', 'BORE_WAT_VOL', 'BORE_WI_VOL',
            'AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE',
            'AVG_DP_TUBING', 'AVG_ANNULUS_PRESS',
            'AVG_CHOKE_SIZE_P', 'AVG_WHP_P', 'AVG_WHT_P', 'DP_CHOKE_SIZE']
    return panel[keep].sort_values(['well_id', 'DATEPRD']).reset_index(drop=True)


def to_wide(panel: pd.DataFrame, role: str, value: str) -> pd.DataFrame:
    sub = panel[panel['role'] == role]
    wide = sub.pivot_table(index='DATEPRD', columns='well_id', values=value, aggfunc='first')
    # reindex 到全时段
    full_idx = pd.date_range(panel['DATEPRD'].min(), panel['DATEPRD'].max(), freq='D')
    wide = wide.reindex(full_idx)
    wide.index.name = 'date'
    return wide


def main():
    raw = load_raw()
    raw = clean_volumes(raw)
    panel = build_panel(raw)
    panel.to_parquet(OUT / 'volve_panel.parquet')

    oil_wide = to_wide(panel, 'producer', 'BORE_OIL_VOL')
    inj_wide = to_wide(panel, 'injector', 'BORE_WI_VOL')
    oil_wide.to_parquet(OUT / 'volve_wide_oil.parquet')
    inj_wide.to_parquet(OUT / 'volve_wide_inj.parquet')

    print(f'panel rows={len(panel)}  date={panel["DATEPRD"].min()}→{panel["DATEPRD"].max()}')
    print(f'oil_wide shape={oil_wide.shape}  non-null per col:')
    print(oil_wide.notna().sum())
    print(f'inj_wide shape={inj_wide.shape}  non-null per col:')
    print(inj_wide.notna().sum())


if __name__ == '__main__':
    main()
