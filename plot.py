"""Batch K-line plotter: one image per N trading days, all contracts of a variety.

Usage: edit the CONFIG block below, then run the file directly. No CLI args.

Output file name:  <VARIETY>-<startdate>-<enddate>.png   (dates = YYYYMMDD of the
                   first / last trading day inside that image)
"""

from __future__ import annotations

from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text


# =============================================================================
# CONFIG - edit only this section
# =============================================================================


# --- run parameters ---
VARIETY: str | list[str] = "cj"   # 品种，如 "m"；也可写多个 ["m", "RM"]
START_DATE = "2025-01-01"        # 开始时间（含）
END_DATE = "2025-12-31"          # 结束时间（含）
WINDOW_DAYS = 20                 # N：每个交易日为 1 根K线，每 N 个交易日输出一张图
OUTPUT_DIR = Path(__file__).resolve().parent / "kline_windows"  # 输出路径

# --- layout / style ---
SUBPLOT_COLS = 4        # 每张图的子图列数（行数按合约数量自动算）
EQUAL_SPACING = True    # True = 交易日等距排列（跳过节假日/长假空白）；False = 真实日期轴
MAX_TICKS = 12          # 每张小图横轴最多显示多少个日期标签
DPI = 200               # 输出图片分辨率
BACKFILL_LAST_WINDOW = True   # True = 最后不足 N 个交易日时，从结束日往前取 N 个交易日（与上一张有重叠）
                              # False = 保留末尾不足 N 天的短窗口

# --- data cleaning ---
DEDUP_ON_KEY = True     # 原始表可能有重复 (date, contract) 行，默认去重
FILTER_ZERO_VOLUME = True  # True = 丢弃成交量为 0 的行（未成交的占位行情）
# =============================================================================


def parse_trade_date(series: pd.Series) -> pd.Series:
    """Parse trade date, supporting both YYYYMMDD and YYYY-MM-DD."""
    s = series.copy()
    s_str = s.astype(str).str.strip()
    parsed = pd.to_datetime(s_str, format="%Y%m%d", errors="coerce")
    mask_missing = parsed.isna()
    if mask_missing.any():
        parsed.loc[mask_missing] = pd.to_datetime(s.loc[mask_missing], errors="coerce")
    return parsed


def load_data(varieties: list[str], start: str, end: str) -> pd.DataFrame:
    """Load daily bars for the given varieties inside [start, end]."""
    # The DB stores mixed-case variety codes ("al" vs "IF"), so match all case
    # variants of every requested variety.
    variants = sorted({v for ins in varieties for v in (ins, ins.upper(), ins.lower())})
    placeholders = ", ".join(f":v{i}" for i in range(len(variants)))

    q = text(
        f"SELECT date, instrument, instrumentid, open, high, low, close, volume "
        f"FROM {TABLE} "
        f"WHERE instrument IN ({placeholders}) AND date BETWEEN :lo AND :hi"
    )
    params = {f"v{i}": v for i, v in enumerate(variants)}
    params["lo"] = start
    params["hi"] = end

    df = pd.read_sql(q, engine, params=params)
    if df.empty:
        raise ValueError(
            f"No rows for variety {varieties} between {start} and {end}. "
            "Check the variety code and the table's data coverage."
        )

    df["date"] = parse_trade_date(df["date"])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["date", "open", "high", "low", "close"]).copy()
    df["contract"] = df["instrumentid"].astype(str)

    if FILTER_ZERO_VOLUME:
        df = df[df["volume"].fillna(0) > 0].copy()

    if DEDUP_ON_KEY:
        before = len(df)
        df = df.drop_duplicates(subset=["date", "contract"], keep="last").copy()
        if before != len(df):
            print(f"Removed {before - len(df):,} duplicate (date, contract) rows.")

    return df.sort_values(["date", "contract"]).reset_index(drop=True)


def draw_candles(ax, data: pd.DataFrame) -> None:
    """Draw daily candlesticks (red up / green down, CN convention) on one axes."""
    from matplotlib.patches import Rectangle

    dates = list(data["date"])
    n = len(dates)

    if EQUAL_SPACING:
        x = np.arange(n, dtype=float)
    else:
        import matplotlib.dates as mdates
        x = mdates.date2num(pd.to_datetime(dates))

    body_width = 0.6
    for xi, o, h, l, c in zip(x, data["open"], data["high"], data["low"], data["close"]):
        up = c >= o
        color = "#d62728" if up else "#2ca02c"  # 涨红跌绿
        ax.vlines(xi, l, h, color=color, linewidth=1.0, alpha=0.9)
        y = min(o, c)
        h_body = max(abs(c - o), 1e-9)
        ax.add_patch(
            Rectangle(
                (xi - body_width / 2, y), body_width, h_body,
                facecolor=color, edgecolor=color, alpha=0.85,
            )
        )

    if EQUAL_SPACING:
        step = max(1, ceil(n / MAX_TICKS))
        ticks = list(range(0, n, step))
        ax.set_xticks(ticks)
        ax.set_xticklabels(
            [d.strftime("%m-%d") for d in dates[::step]], rotation=45, fontsize=7
        )
        ax.set_xlim(-0.8, n - 0.2)
    else:
        import matplotlib.dates as mdates
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)

    ax.tick_params(axis="y", labelsize=7)
    ax.grid(alpha=0.2)
    ax.margins(y=0.08)


def plot_window(
    sub: pd.DataFrame,
    window_dates: list[pd.Timestamp],
    contracts: list[str],
    variety_label: str,
    out_path: Path,
) -> None:
    """Plot every contract of `sub` as one subplot, save to `out_path`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_contracts = max(1, len(contracts))
    cols = min(SUBPLOT_COLS, n_contracts)
    rows = ceil(n_contracts / cols)

    cell_w, cell_h = 4.4, 2.8
    cell_h = min(cell_h, 40.0 / rows)  # keep the figure inside matplotlib's size limit

    fig, axes = plt.subplots(
        rows, cols, figsize=(cell_w * cols, cell_h * rows + 0.6), squeeze=False
    )
    axes_list = axes.reshape(-1)

    for i, ct in enumerate(contracts):
        ax = axes_list[i]
        ct_data = sub[sub["contract"] == ct].sort_values("date")
        if ct_data.empty:
            ax.axis("off")
            ax.set_title(f"{ct} (no data)", fontsize=9, color="gray")
            continue
        draw_candles(ax, ct_data)
        ax.set_title(ct, fontsize=10)

    for j in range(len(contracts), len(axes_list)):
        axes_list[j].axis("off")

    w_start = window_dates[0].strftime("%Y-%m-%d")
    w_end = window_dates[-1].strftime("%Y-%m-%d")
    fig.suptitle(
        f"{variety_label}  {w_start} ~ {w_end}  ({len(window_dates)} trading days)",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def main() -> None:
    varieties = [VARIETY] if isinstance(VARIETY, str) else list(VARIETY)
    varieties = [v.upper() for v in varieties]
    variety_label = "_".join(varieties)

    df = load_data(varieties, START_DATE, END_DATE)

    all_dates = sorted(df["date"].unique())
    all_dates = [pd.Timestamp(d) for d in all_dates]
    contracts = sorted(df["contract"].unique())

    print(f"Varieties            : {variety_label}")
    print(f"Date range requested : {START_DATE} ~ {END_DATE}")
    print(f"Trading days found   : {len(all_dates)} "
          f"({all_dates[0].date()} ~ {all_dates[-1].date()})")
    print(f"Contracts found      : {len(contracts)} -> {contracts}")
    print(f"Window size N        : {WINDOW_DAYS} trading days")

    windows = [all_dates[i:i + WINDOW_DAYS] for i in range(0, len(all_dates), WINDOW_DAYS)]

    # If the last chunk is short (fewer than N trading days), replace it with the
    # trailing N trading days counted back from END_DATE. This window then
    # overlaps the previous one instead of ending with a partial picture.
    if BACKFILL_LAST_WINDOW and len(all_dates) > WINDOW_DAYS and len(windows[-1]) < WINDOW_DAYS:
        replaced = windows[-1]
        windows[-1] = all_dates[-WINDOW_DAYS:]
        print(
            f"Last window backfilled: {replaced[0].date()} ~ {replaced[-1].date()} "
            f"({len(replaced)} days) -> {windows[-1][0].date()} ~ {windows[-1][-1].date()} "
            f"({len(windows[-1])} days, overlaps previous window)"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Images to write      : {len(windows)} -> {OUTPUT_DIR}\n")

    for w in windows:
        sub = df[df["date"].isin(w)].copy()
        name = (f"{variety_label}-{w[0].strftime('%Y%m%d')}-{w[-1].strftime('%Y%m%d')}.png")
        out_path = OUTPUT_DIR / name
        plot_window(sub, w, contracts, variety_label, out_path)
        print(f"  saved: {name}  ({len(w)} trading days)")

    print(f"\nDone. {len(windows)} image(s) saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
