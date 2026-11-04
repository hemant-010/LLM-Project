"""Create five-session market-adjusted stock-price labels using yfinance.

Baseline Price Selection Architecture:
1. Pre-Market Events (published_at < 09:30 ET on a trading day):
   - Baseline price (d0) is set to the previous trading session's close.
   - Outcome horizon (d5) is the 5th trading session following d0 (including event day).
2. Intraday Events (09:30 <= published_at < 16:00 ET on a trading day):
   - When intraday high-frequency bars are unavailable from daily feeds, uses the
     event day's close (post-publication), explicitly measuring the 5 sessions of
     post-announcement drift and excluding the same-day reaction.
   - Optional parameter 'intraday_baseline=prior_close' captures same-day reaction.
3. After-Hours Events (published_at >= 16:00 ET on a trading day):
   - Baseline price (d0) is set to the event day's close (established before the event).
   - Outcome horizon (d5) is the 5th trading session following d0.
4. Non-Trading Days (Weekends & Exchange Holidays):
   - Baseline price (d0) is set to the latest close prior to the event (e.g. Friday close).
   - Outcome horizon (d5) is the 5th trading session following d0.
"""
import argparse
from datetime import time
from pathlib import Path
import pandas as pd
import yfinance as yf

def determine_baseline_and_target_dates(pub_dt, trading_days, intraday_baseline="post_close"):
    """Determine d0 (baseline price date), d5 (5-trading-session outcome date), and horizon metadata

    conditioned on the publication timestamp in America/New_York.
    """
    pub_et = pub_dt.tz_convert("America/New_York") if pub_dt.tzinfo else pub_dt.tz_localize("UTC").tz_convert("America/New_York")
    pub_date = pub_et.date()
    pub_time = pub_et.time()

    market_open = time(9, 30)
    market_close = time(16, 0)
    td_dates = [d.date() for d in trading_days]
    is_trading_day = pub_date in td_dates

    if is_trading_day:
        if pub_time < market_open:
            prior = [d for d in trading_days if d.date() < pub_date]
            if not prior:
                return None, None, {"error": "insufficient_prior_history"}
            d0 = prior[-1]
            future = [d for d in trading_days if d.date() > d0.date()]
            if len(future) < 5:
                return None, None, {"error": "insufficient_future_sessions"}
            d5 = future[4]
            timing_info = {
                "baseline_timing_regime": "pre_market_baseline_prior_close",
                "horizon_sessions": 5,
                "horizon_type": "5_sessions_full_exposure",
                "first_trading_session": str(future[0].date()),
                "includes_announcement_day": True,
                "is_homogeneous_5session_label": True
            }
        elif pub_time >= market_close:
            prior = [d for d in trading_days if d.date() <= pub_date]
            if not prior:
                return None, None, {"error": "insufficient_prior_history"}
            d0 = prior[-1]
            future = [d for d in trading_days if d.date() > d0.date()]
            if len(future) < 5:
                return None, None, {"error": "insufficient_future_sessions"}
            d5 = future[4]
            timing_info = {
                "baseline_timing_regime": "after_hours_baseline_same_day_close",
                "horizon_sessions": 5,
                "horizon_type": "5_sessions_full_exposure",
                "first_trading_session": str(future[0].date()),
                "includes_announcement_day": False,
                "is_homogeneous_5session_label": True
            }
        else:
            if intraday_baseline == "prior_close":
                prior = [d for d in trading_days if d.date() < pub_date]
                if not prior:
                    return None, None, {"error": "insufficient_prior_history"}
                d0 = prior[-1]
                future = [d for d in trading_days if d.date() > d0.date()]
                if len(future) < 5:
                    return None, None, {"error": "insufficient_future_sessions"}
                d5 = future[4]
                timing_info = {
                    "baseline_timing_regime": "intraday_baseline_prior_close_same_day_included",
                    "horizon_sessions": 5,
                    "horizon_type": "5_sessions_announcement_and_drift",
                    "first_trading_session": str(future[0].date()),
                    "includes_announcement_day": True,
                    "is_homogeneous_5session_label": False
                }
            else:
                # Default: use post-publication close, explicitly measuring post-close drift
                prior = [d for d in trading_days if d.date() <= pub_date]
                if not prior:
                    return None, None, {"error": "insufficient_prior_history"}
                d0 = prior[-1]
                future = [d for d in trading_days if d.date() > d0.date()]
                if len(future) < 5:
                    return None, None, {"error": "insufficient_future_sessions"}
                d5 = future[4]
                timing_info = {
                    "baseline_timing_regime": "intraday_baseline_post_close_same_day_excluded",
                    "horizon_sessions": 5,
                    "horizon_type": "5_sessions_post_close_drift",
                    "first_trading_session": str(future[0].date()),
                    "includes_announcement_day": False,
                    "is_homogeneous_5session_label": False
                }
    else:
        prior = [d for d in trading_days if d.date() < pub_date]
        if not prior:
            return None, None, {"error": "insufficient_prior_history"}
        d0 = prior[-1]
        future = [d for d in trading_days if d.date() > d0.date()]
        if len(future) < 5:
            return None, None, {"error": "insufficient_future_sessions"}
        d5 = future[4]
        timing_info = {
            "baseline_timing_regime": "non_trading_day_baseline_prior_close",
            "horizon_sessions": 5,
            "horizon_type": "5_sessions_full_exposure",
            "first_trading_session": str(future[0].date()),
            "includes_announcement_day": False,
            "is_homogeneous_5session_label": True
        }

    return d0, d5, timing_info

def build(
    input_path: str | Path,
    output_path: str | Path,
    threshold_pct: float = 2.0,
    intraday_baseline: str = "post_close",
    filter_horizon: str = "all"
) -> pd.DataFrame:
    events = pd.read_csv(input_path)
    events["published_at"] = pd.to_datetime(events["published_at"], utc=True, errors="coerce")
    rows = []

    for ticker, group in events.dropna(subset=["published_at"]).groupby("ticker"):
        ticker = str(ticker).upper().strip()
        if not ticker:
            continue

        start = (group.published_at.min() - pd.Timedelta(days=15)).date()
        end = (group.published_at.max() + pd.Timedelta(days=45)).date()

        try:
            data = yf.download([ticker, "SPY"], start=start, end=end, auto_adjust=True, progress=False, group_by="ticker")
        except Exception as e:
            print(f"Warning: Failed to download market prices for {ticker}: {e}")
            continue

        if data.empty:
            continue

        def close_for(symbol):
            try:
                x = data[symbol]["Close"]
                if isinstance(x, pd.DataFrame):
                    x = x.iloc[:, 0]
                return x.dropna()
            except Exception:
                return pd.Series(dtype=float)

        stock = close_for(ticker)
        market = close_for("SPY")
        common = stock.index.intersection(market.index).sort_values()
        stock = stock.loc[common]
        market = market.loc[common]

        if len(common) < 7:
            continue

        trading_days = list(common)

        for _, event in group.iterrows():
            d0, d5, timing_info = determine_baseline_and_target_dates(
                event.published_at, trading_days, intraday_baseline=intraday_baseline
            )

            if d0 is None or d5 is None:
                continue

            # Optional horizon filtering to prevent mixing incompatible horizons
            if filter_horizon == "full_exposure_only" and timing_info.get("horizon_type") != "5_sessions_full_exposure":
                continue
            if filter_horizon == "post_close_drift_only" and timing_info.get("horizon_type") != "5_sessions_post_close_drift":
                continue

            p0_stock = stock.loc[d0]
            p5_stock = stock.loc[d5]
            p0_mkt = market.loc[d0]
            p5_mkt = market.loc[d5]

            if p0_stock <= 0 or p0_mkt <= 0:
                continue

            stock_ret = (p5_stock / p0_stock - 1) * 100
            market_ret = (p5_mkt / p0_mkt - 1) * 100
            adjusted = stock_ret - market_ret
            label = "increase" if adjusted > threshold_pct else "decrease" if adjusted < -threshold_pct else "no_significant_change"

            r = event.to_dict()
            r.update({
                "price_start_date": str(d0.date()),
                "price_end_date": str(d5.date()),
                "first_trading_session": timing_info["first_trading_session"],
                "stock_return_pct": round(stock_ret, 4),
                "benchmark_return_pct": round(market_ret, 4),
                "market_adjusted_return_pct": round(adjusted, 4),
                "stock_price_label": label,
                "horizon_sessions": timing_info["horizon_sessions"],
                "horizon_type": timing_info["horizon_type"],
                "baseline_timing_regime": timing_info["baseline_timing_regime"],
                "includes_announcement_day": timing_info["includes_announcement_day"],
                "is_homogeneous_5session_label": timing_info["is_homogeneous_5session_label"]
            })
            rows.append(r)

    result = pd.DataFrame(rows)
    if not result.empty and result["horizon_type"].nunique() > 1:
        types = result["horizon_type"].value_counts().to_dict()
        print(f"NOTICE: Price labels contain multiple horizon specifications: {types}. Use 'horizon_type' or 'is_homogeneous_5session_label' to prevent mixing in model training.")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Wrote {len(result)} price-labelled rows to {output_path}")
    return result

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="data/processed/events_with_price_labels.csv")
    p.add_argument("--threshold-pct", type=float, default=2.0)
    p.add_argument("--intraday-baseline", choices=["post_close", "prior_close"], default="post_close",
                   help="Baseline price timing for intraday events ('post_close' excludes same-day reaction; 'prior_close' captures same-day reaction)")
    p.add_argument("--filter-horizon", choices=["all", "full_exposure_only", "post_close_drift_only"], default="all",
                   help="Filter output to prevent mixing incompatible prediction horizons in training datasets")
    a = p.parse_args()
    build(a.input, a.output, a.threshold_pct, intraday_baseline=a.intraday_baseline, filter_horizon=a.filter_horizon)
