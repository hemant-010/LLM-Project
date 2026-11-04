import React from 'react';
import { TrendingUp, TrendingDown, Minus, BarChart3, Clock, Database } from 'lucide-react';

export default function StatsOverview({ history = [] }) {
  const total = history.length;

  const countByDirection = history.reduce((acc, item) => {
    const dir = item?.short_interest?.direction;
    if (dir === 'increase') acc.increase = (acc.increase || 0) + 1;
    else if (dir === 'decrease') acc.decrease = (acc.decrease || 0) + 1;
    else if (dir === 'no_significant_change') acc.neutral = (acc.neutral || 0) + 1;
    return acc;
  }, { increase: 0, decrease: 0, neutral: 0 });

  const pctIncrease = total > 0 ? Math.round((countByDirection.increase / total) * 100) : 0;
  const pctDecrease = total > 0 ? Math.round((countByDirection.decrease / total) * 100) : 0;
  const pctNeutral = total > 0 ? Math.round((countByDirection.neutral / total) * 100) : 0;

  const latest = history[0];

  return (
    <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
      {/* Total Predictions Card */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 shadow-sm flex flex-col justify-between">
        <div className="flex items-center justify-between text-slate-400 mb-2">
          <span className="text-xs font-semibold uppercase tracking-wider">Session Predictions</span>
          <Database className="h-4 w-4 text-blue-400" />
        </div>
        <div>
          <div className="text-3xl font-extrabold text-white tracking-tight">{total}</div>
          <p className="text-[11px] text-slate-500 mt-1">
            {total === 0 ? "No predictions submitted yet" : "Real-time user queries evaluated"}
          </p>
        </div>
      </div>

      {/* Latest Prediction Card */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 shadow-sm flex flex-col justify-between">
        <div className="flex items-center justify-between text-slate-400 mb-2">
          <span className="text-xs font-semibold uppercase tracking-wider">Latest Output</span>
          <Clock className="h-4 w-4 text-indigo-400" />
        </div>
        <div>
          {latest ? (
            <div>
              <div className="flex items-center gap-2">
                <span className={`inline-flex items-center gap-1 text-sm font-bold uppercase px-2 py-0.5 rounded ${
                  latest.short_interest?.direction === 'increase'
                    ? 'bg-rose-500/10 text-rose-400 border border-rose-500/30'
                    : latest.short_interest?.direction === 'decrease'
                    ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30'
                    : 'bg-slate-700/30 text-slate-300 border border-slate-600/30'
                }`}>
                  {latest.short_interest?.direction === 'increase' && <TrendingUp className="h-3.5 w-3.5" />}
                  {latest.short_interest?.direction === 'decrease' && <TrendingDown className="h-3.5 w-3.5" />}
                  {latest.short_interest?.direction === 'no_significant_change' && <Minus className="h-3.5 w-3.5" />}
                  {latest.short_interest?.direction?.replace(/_/g, ' ') || 'Predicted'}
                </span>
                <span className="text-xs font-mono text-slate-400">[{latest.event?.ticker || 'N/A'}]</span>
              </div>
              <p className="text-[11px] text-slate-400 truncate mt-1.5" title={latest.event?.title}>
                {latest.event?.title}
              </p>
            </div>
          ) : (
            <div>
              <span className="text-sm text-slate-500 italic">Awaiting first input</span>
              <p className="text-[11px] text-slate-500 mt-1">Submit a headline below</p>
            </div>
          )}
        </div>
      </div>

      {/* Distribution Breakdown Card (spanning 2 columns) */}
      <div className="md:col-span-2 bg-slate-900/60 border border-slate-800 rounded-xl p-4 shadow-sm flex flex-col justify-between">
        <div className="flex items-center justify-between text-slate-400 mb-2">
          <span className="text-xs font-semibold uppercase tracking-wider">Class Distribution (Actual Outputs)</span>
          <BarChart3 className="h-4 w-4 text-cyan-400" />
        </div>

        {total > 0 ? (
          <div>
            {/* Multi-segment progress bar */}
            <div className="h-2.5 w-full bg-slate-800 rounded-full overflow-hidden flex mb-2.5">
              <div 
                style={{ width: `${pctIncrease}%` }} 
                className="bg-rose-500 transition-all duration-500" 
                title={`Increase: ${countByDirection.increase} (${pctIncrease}%)`} 
              />
              <div 
                style={{ width: `${pctDecrease}%` }} 
                className="bg-emerald-500 transition-all duration-500" 
                title={`Decrease: ${countByDirection.decrease} (${pctDecrease}%)`} 
              />
              <div 
                style={{ width: `${pctNeutral}%` }} 
                className="bg-slate-600 transition-all duration-500" 
                title={`No Significant Change: ${countByDirection.neutral} (${pctNeutral}%)`} 
              />
            </div>

            {/* Labels */}
            <div className="grid grid-cols-3 text-xs gap-2">
              <div className="flex items-center gap-1.5">
                <span className="h-2 w-2 rounded-full bg-rose-500" />
                <span className="text-slate-400">Increase:</span>
                <strong className="text-slate-200">{countByDirection.increase} ({pctIncrease}%)</strong>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="h-2 w-2 rounded-full bg-emerald-500" />
                <span className="text-slate-400">Decrease:</span>
                <strong className="text-slate-200">{countByDirection.decrease} ({pctDecrease}%)</strong>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="h-2 w-2 rounded-full bg-slate-500" />
                <span className="text-slate-400">Neutral:</span>
                <strong className="text-slate-200">{countByDirection.neutral} ({pctNeutral}%)</strong>
              </div>
            </div>
          </div>
        ) : (
          <div className="py-2 text-center text-xs text-slate-500 italic">
            Statistical distribution will update live as predictions are generated.
          </div>
        )}
      </div>
    </div>
  );
}

