import React from 'react';
import { TrendingUp, TrendingDown, Minus, Activity, Clock, Layers } from 'lucide-react';

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
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      {/* 1. Total Predictions */}
      <div className="bg-[#242720] border border-[#393C33] rounded p-4 flex flex-col justify-between">
        <div className="flex items-center justify-between text-[#85877C] pb-2 border-b border-[#393C33]">
          <span className="text-[11px] font-mono uppercase tracking-wider">
            01 / SESSION VOLUME
          </span>
          <Activity className="h-3.5 w-3.5 text-[#A18B62]" />
        </div>
        <div className="py-3">
          <div className="text-3xl font-mono font-semibold text-[#E8E4D9] tracking-tight">
            {total.toString().padStart(2, '0')}
          </div>
          <p className="text-[11px] text-[#AAA99D] mt-1 font-mono">
            {total === 0 ? "Awaiting evaluation inputs" : `${total} corporate catalysts evaluated`}
          </p>
        </div>
        <div className="pt-2 border-t border-[#393C33] flex items-center justify-between text-[10px] font-mono text-[#85877C]">
          <span>STORAGE: IN-MEMORY SESSION</span>
          <span className="text-[#A18B62]">ACTIVE</span>
        </div>
      </div>

      {/* 2. Latest Prediction */}
      <div className="bg-[#242720] border border-[#393C33] rounded p-4 flex flex-col justify-between">
        <div className="flex items-center justify-between text-[#85877C] pb-2 border-b border-[#393C33]">
          <span className="text-[11px] font-mono uppercase tracking-wider">
            02 / LATEST FORECAST
          </span>
          <Clock className="h-3.5 w-3.5 text-[#A18B62]" />
        </div>
        <div className="py-2.5">
          {latest ? (
            <div>
              <div className="flex items-center gap-2 mb-1">
                <span className={`inline-flex items-center gap-1 text-[11px] font-mono uppercase px-2 py-0.5 rounded border ${
                  latest.short_interest?.direction === 'increase'
                    ? 'border-[#B66F58]/40 bg-[#B66F58]/15 text-[#B66F58]'
                    : latest.short_interest?.direction === 'decrease'
                    ? 'border-[#829477]/40 bg-[#829477]/15 text-[#829477]'
                    : 'border-[#393C33] bg-[#2B2E27] text-[#AAA99D]'
                }`}>
                  {latest.short_interest?.direction === 'increase' && <TrendingUp className="h-3 w-3" />}
                  {latest.short_interest?.direction === 'decrease' && <TrendingDown className="h-3 w-3" />}
                  {latest.short_interest?.direction === 'no_significant_change' && <Minus className="h-3 w-3" />}
                  {latest.short_interest?.direction?.replace(/_/g, ' ') || 'Forecast'}
                </span>
                {latest.event?.ticker && (
                  <span className="text-[11px] font-mono text-[#A18B62] font-semibold">
                    ${latest.event?.ticker}
                  </span>
                )}
              </div>
              <p className="text-[11px] text-[#AAA99D] truncate font-sans" title={latest.event?.title}>
                {latest.event?.title}
              </p>
            </div>
          ) : (
            <div>
              <span className="text-xs text-[#85877C] italic">No catalysts processed yet</span>
              <p className="text-[11px] text-[#85877C] mt-1 font-mono">Submit event headline below</p>
            </div>
          )}
        </div>
        <div className="pt-2 border-t border-[#393C33] flex items-center justify-between text-[10px] font-mono text-[#85877C]">
          <span>CONFIDENCE</span>
          <span className="text-[#E8E4D9]">
            {latest?.short_interest?.confidence ? `${Math.round(latest.short_interest.confidence * 100)}%` : '---'}
          </span>
        </div>
      </div>

      {/* 3. Prediction Class Distribution */}
      <div className="bg-[#242720] border border-[#393C33] rounded p-4 flex flex-col justify-between">
        <div className="flex items-center justify-between text-[#85877C] pb-2 border-b border-[#393C33]">
          <span className="text-[11px] font-mono uppercase tracking-wider">
            03 / DISTRIBUTION (ACTUAL OUTPUTS)
          </span>
          <Layers className="h-3.5 w-3.5 text-[#A18B62]" />
        </div>
        <div className="py-2.5">
          {total > 0 ? (
            <div>
              {/* Segmented ledger bar */}
              <div className="h-2 w-full bg-[#1D201B] rounded overflow-hidden flex mb-2.5 border border-[#393C33]">
                <div 
                  style={{ width: `${pctIncrease}%` }} 
                  className="bg-[#B66F58] transition-all duration-300" 
                  title={`Increase: ${countByDirection.increase} (${pctIncrease}%)`} 
                />
                <div 
                  style={{ width: `${pctDecrease}%` }} 
                  className="bg-[#829477] transition-all duration-300" 
                  title={`Decrease: ${countByDirection.decrease} (${pctDecrease}%)`} 
                />
                <div 
                  style={{ width: `${pctNeutral}%` }} 
                  className="bg-[#85877C] transition-all duration-300" 
                  title={`Neutral: ${countByDirection.neutral} (${pctNeutral}%)`} 
                />
              </div>

              {/* Segmented numbers */}
              <div className="grid grid-cols-3 text-[10px] font-mono gap-1 text-[#AAA99D]">
                <div>
                  <span className="text-[#B66F58] font-bold block">INC</span>
                  <span>{countByDirection.increase} ({pctIncrease}%)</span>
                </div>
                <div>
                  <span className="text-[#829477] font-bold block">DEC</span>
                  <span>{countByDirection.decrease} ({pctDecrease}%)</span>
                </div>
                <div>
                  <span className="text-[#85877C] font-bold block">NEUT</span>
                  <span>{countByDirection.neutral} ({pctNeutral}%)</span>
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center py-2 text-xs text-[#85877C] italic">
              Empirical session ratios populate on execution
            </div>
          )}
        </div>
        <div className="pt-2 border-t border-[#393C33] flex items-center justify-between text-[10px] font-mono text-[#85877C]">
          <span>TARGET HORIZON</span>
          <span className="text-[#AAA99D]">FINRA SETTLEMENT</span>
        </div>
      </div>
    </div>
  );
}
