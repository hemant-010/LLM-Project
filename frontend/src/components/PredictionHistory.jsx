import React from 'react';
import { History, TrendingUp, TrendingDown, Minus, ArrowUpRight } from 'lucide-react';

export default function PredictionHistory({ history = [], onSelectPrediction, onClearHistory }) {
  if (!history || history.length === 0) return null;

  return (
    <div className="bg-[#242720] border border-[#393C33] rounded p-5 shadow-sm space-y-3">
      <div className="flex items-center justify-between pb-3 border-b border-[#393C33]">
        <div className="flex items-center gap-2">
          <History className="h-4 w-4 text-[#A18B62]" />
          <h3 className="text-xs font-mono font-semibold text-[#E8E4D9] uppercase tracking-wider">
            SESSION EVALUATION LEDGER ({history.length.toString().padStart(2, '0')})
          </h3>
        </div>
        <button
          onClick={onClearHistory}
          className="text-[11px] font-mono text-[#85877C] hover:text-[#B66F58] transition-colors"
        >
          CLEAR LEDGER
        </button>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs text-[#AAA99D]">
          <thead>
            <tr className="border-b border-[#393C33] text-[10px] font-mono uppercase text-[#85877C]">
              <th className="py-2 px-3">TICKER</th>
              <th className="py-2 px-3">HEADLINE</th>
              <th className="py-2 px-3">CATEGORY</th>
              <th className="py-2 px-3">MODEL</th>
              <th className="py-2 px-3">PREDICTED DIRECTION</th>
              <th className="py-2 px-3">CONF</th>
              <th className="py-2 px-3 text-right">ACTION</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[#393C33]/60 font-mono">
            {history.map((item, idx) => {
              const dir = item.short_interest?.direction;
              return (
                <tr key={idx} className="hover:bg-[#1D201B] transition-colors group">
                  <td className="py-2 px-3 font-bold text-[#E8E4D9]">
                    ${item.event?.ticker || '—'}
                  </td>
                  <td className="py-2 px-3 max-w-xs truncate font-sans text-[#E8E4D9]" title={item.event?.title}>
                    {item.event?.title}
                  </td>
                  <td className="py-2 px-3 capitalize text-[11px] text-[#85877C]">
                    {item.event?.event_type?.replace(/_/g, ' ') || 'unspecified'}
                  </td>
                  <td className="py-2 px-3 text-[11px] text-[#85877C]">
                    {item.model_family === 'xgboost' ? 'XGB' : 'LR-30k'}
                  </td>
                  <td className="py-2 px-3">
                    <span className={`inline-flex items-center gap-1 text-[10px] font-bold uppercase px-2 py-0.5 rounded border ${
                      dir === 'increase'
                        ? 'border-[#B66F58]/40 bg-[#B66F58]/15 text-[#B66F58]'
                        : dir === 'decrease'
                        ? 'border-[#829477]/40 bg-[#829477]/15 text-[#829477]'
                        : 'border-[#393C33] bg-[#2B2E27] text-[#85877C]'
                    }`}>
                      {dir === 'increase' && <TrendingUp className="h-3 w-3" />}
                      {dir === 'decrease' && <TrendingDown className="h-3 w-3" />}
                      {dir === 'no_significant_change' && <Minus className="h-3 w-3" />}
                      {dir?.replace(/_/g, ' ')}
                    </span>
                  </td>
                  <td className="py-2 px-3 text-[#AAA99D]">
                    {Math.round((item.short_interest?.confidence || 0) * 100)}%
                  </td>
                  <td className="py-2 px-3 text-right">
                    <button
                      onClick={() => onSelectPrediction(item)}
                      className="text-[11px] text-[#A18B62] hover:text-[#B29B70] font-medium inline-flex items-center gap-0.5 transition-colors"
                    >
                      INSPECT <ArrowUpRight className="h-3 w-3" />
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
