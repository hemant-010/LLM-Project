import React from 'react';
import { History, TrendingUp, TrendingDown, Minus, ArrowUpRight } from 'lucide-react';

export default function PredictionHistory({ history = [], onSelectPrediction, onClearHistory }) {
  if (!history || history.length === 0) return null;

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 shadow-sm">
      <div className="flex items-center justify-between pb-3 mb-3 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <History className="h-4 w-4 text-blue-400" />
          <h3 className="text-sm font-bold text-white uppercase tracking-wider">
            Session Query History ({history.length})
          </h3>
        </div>
        <button
          onClick={onClearHistory}
          className="text-[11px] text-slate-500 hover:text-slate-300 transition-colors"
        >
          Clear History
        </button>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs text-slate-300">
          <thead>
            <tr className="border-b border-slate-800 text-[10px] uppercase font-semibold text-slate-500">
              <th className="py-2.5 px-3">Ticker</th>
              <th className="py-2.5 px-3">Headline</th>
              <th className="py-2.5 px-3">Category</th>
              <th className="py-2.5 px-3">Model</th>
              <th className="py-2.5 px-3">Predicted Trend</th>
              <th className="py-2.5 px-3">Conf</th>
              <th className="py-2.5 px-3 text-right">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60 font-sans">
            {history.map((item, idx) => {
              const dir = item.short_interest?.direction;
              return (
                <tr key={idx} className="hover:bg-slate-800/40 transition-colors group">
                  <td className="py-2 px-3 font-mono font-bold text-white">
                    {item.event?.ticker || '—'}
                  </td>
                  <td className="py-2 px-3 max-w-xs truncate" title={item.event?.title}>
                    {item.event?.title}
                  </td>
                  <td className="py-2 px-3 capitalize font-mono text-[11px] text-slate-400">
                    {item.event?.event_type?.replace(/_/g, ' ') || 'unspecified'}
                  </td>
                  <td className="py-2 px-3 font-mono text-[11px] text-slate-400">
                    {item.model_family === 'xgboost' ? 'XGBoost' : 'LogisticReg'}
                  </td>
                  <td className="py-2 px-3">
                    <span className={`inline-flex items-center gap-1 text-[10px] font-bold uppercase px-2 py-0.5 rounded ${
                      dir === 'increase'
                        ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20'
                        : dir === 'decrease'
                        ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                        : 'bg-slate-800 text-slate-400 border border-slate-700'
                    }`}>
                      {dir === 'increase' && <TrendingUp className="h-3 w-3" />}
                      {dir === 'decrease' && <TrendingDown className="h-3 w-3" />}
                      {dir === 'no_significant_change' && <Minus className="h-3 w-3" />}
                      {dir?.replace(/_/g, ' ')}
                    </span>
                  </td>
                  <td className="py-2 px-3 font-mono text-slate-400">
                    {Math.round((item.short_interest?.confidence || 0) * 100)}%
                  </td>
                  <td className="py-2 px-3 text-right">
                    <button
                      onClick={() => onSelectPrediction(item)}
                      className="text-[11px] text-blue-400 hover:text-blue-300 font-medium inline-flex items-center gap-0.5 opacity-80 group-hover:opacity-100 transition-opacity"
                    >
                      View <ArrowUpRight className="h-3 w-3" />
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

