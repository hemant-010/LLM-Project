import React, { useState } from 'react';
import { 
  TrendingUp, 
  TrendingDown, 
  Minus, 
  CheckCircle2, 
  AlertTriangle, 
  Info, 
  Layers, 
  FileText, 
  Tag, 
  ShieldAlert, 
  ChevronDown, 
  ChevronUp 
} from 'lucide-react';

export default function PredictionResult({ result }) {
  const [showStockPrice, setShowStockPrice] = useState(false);

  if (!result) return null;

  const { event = {}, short_interest = {}, stock_price = {}, model_family, disclaimer } = result;
  const direction = short_interest.direction || 'no_significant_change';
  const probs = short_interest.probabilities || { increase: 0.33, decrease: 0.33, no_significant_change: 0.34 };

  const pctIncrease = Math.round((probs.increase || 0) * 100);
  const pctDecrease = Math.round((probs.decrease || 0) * 100);
  const pctNeutral = Math.round((probs.no_significant_change || 0) * 100);

  // Direction styling
  const dirConfig = {
    increase: {
      label: "INCREASE SHORT-INTEREST",
      subtitle: "Elevated Bearish Pressure / Short Borrow Activity",
      badgeClass: "bg-rose-500/15 border-rose-500/40 text-rose-300",
      icon: TrendingUp,
      colorClass: "text-rose-400",
      barColor: "bg-rose-500",
      description: "Short sellers are statistically projected to build borrow exposure in response to this negative corporate catalyst."
    },
    decrease: {
      label: "DECREASE SHORT-INTEREST",
      subtitle: "Short Covering / Borrow Exposure Reduction",
      badgeClass: "bg-emerald-500/15 border-emerald-500/40 text-emerald-300",
      icon: TrendingDown,
      colorClass: "text-emerald-400",
      barColor: "bg-emerald-500",
      description: "Short sellers are projected to cover existing borrows or reduce bearish positioning."
    },
    no_significant_change: {
      label: "NO SIGNIFICANT CHANGE",
      subtitle: "Neutral / Immaterial Short-Interest Movement",
      badgeClass: "bg-slate-700/30 border-slate-600/40 text-slate-300",
      icon: Minus,
      colorClass: "text-slate-400",
      barColor: "bg-slate-500",
      description: "Catalyst magnitude falls within standard bi-monthly FINRA settlement equilibrium."
    }
  }[direction] || {
    label: direction.replace(/_/g, ' ').toUpperCase(),
    subtitle: "Forecast Generated",
    badgeClass: "bg-blue-500/15 border-blue-500/40 text-blue-300",
    icon: Info,
    colorClass: "text-blue-400",
    barColor: "bg-blue-500",
    description: "Statistical forecast computed."
  };

  const DirIcon = dirConfig.icon;

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 shadow-xl space-y-5 animate-fade-in">
      {/* Header Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-800 gap-3">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xs uppercase font-semibold text-slate-400">Forecasting Engine Result</span>
            <span className="text-[10px] px-2 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20 font-mono">
              Model: {short_interest.model_type || model_family}
            </span>
          </div>
          <h3 className="text-lg font-extrabold text-white mt-1 flex items-center gap-2">
            {event.ticker && (
              <span className="px-2 py-0.5 bg-slate-800 text-blue-400 font-mono rounded text-sm border border-slate-700">
                {event.ticker}
              </span>
            )}
            <span className="truncate">{event.title}</span>
          </h3>
        </div>

        {/* Primary Direction Badge */}
        <div className={`flex items-center gap-2.5 px-4 py-2 rounded-xl border ${dirConfig.badgeClass} shrink-0`}>
          <DirIcon className={`h-6 w-6 ${dirConfig.colorClass}`} />
          <div>
            <div className="text-xs font-black tracking-wider uppercase">
              {dirConfig.label}
            </div>
            <div className="text-[10px] text-slate-400">
              Confidence: <strong>{Math.round((short_interest.confidence || 0) * 100)}%</strong>
            </div>
          </div>
        </div>
      </div>

      {/* Grid: Probability Distribution & Event Classification */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* Left Column: Multi-Class Probabilities */}
        <div className="bg-slate-950/70 border border-slate-800/80 rounded-xl p-4 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-bold uppercase text-slate-300 flex items-center gap-1.5">
                <Layers className="h-4 w-4 text-blue-400" />
                Multi-Class Probability Distribution
              </span>
              <span className="text-[11px] font-mono text-slate-400">
                Horizon: {short_interest.horizon}
              </span>
            </div>

            {/* Individual Class Probability Bars */}
            <div className="space-y-3.5">
              {/* Increase */}
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="font-medium text-rose-400 flex items-center gap-1.5">
                    <TrendingUp className="h-3 w-3" /> Increase
                  </span>
                  <span className="font-mono font-bold text-white">
                    {pctIncrease}% <span className="text-[10px] text-slate-500 font-normal">({probs.increase})</span>
                  </span>
                </div>
                <div className="h-2 w-full bg-slate-900 rounded-full overflow-hidden">
                  <div 
                    style={{ width: `${pctIncrease}%` }} 
                    className="h-full bg-rose-500 rounded-full transition-all duration-700" 
                  />
                </div>
              </div>

              {/* Decrease */}
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="font-medium text-emerald-400 flex items-center gap-1.5">
                    <TrendingDown className="h-3 w-3" /> Decrease
                  </span>
                  <span className="font-mono font-bold text-white">
                    {pctDecrease}% <span className="text-[10px] text-slate-500 font-normal">({probs.decrease})</span>
                  </span>
                </div>
                <div className="h-2 w-full bg-slate-900 rounded-full overflow-hidden">
                  <div 
                    style={{ width: `${pctDecrease}%` }} 
                    className="h-full bg-emerald-500 rounded-full transition-all duration-700" 
                  />
                </div>
              </div>

              {/* No Significant Change */}
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="font-medium text-slate-400 flex items-center gap-1.5">
                    <Minus className="h-3 w-3" /> No Significant Change
                  </span>
                  <span className="font-mono font-bold text-white">
                    {pctNeutral}% <span className="text-[10px] text-slate-500 font-normal">({probs.no_significant_change})</span>
                  </span>
                </div>
                <div className="h-2 w-full bg-slate-900 rounded-full overflow-hidden">
                  <div 
                    style={{ width: `${pctNeutral}%` }} 
                    className="h-full bg-slate-500 rounded-full transition-all duration-700" 
                  />
                </div>
              </div>
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-slate-800/80 text-[11px] text-slate-400">
            {dirConfig.description}
          </div>
        </div>

        {/* Right Column: Event Classification Metadata */}
        <div className="bg-slate-950/70 border border-slate-800/80 rounded-xl p-4 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-bold uppercase text-slate-300 flex items-center gap-1.5">
                <Tag className="h-4 w-4 text-cyan-400" />
                Event Taxonomy & Risk Context
              </span>
              <span className={`text-[10px] px-2 py-0.5 rounded font-semibold uppercase ${
                event.classification_source === 'gemini_llm'
                  ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/20'
                  : 'bg-slate-800 text-slate-400 border border-slate-700'
              }`}>
                Source: {event.classification_source || 'heuristic'}
              </span>
            </div>

            <div className="grid grid-cols-2 gap-3 mb-3">
              <div className="p-2.5 rounded-lg bg-slate-900/60 border border-slate-800">
                <span className="text-[10px] uppercase font-semibold text-slate-500 block">Event Category</span>
                <span className="text-sm font-bold text-slate-200 capitalize font-mono">
                  {event.event_type?.replace(/_/g, ' ') || 'Unspecified'}
                </span>
              </div>

              <div className="p-2.5 rounded-lg bg-slate-900/60 border border-slate-800">
                <span className="text-[10px] uppercase font-semibold text-slate-500 block">Assessed Severity</span>
                <span className={`text-sm font-bold capitalize font-mono ${
                  event.event_severity === 'high' || event.event_severity === 'critical'
                    ? 'text-rose-400'
                    : event.event_severity === 'medium'
                    ? 'text-amber-400'
                    : 'text-emerald-400'
                }`}>
                  {event.event_severity || 'Medium'}
                </span>
              </div>
            </div>

            {/* Explanation / Rationale */}
            {event.explanation && (
              <div className="text-xs text-slate-300 mb-2 p-2 rounded bg-slate-900/40 border border-slate-800/60">
                <strong className="text-slate-400">Analysis: </strong>
                {event.explanation}
              </div>
            )}

            {/* Evidence Quotes */}
            {event.evidence && event.evidence.length > 0 && (
              <div className="text-[11px] text-slate-400 mt-2">
                <strong className="text-slate-400">Verbatim Text Evidence: </strong>
                <div className="flex flex-wrap gap-1 mt-1">
                  {event.evidence.map((quote, idx) => (
                    <span key={idx} className="px-2 py-0.5 rounded bg-cyan-950/40 border border-cyan-500/30 text-cyan-300 italic">
                      "{quote}"
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Context Flags */}
            {event.context_flags && (
              <div className="text-[11px] text-slate-400 mt-2">
                <strong className="text-slate-400">Context Flags: </strong>
                <span className="text-slate-300">{event.context_flags}</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Stock Price Model (Legacy Fixture Limitation Section) */}
      <div className="border border-slate-800/80 rounded-xl overflow-hidden bg-slate-950/40">
        <button
          type="button"
          onClick={() => setShowStockPrice(!showStockPrice)}
          className="w-full px-4 py-2.5 flex items-center justify-between text-left text-xs font-semibold text-slate-400 hover:text-slate-200 transition-colors"
        >
          <span className="flex items-center gap-2">
            <AlertTriangle className="h-3.5 w-3.5 text-amber-400" />
            <span>Dual Horizon: 5-Day Stock Price Model</span>
            <span className="text-[10px] font-normal uppercase px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20">
              Experimental / Toy Fixture
            </span>
          </span>
          {showStockPrice ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
        </button>

        {showStockPrice && (
          <div className="px-4 pb-4 pt-1 border-t border-slate-800/80 text-xs text-slate-400 space-y-2">
            <div className="p-3 rounded-lg bg-amber-950/20 border border-amber-500/30 text-amber-300 text-[11px]">
              <strong>Limitation Notice: </strong>
              The stock-price model artifact is currently based on a legacy 16-sample toy fixture and has <strong>not</strong> been trained or validated on historical CRSP/TAQ pricing. It is retained strictly for architectural pipeline demonstration and must not be used for trading or financial analysis.
            </div>
            {stock_price.status === 'predicted' ? (
              <div className="flex items-center gap-4 pt-1 font-mono">
                <span>Direction: <strong className="text-white capitalize">{stock_price.direction?.replace(/_/g, ' ')}</strong></span>
                <span>Horizon: <strong className="text-white">{stock_price.horizon}</strong></span>
                <span>Confidence: <strong className="text-white">{Math.round((stock_price.confidence || 0) * 100)}%</strong></span>
              </div>
            ) : (
              <p className="text-slate-500 italic">Stock price model not active.</p>
            )}
          </div>
        )}
      </div>

      {/* Research Disclaimer */}
      <div className="text-[10px] text-slate-500 leading-relaxed border-t border-slate-800/60 pt-3 flex items-start gap-1.5">
        <Info className="h-3.5 w-3.5 text-slate-500 shrink-0 mt-0.5" />
        <span>{disclaimer}</span>
      </div>
    </div>
  );
}

