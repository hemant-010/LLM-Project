import React, { useState } from 'react';
import { 
  TrendingUp, 
  TrendingDown, 
  Minus, 
  AlertTriangle, 
  Info, 
  Layers, 
  Tag, 
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

  // Direction styling configuration (Strict Institutional Palette)
  const dirConfig = {
    increase: {
      label: "INCREASE SHORT-INTEREST",
      subtitle: "Elevated Bearish Pressure / Short Borrow Activity",
      badgeClass: "border-[#B66F58]/50 bg-[#B66F58]/15 text-[#B66F58]",
      icon: TrendingUp,
      colorClass: "text-[#B66F58]",
      barColor: "bg-[#B66F58]",
      description: "Short sellers are statistically projected to build borrow exposure in response to this negative corporate catalyst."
    },
    decrease: {
      label: "DECREASE SHORT-INTEREST",
      subtitle: "Short Covering / Borrow Exposure Reduction",
      badgeClass: "border-[#829477]/50 bg-[#829477]/15 text-[#829477]",
      icon: TrendingDown,
      colorClass: "text-[#829477]",
      barColor: "bg-[#829477]",
      description: "Short sellers are projected to cover existing borrows or reduce bearish positioning."
    },
    no_significant_change: {
      label: "NO SIGNIFICANT CHANGE",
      subtitle: "Neutral / Immaterial Short-Interest Movement",
      badgeClass: "border-[#393C33] bg-[#2B2E27] text-[#AAA99D]",
      icon: Minus,
      colorClass: "text-[#AAA99D]",
      barColor: "bg-[#85877C]",
      description: "Catalyst magnitude falls within standard bi-monthly FINRA settlement equilibrium."
    }
  }[direction] || {
    label: direction.replace(/_/g, ' ').toUpperCase(),
    subtitle: "Forecast Generated",
    badgeClass: "border-[#393C33] bg-[#2B2E27] text-[#E8E4D9]",
    icon: Info,
    colorClass: "text-[#A18B62]",
    barColor: "bg-[#A18B62]",
    description: "Statistical forecast computed."
  };

  const DirIcon = dirConfig.icon;

  return (
    <div className="bg-[#242720] border border-[#393C33] rounded p-6 shadow-sm space-y-6 animate-fade-in">
      {/* Header Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-[#393C33] gap-3">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-[11px] font-mono uppercase tracking-wider text-[#85877C]">
              FORECASTING ENGINE OUTPUT
            </span>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded border border-[#393C33] bg-[#1D201B] text-[#A18B62]">
              MODEL: {short_interest.model_type || model_family}
            </span>
          </div>
          <h3 className="text-lg font-semibold text-[#E8E4D9] mt-1 flex items-center gap-2 font-mono">
            {event.ticker && (
              <span className="px-2 py-0.5 bg-[#1D201B] text-[#A18B62] rounded text-sm border border-[#393C33]">
                ${event.ticker}
              </span>
            )}
            <span className="truncate">{event.title}</span>
          </h3>
        </div>

        {/* Primary Direction Badge */}
        <div className={`flex items-center gap-2.5 px-4 py-2 rounded border ${dirConfig.badgeClass} shrink-0`}>
          <DirIcon className={`h-5 w-5 ${dirConfig.colorClass}`} />
          <div>
            <div className="text-xs font-mono font-bold tracking-wider uppercase">
              {dirConfig.label}
            </div>
            <div className="text-[11px] font-mono text-[#AAA99D]">
              CONFIDENCE: <strong className="text-[#E8E4D9]">{Math.round((short_interest.confidence || 0) * 100)}%</strong>
            </div>
          </div>
        </div>
      </div>

      {/* Grid: Probability Distribution & Event Classification */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* Left Column: Multi-Class Probabilities */}
        <div className="bg-[#1D201B] border border-[#393C33] rounded p-4 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-4 pb-2 border-b border-[#393C33]">
              <span className="text-[11px] font-mono uppercase text-[#AAA99D] flex items-center gap-1.5">
                <Layers className="h-3.5 w-3.5 text-[#A18B62]" />
                Multi-Class Probability Distribution
              </span>
              <span className="text-[10px] font-mono text-[#85877C]">
                HORIZON: {short_interest.horizon}
              </span>
            </div>

            {/* Individual Class Probability Bars */}
            <div className="space-y-4">
              {/* Increase */}
              <div>
                <div className="flex justify-between text-xs font-mono mb-1.5">
                  <span className="text-[#B66F58] flex items-center gap-1.5">
                    <TrendingUp className="h-3 w-3" /> INCREASE (BEARISH)
                  </span>
                  <span className="text-[#E8E4D9]">
                    {pctIncrease}% <span className="text-[10px] text-[#85877C]">({probs.increase})</span>
                  </span>
                </div>
                <div className="h-1.5 w-full bg-[#242720] rounded overflow-hidden border border-[#393C33]">
                  <div 
                    style={{ width: `${pctIncrease}%` }} 
                    className="h-full bg-[#B66F58] transition-all duration-500" 
                  />
                </div>
              </div>

              {/* Decrease */}
              <div>
                <div className="flex justify-between text-xs font-mono mb-1.5">
                  <span className="text-[#829477] flex items-center gap-1.5">
                    <TrendingDown className="h-3 w-3" /> DECREASE (COVERING)
                  </span>
                  <span className="text-[#E8E4D9]">
                    {pctDecrease}% <span className="text-[10px] text-[#85877C]">({probs.decrease})</span>
                  </span>
                </div>
                <div className="h-1.5 w-full bg-[#242720] rounded overflow-hidden border border-[#393C33]">
                  <div 
                    style={{ width: `${pctDecrease}%` }} 
                    className="h-full bg-[#829477] transition-all duration-500" 
                  />
                </div>
              </div>

              {/* No Significant Change */}
              <div>
                <div className="flex justify-between text-xs font-mono mb-1.5">
                  <span className="text-[#85877C] flex items-center gap-1.5">
                    <Minus className="h-3 w-3" /> NO SIGNIFICANT CHANGE
                  </span>
                  <span className="text-[#E8E4D9]">
                    {pctNeutral}% <span className="text-[10px] text-[#85877C]">({probs.no_significant_change})</span>
                  </span>
                </div>
                <div className="h-1.5 w-full bg-[#242720] rounded overflow-hidden border border-[#393C33]">
                  <div 
                    style={{ width: `${pctNeutral}%` }} 
                    className="h-full bg-[#85877C] transition-all duration-500" 
                  />
                </div>
              </div>
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-[#393C33] text-[11px] text-[#85877C] leading-relaxed">
            {dirConfig.description}
          </div>
        </div>

        {/* Right Column: Event Classification Metadata */}
        <div className="bg-[#1D201B] border border-[#393C33] rounded p-4 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-4 pb-2 border-b border-[#393C33]">
              <span className="text-[11px] font-mono uppercase text-[#AAA99D] flex items-center gap-1.5">
                <Tag className="h-3.5 w-3.5 text-[#A18B62]" />
                Event Taxonomy & Risk Context
              </span>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded border border-[#393C33] bg-[#242720] text-[#A18B62]">
                SOURCE: {event.classification_source || 'HEURISTIC'}
              </span>
            </div>

            <div className="grid grid-cols-2 gap-3 mb-3">
              <div className="p-2.5 rounded bg-[#242720] border border-[#393C33]">
                <span className="text-[10px] font-mono uppercase text-[#85877C] block">EVENT CATEGORY</span>
                <span className="text-xs font-mono font-semibold text-[#E8E4D9] capitalize">
                  {event.event_type?.replace(/_/g, ' ') || 'Unspecified'}
                </span>
              </div>

              <div className="p-2.5 rounded bg-[#242720] border border-[#393C33]">
                <span className="text-[10px] font-mono uppercase text-[#85877C] block">ASSESSED SEVERITY</span>
                <span className={`text-xs font-mono font-semibold capitalize ${
                  event.event_severity === 'high' || event.event_severity === 'critical'
                    ? 'text-[#B66F58]'
                    : event.event_severity === 'medium'
                    ? 'text-[#C49A58]'
                    : 'text-[#829477]'
                }`}>
                  {event.event_severity || 'Medium'}
                </span>
              </div>
            </div>

            {/* Explanation / Rationale */}
            {event.explanation && (
              <div className="text-xs text-[#AAA99D] mb-2 p-2.5 rounded bg-[#242720] border border-[#393C33] leading-relaxed">
                <strong className="text-[#E8E4D9] font-mono">ANALYSIS: </strong>
                {event.explanation}
              </div>
            )}

            {/* Evidence Quotes */}
            {event.evidence && event.evidence.length > 0 && (
              <div className="text-[11px] text-[#85877C] mt-2">
                <strong className="text-[#AAA99D] font-mono block mb-1">VERBATIM TEXT EVIDENCE:</strong>
                <div className="flex flex-wrap gap-1">
                  {event.evidence.map((quote, idx) => (
                    <span key={idx} className="px-2 py-0.5 rounded bg-[#242720] border border-[#393C33] text-[#A18B62] font-mono text-[10px]">
                      "{quote}"
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Context Flags */}
            {event.context_flags && (
              <div className="text-[11px] text-[#85877C] mt-2 font-mono">
                <strong className="text-[#AAA99D]">FLAGS: </strong>
                <span className="text-[#E8E4D9]">{event.context_flags}</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Stock Price Model (Legacy Fixture Limitation Section) */}
      <div className="border border-[#393C33] rounded overflow-hidden bg-[#1D201B]">
        <button
          type="button"
          onClick={() => setShowStockPrice(!showStockPrice)}
          className="w-full px-4 py-2.5 flex items-center justify-between text-left text-xs font-mono text-[#85877C] hover:text-[#E8E4D9] transition-colors"
        >
          <span className="flex items-center gap-2">
            <AlertTriangle className="h-3.5 w-3.5 text-[#C49A58]" />
            <span>Dual Horizon: 5-Day Stock Price Model</span>
            <span className="text-[10px] uppercase px-1.5 py-0.2 rounded border border-[#C49A58]/30 bg-[#C49A58]/10 text-[#C49A58]">
              Experimental / Toy Fixture
            </span>
          </span>
          {showStockPrice ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
        </button>

        {showStockPrice && (
          <div className="px-4 pb-4 pt-2 border-t border-[#393C33] text-xs text-[#AAA99D] space-y-2">
            <div className="p-3 rounded bg-[#242720] border border-[#C49A58]/30 text-[#C49A58] text-[11px] leading-relaxed">
              <strong>Limitation Notice: </strong>
              The stock-price model artifact is currently based on a legacy 16-sample toy fixture and has <strong>not</strong> been trained or validated on historical CRSP/TAQ pricing. It is retained strictly for architectural pipeline demonstration and must not be used for trading or financial analysis.
            </div>
            {stock_price.status === 'predicted' ? (
              <div className="flex items-center gap-4 pt-1 font-mono text-[11px]">
                <span>DIRECTION: <strong className="text-[#E8E4D9] capitalize">{stock_price.direction?.replace(/_/g, ' ')}</strong></span>
                <span>HORIZON: <strong className="text-[#E8E4D9]">{stock_price.horizon}</strong></span>
                <span>CONFIDENCE: <strong className="text-[#E8E4D9]">{Math.round((stock_price.confidence || 0) * 100)}%</strong></span>
              </div>
            ) : (
              <p className="text-[#85877C] italic">Stock price model not active.</p>
            )}
          </div>
        )}
      </div>

      {/* Research Disclaimer */}
      <div className="text-[10px] text-[#85877C] leading-relaxed border-t border-[#393C33] pt-3 flex items-start gap-1.5 font-mono">
        <Info className="h-3.5 w-3.5 text-[#85877C] shrink-0 mt-0.5" />
        <span>{disclaimer}</span>
      </div>
    </div>
  );
}
