import React, { useState, useEffect } from 'react';
import Header from './components/Header.jsx';
import StatsOverview from './components/StatsOverview.jsx';
import PredictionForm from './components/PredictionForm.jsx';
import PredictionResult from './components/PredictionResult.jsx';
import InsiderSellingCard from './components/InsiderSellingCard.jsx';
import PredictionHistory from './components/PredictionHistory.jsx';

const API_BASE = import.meta.env.VITE_API_BASE || '';

export default function App() {
  const [backendHealth, setBackendHealth] = useState(null);
  const [isCheckingHealth, setIsCheckingHealth] = useState(true);
  const [currentResult, setCurrentResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState(() => {
    try {
      const saved = localStorage.getItem('short_interest_session_history');
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });

  // Check backend health
  const checkHealth = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/health`);
      if (res.ok) {
        const data = await res.json();
        setBackendHealth(data);
      } else {
        setBackendHealth({ status: 'unhealthy' });
      }
    } catch (err) {
      setBackendHealth({ status: 'offline' });
    } finally {
      setIsCheckingHealth(false);
    }
  };

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  // Save history to localStorage
  useEffect(() => {
    try {
      localStorage.setItem('short_interest_session_history', JSON.stringify(history));
    } catch (e) {
      console.error("Failed to save history to localStorage", e);
    }
  }, [history]);

  // Handle Prediction Submission
  const handlePredict = async (payload) => {
    setIsLoading(true);
    setError(null);

    try {
      const res = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      const data = await res.json();

      if (!res.ok) {
        const errorMsg = data?.message || data?.errors?.join(', ') || 'Failed to generate prediction';
        throw new Error(errorMsg);
      }

      setCurrentResult(data);
      setHistory((prev) => [data, ...prev.slice(0, 49)]); // keep latest 50

      // Scroll to result smoothly
      setTimeout(() => {
        const resultElement = document.getElementById('prediction-result-section');
        if (resultElement) {
          resultElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      }, 100);

    } catch (err) {
      console.error("Prediction API failure:", err);
      setError(err.message || 'Network error connecting to prediction API');
    } finally {
      setIsLoading(false);
    }
  };

  const handleClearHistory = () => {
    setHistory([]);
    try {
      localStorage.removeItem('short_interest_session_history');
    } catch {}
  };

  const handleSelectNotification = (notif) => {
    const title = notif.title || `Alert: ${notif.ticker || 'Corporate Event'}`;
    const desc = notif.message || '';
    handlePredict({
      title,
      ticker: notif.ticker || undefined,
      description: desc,
      model_family: 'logistic_regression',
      use_llm: false
    });
  };

  return (
    <div className="min-h-screen bg-[#171916] text-[#E8E4D9] flex flex-col font-sans">
      {/* Top Navigation */}
      <Header 
        backendHealth={backendHealth} 
        isCheckingHealth={isCheckingHealth} 
        onSelectNotification={handleSelectNotification}
      />

      {/* Main Content Dashboard */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
        {/* KPI & Distribution Stats Overview */}
        <StatsOverview history={history} />

        {/* Prediction Input Form */}
        <PredictionForm
          onSubmit={handlePredict}
          isLoading={isLoading}
          error={error}
        />

        {/* Prediction Results Display */}
        {currentResult && (
          <div id="prediction-result-section">
            <PredictionResult result={currentResult} />
          </div>
        )}

        {/* Session Query History */}
        <PredictionHistory
          history={history}
          onSelectPrediction={(item) => {
            setCurrentResult(item);
            const el = document.getElementById('prediction-result-section');
            if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
          }}
          onClearHistory={handleClearHistory}
        />

        {/* SEC Form 4 Insider Intelligence Module */}
        <InsiderSellingCard onEvaluateTransaction={handlePredict} />
      </main>

      {/* Footer */}
      <footer className="border-t border-[#393C33] bg-[#171916] py-5 text-center text-xs text-[#85877C]">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span className="font-mono">Short Interest Trend Prediction &bull; Phase 2 Academic Research Platform</span>
          <span className="font-mono text-[11px] text-[#85877C]">Models: LogisticRegression (30k TF-IDF) &bull; XGBoost (Trees)</span>
        </div>
      </footer>
    </div>
  );
}

