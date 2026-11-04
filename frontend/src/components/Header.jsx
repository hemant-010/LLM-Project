import React, { useState, useEffect, useRef } from 'react';
import { 
  Activity, 
  ShieldCheck, 
  Cpu, 
  Sparkles, 
  AlertCircle, 
  Bell, 
  Check, 
  Trash2, 
  ExternalLink,
  ShieldAlert,
  ArrowRight,
  X
} from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_BASE || '';

export default function Header({ backendHealth, isCheckingHealth, onSelectNotification }) {
  const isHealthy = backendHealth?.status === 'healthy';
  const models = backendHealth?.models || {};

  const [notifications, setNotifications] = useState([]);
  const [unreadCount, setUnreadCount] = useState(0);
  const [isOpen, setIsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const dropdownRef = useRef(null);

  // Fetch persistent notifications
  const fetchNotifications = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/notifications?limit=25`);
      if (res.ok) {
        const data = await res.json();
        setNotifications(data.notifications || []);
        setUnreadCount(data.unread_count || 0);
      }
    } catch (err) {
      console.error('Failed to fetch notifications:', err);
    }
  };

  useEffect(() => {
    fetchNotifications();
    const interval = setInterval(fetchNotifications, 15000);
    return () => clearInterval(interval);
  }, []);

  // Close dropdown on click outside
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Mark single or all as read
  const handleMarkAsRead = async (notifId = null) => {
    try {
      const res = await fetch(`${API_BASE}/api/notifications/mark-read`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ notification_id: notifId })
      });
      if (res.ok) {
        const data = await res.json();
        setUnreadCount(data.unread_count || 0);
        setNotifications((prev) =>
          prev.map((n) => (!notifId || n.id === notifId ? { ...n, is_read: true } : n))
        );
      }
    } catch (err) {
      console.error('Failed to mark read:', err);
    }
  };

  // Clear all notifications
  const handleClearAll = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/notifications`, {
        method: 'DELETE'
      });
      if (res.ok) {
        setNotifications([]);
        setUnreadCount(0);
      }
    } catch (err) {
      console.error('Failed to clear notifications:', err);
    }
  };

  const handleNotificationClick = (notif) => {
    if (!notif.is_read) {
      handleMarkAsRead(notif.id);
    }
    if (onSelectNotification) {
      onSelectNotification(notif);
      setIsOpen(false);
    }
  };

  return (
    <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3 flex flex-col md:flex-row items-center justify-between gap-3">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-indigo-600 via-blue-600 to-cyan-400 flex items-center justify-center shadow-lg shadow-blue-500/20">
            <Activity className="h-6 w-6 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-lg font-bold tracking-tight text-white">
                SHORT INTEREST INTELLIGENCE
              </h1>
              <span className="text-[10px] font-semibold uppercase px-2 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20">
                Phase 2
              </span>
            </div>
            <p className="text-xs text-slate-400">
              News-Driven FINRA Short-Interest Trend Prediction with Large Language Models
            </p>
          </div>
        </div>

        {/* Status Badges & Notification Center */}
        <div className="flex items-center flex-wrap gap-2 text-xs">
          {/* Backend Status */}
          <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full border ${
            isHealthy 
              ? 'bg-emerald-950/40 border-emerald-500/30 text-emerald-300' 
              : 'bg-rose-950/40 border-rose-500/30 text-rose-300'
          }`}>
            <span className={`h-2 w-2 rounded-full ${isHealthy ? 'bg-emerald-400 animate-pulse' : 'bg-rose-500'}`} />
            <span className="font-medium">
              Backend: {isHealthy ? 'Online (Port 8000)' : 'Connecting...'}
            </span>
          </div>

          {/* LR Baseline Badge */}
          <div className="flex items-center gap-1 px-2.5 py-1 rounded-md bg-slate-800/60 border border-slate-700/60 text-slate-300">
            <Cpu className="h-3.5 w-3.5 text-blue-400" />
            <span>LR: <strong className="text-white">{models.logistic_regression === 'ready' ? 'Ready (30k feat)' : 'Offline'}</strong></span>
          </div>

          {/* XGBoost Badge */}
          <div className="flex items-center gap-1 px-2.5 py-1 rounded-md bg-slate-800/60 border border-slate-700/60 text-slate-300">
            <ShieldCheck className="h-3.5 w-3.5 text-indigo-400" />
            <span>XGBoost: <strong className="text-white">{models.xgboost === 'ready' ? 'Ready' : 'Offline'}</strong></span>
          </div>

          {/* Gemini Badge */}
          <div className={`flex items-center gap-1 px-2.5 py-1 rounded-md border ${
            backendHealth?.environment?.gemini_configured
              ? 'bg-cyan-950/30 border-cyan-500/30 text-cyan-300'
              : 'bg-slate-800/40 border-slate-700/50 text-slate-400'
          }`}>
            <Sparkles className="h-3.5 w-3.5 text-cyan-400" />
            <span>Gemini: {backendHealth?.environment?.gemini_configured ? 'Active' : 'Heuristic Fallback'}</span>
          </div>

          {/* Persistent Investor Notification Bell */}
          <div className="relative ml-1" ref={dropdownRef}>
            <button
              onClick={() => setIsOpen(!isOpen)}
              className="relative p-2 rounded-lg bg-slate-800/80 hover:bg-slate-700/90 text-slate-300 hover:text-white border border-slate-700/80 transition-colors flex items-center justify-center"
              title="Investor Alerts & Notifications"
            >
              <Bell className="h-4 w-4" />
              {unreadCount > 0 && (
                <span className="absolute -top-1 -right-1 h-4 min-w-[16px] px-1 bg-red-500 text-white font-bold text-[9px] rounded-full flex items-center justify-center ring-2 ring-slate-900 animate-pulse">
                  {unreadCount > 9 ? '9+' : unreadCount}
                </span>
              )}
            </button>

            {/* Notification Popover Drawer */}
            {isOpen && (
              <div className="absolute right-0 mt-2 w-80 sm:w-96 bg-slate-900 border border-slate-700/90 rounded-xl shadow-2xl z-50 overflow-hidden animate-fadeIn">
                {/* Header */}
                <div className="p-3 bg-slate-950/90 border-b border-slate-800 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Bell className="h-4 w-4 text-amber-400" />
                    <span className="font-bold text-white text-xs">Investor Notifications</span>
                    {unreadCount > 0 && (
                      <span className="text-[10px] bg-red-500/20 text-red-300 px-1.5 py-0.2 rounded border border-red-500/30 font-semibold">
                        {unreadCount} unread
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-1">
                    {unreadCount > 0 && (
                      <button
                        onClick={() => handleMarkAsRead(null)}
                        className="text-[10px] text-blue-400 hover:text-blue-300 flex items-center gap-0.5 px-1.5 py-0.5 rounded hover:bg-slate-800"
                        title="Mark all as read"
                      >
                        <Check className="h-3 w-3" />
                        Mark all read
                      </button>
                    )}
                    <button
                      onClick={() => setIsOpen(false)}
                      className="text-slate-400 hover:text-white p-1 rounded hover:bg-slate-800"
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                  </div>
                </div>

                {/* Notifications List */}
                <div className="max-h-80 overflow-y-auto divide-y divide-slate-800/60">
                  {notifications.length === 0 ? (
                    <div className="py-8 text-center text-xs text-slate-500">
                      No investor notifications yet.
                    </div>
                  ) : (
                    notifications.map((notif) => {
                      const isSignificant = notif.severity === 'significant';
                      const isReview = notif.severity === 'requires_review';

                      return (
                        <div
                          key={notif.id}
                          onClick={() => handleNotificationClick(notif)}
                          className={`p-3 transition-colors cursor-pointer flex items-start gap-2.5 ${
                            !notif.is_read
                              ? 'bg-slate-800/40 hover:bg-slate-800/70'
                              : 'hover:bg-slate-950/60'
                          }`}
                        >
                          <div className="mt-0.5 shrink-0">
                            {isSignificant ? (
                              <ShieldAlert className="h-4 w-4 text-red-400" />
                            ) : isReview ? (
                              <AlertCircle className="h-4 w-4 text-amber-400" />
                            ) : (
                              <Activity className="h-4 w-4 text-blue-400" />
                            )}
                          </div>

                          <div className="flex-1 min-w-0">
                            <div className="flex items-center justify-between gap-1 mb-0.5">
                              <span className="font-semibold text-xs text-white truncate">
                                {notif.title}
                              </span>
                              {!notif.is_read && (
                                <span className="h-1.5 w-1.5 rounded-full bg-blue-400 shrink-0" />
                              )}
                            </div>
                            <p className="text-[11px] text-slate-400 line-clamp-2 leading-relaxed">
                              {notif.message}
                            </p>
                            <div className="flex items-center gap-2 mt-1.5 text-[10px] text-slate-500">
                              {notif.ticker && (
                                <span className="font-mono font-bold text-blue-400">
                                  ${notif.ticker}
                                </span>
                              )}
                              <span>&bull;</span>
                              <span>
                                {new Date(notif.created_at).toLocaleDateString(undefined, {
                                  month: 'short',
                                  day: 'numeric',
                                  hour: '2-digit',
                                  minute: '2-digit'
                                })}
                              </span>
                            </div>
                          </div>
                        </div>
                      );
                    })
                  )}
                </div>

                {/* Footer */}
                {notifications.length > 0 && (
                  <div className="p-2 bg-slate-950/80 border-t border-slate-800 flex items-center justify-between text-[10px] text-slate-500">
                    <span>Persistent SQLite Store</span>
                    <button
                      onClick={handleClearAll}
                      className="text-slate-400 hover:text-red-400 flex items-center gap-1 transition-colors"
                    >
                      <Trash2 className="h-3 w-3" />
                      Clear all
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}
