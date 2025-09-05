import React, { useState, useRef, useEffect, useMemo } from 'react';
import { Play, Pause, RotateCcw, ChevronLeft, ChevronRight } from 'lucide-react';

const API_BASE = 'http://127.0.0.1:5001';

// --- helpers ---
const scaleX = (x, pitchWidth) => (x / 105) * pitchWidth;
const scaleY = (y, pitchHeight) => (y / 68) * pitchHeight;

// step adapter is already done server-side, but keep a guard
const coerceSteps = (payload) => {
  const steps = Array.isArray(payload?.steps) ? payload.steps : [];
  const totalScore = Number(payload?.totalScore ?? 0);
  return { steps, totalScore };
};

// --- FootballPitch (unchanged visuals, now takes data prop) ---
const FootballPitch = ({ scenario, currentStep, showTrails = true, data }) => {
  const pitchWidth = 480;
  const pitchHeight = 320;

  const getStepColor = (step, isActive) => {
    if (scenario === 'baseline') {
      return isActive ? '#00ff88' : '#00cc66';
    } else {
      return isActive ? '#ff4757' : '#ff3742';
    }
  };

  const getActionSymbol = (actionType) => {
    switch (actionType) {
      case 'pass': return '→';
      case 'shot': return '⚽';
      case 'carry': return '↗';
      case 'takeon': return '⚡';
      case 'clear': return '↶';
      default: return '•';
    }
  };

  return (
    <div className="relative overflow-hidden rounded-xl bg-gradient-to-br from-green-900 via-green-800 to-green-900">
      <div className={`absolute inset-0 rounded-xl ${scenario === 'baseline' ? 'shadow-lg shadow-green-500/30' : 'shadow-lg shadow-red-500/30'}`}></div>
      <svg width={pitchWidth} height={pitchHeight} className="relative z-10">
        <defs>
          <radialGradient id="pitchGradient" cx="50%" cy="50%" r="70%">
            <stop offset="0%" stopColor="#1a5f3f"/>
            <stop offset="100%" stopColor="#0d3d2a"/>
          </radialGradient>
          <filter id="glow"><feGaussianBlur stdDeviation="3" result="coloredBlur"/><feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
          <filter id="actionGlow"><feGaussianBlur stdDeviation="4" result="coloredBlur"/><feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
        </defs>

        <rect width={pitchWidth} height={pitchHeight} fill="url(#pitchGradient)"/>
        <pattern id="pitchLines" patternUnits="userSpaceOnUse" width="40" height="40">
          <rect width="40" height="40" fill="none"/>
          <path d="M0 20h40M20 0v40" stroke="rgba(255,255,255,0.05)" strokeWidth="1"/>
        </pattern>
        <rect width={pitchWidth} height={pitchHeight} fill="url(#pitchLines)"/>

        <g filter="url(#glow)">
          <line x1={pitchWidth/2} y1="0" x2={pitchWidth/2} y2={pitchHeight} stroke="#00ff88" strokeWidth="3" opacity="0.8"/>
          <circle cx={pitchWidth/2} cy={pitchHeight/2} r="45" fill="none" stroke="#00ff88" strokeWidth="3" opacity="0.8"/>
          <circle cx={pitchWidth/2} cy={pitchHeight/2} r="3" fill="#00ff88" opacity="0.8"/>
          <rect x="0" y="80" width="75" height="160" fill="none" stroke="#00ff88" strokeWidth="3" opacity="0.8"/>
          <rect x={pitchWidth-75} y="80" width="75" height="160" fill="none" stroke="#00ff88" strokeWidth="3" opacity="0.8"/>
          <rect x="0" y="128" width="25" height="64" fill="none" stroke="#00ff88" strokeWidth="2" opacity="0.6"/>
          <rect x={pitchWidth-25} y="128" width="25" height="64" fill="none" stroke="#00ff88" strokeWidth="2" opacity="0.6"/>
          <rect x="-5" y="140" width="12" height="40" fill="#ffffff" stroke="#00ff88" strokeWidth="2"/>
          <rect x={pitchWidth-7} y="140" width="12" height="40" fill="#ffffff" stroke="#00ff88" strokeWidth="2"/>
        </g>

        {showTrails && data.steps.length > 1 && (
          <g>
            <defs>
              <linearGradient id={`trail-${scenario}`} x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor={scenario === 'baseline' ? '#00ff88' : '#ff4757'} stopOpacity="0"/>
                <stop offset="50%" stopColor={scenario === 'baseline' ? '#00ff88' : '#ff4757'} stopOpacity="0.6"/>
                <stop offset="100%" stopColor={scenario === 'baseline' ? '#00ff88' : '#ff4757'} stopOpacity="1"/>
              </linearGradient>
            </defs>
            <polyline
              points={data.steps.slice(0, currentStep + 1).map(step =>
                `${scaleX(step.target.x, pitchWidth)},${scaleY(step.target.y, pitchHeight)}`
              ).join(' ')}
              fill="none"
              stroke={`url(#trail-${scenario})`}
              strokeWidth="4"
              strokeDasharray="8,4"
              filter="url(#glow)"
              className="animate-pulse"
            />
          </g>
        )}

        {data.steps.slice(0, currentStep + 1).map((step, index) => {
          const x = scaleX(step.target.x, pitchWidth);
          const y = scaleY(step.target.y, pitchHeight);
          const isActive = index === currentStep;
          const stepColor = (scenario === 'baseline')
            ? (isActive ? '#00ff88' : '#00cc66')
            : (isActive ? '#ff4757' : '#ff3742');

          const symbol = (() => {
            switch (step.action.type) {
              case 'pass': return '→';
              case 'shot': return '⚽';
              case 'carry': return '↗';
              case 'takeon': return '⚡';
              case 'clear': return '↶';
              default: return '•';
            }
          })();

          return (
            <g key={index} filter={isActive ? "url(#actionGlow)" : undefined}>
              {isActive && (
                <circle cx={x} cy={y} r="18" fill="none" stroke={stepColor} strokeWidth="2" opacity="0.5" className="animate-ping" />
              )}
              <circle cx={x} cy={y} r={isActive ? "12" : "8"} fill={stepColor} stroke="#ffffff" strokeWidth="3" className={isActive ? "animate-pulse" : ""} />
              <text x={x} y={y + 2} textAnchor="middle" className="text-sm font-bold fill-white" style={{ textShadow: '2px 2px 4px rgba(0,0,0,0.8)' }}>
                {symbol}
              </text>
              {isActive && (
                <g>
                  <rect x={x - 25} y={y - 35} width="50" height="16" fill="rgba(0,0,0,0.8)" stroke={stepColor} strokeWidth="1" rx="8" />
                  <text x={x} y={y - 25} textAnchor="middle" className="text-xs font-bold fill-white">
                    {step.actor.name}
                  </text>
                </g>
              )}
            </g>
          );
        })}

        <g opacity="0.3">
          <circle cx="20" cy="20" r="3" fill="#00ff88"/>
          <circle cx={pitchWidth-20} cy="20" r="3" fill="#00ff88"/>
          <circle cx="20" cy={pitchHeight-20} r="3" fill="#00ff88"/>
          <circle cx={pitchWidth-20} cy={pitchHeight-20} r="3" fill="#00ff88"/>
        </g>
      </svg>

      <div className={`absolute top-4 left-4 px-3 py-1 rounded-full text-xs font-bold text-white ${
        scenario === 'baseline' ? 'bg-green-600/90' : 'bg-red-600/90'
      }`}>
        {scenario === 'baseline' ? 'BASELINE' : 'COUNTERFACTUAL'}
      </div>
    </div>
  );
};

// --- main component ---
const LIMFootballVisualizer = () => {
  // pick an existing match/event (adjust these!)
  const [matchBase, setMatchBase] = useState('20241016_FU17WWC_01_DOM-ECU_FIFA_Unified_Event_Data_Events');
  const [eventId, setEventId] = useState('1108');

  const [baseline, setBaseline] = useState({ steps: [], totalScore: 0 });
  const [ctrf, setCtrf] = useState({ steps: [], totalScore: 0 });
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1000);
  const [showTrails, setShowTrails] = useState(true);
  const intervalRef = useRef();

  // Fetch baseline + a simple counterfactual
  useEffect(() => {
    let cancelled = false;

    async function load() {
      // baseline
      const r1 = await fetch(`${API_BASE}/sim/${encodeURIComponent(matchBase)}/${encodeURIComponent(eventId)}`);
      const j1 = await r1.json();
      if (!cancelled) setBaseline(coerceSteps(j1));

      // counterfactual: e.g., at k=3 force a shot to the goal center
      const r2 = await fetch(`${API_BASE}/sim/${encodeURIComponent(matchBase)}/${encodeURIComponent(eventId)}`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          force_action: {"3":"shot"},
          force_target: {"3":[105, 34]},
          temp_override: {"1": 0.7},   // a bit more deterministic at k=1
          horizon_K: 5,
          rollouts_R: 200
        })
      });
      const j2 = await r2.json();
      if (!cancelled) setCtrf(coerceSteps(j2));

      // reset playback
      setCurrentStep(0);
      setIsPlaying(false);
    }
    load();

    return () => { cancelled = true; };
  }, [matchBase, eventId]);

  const maxSteps = useMemo(() => Math.max(baseline.steps.length, ctrf.steps.length, 1), [baseline, ctrf]);

  useEffect(() => {
    if (isPlaying) {
      intervalRef.current = setInterval(() => {
        setCurrentStep(prev => {
          if (prev >= maxSteps - 1) { setIsPlaying(false); return prev; }
          return prev + 1;
        });
      }, playbackSpeed);
    } else {
      clearInterval(intervalRef.current);
    }
    return () => clearInterval(intervalRef.current);
  }, [isPlaying, playbackSpeed, maxSteps]);

  const handlePlayPause = () => setIsPlaying(!isPlaying);
  const handleReset = () => { setIsPlaying(false); setCurrentStep(0); };
  const handleStepForward = () => currentStep < maxSteps - 1 && setCurrentStep(currentStep + 1);
  const handleStepBackward = () => currentStep > 0 && setCurrentStep(currentStep - 1);

  return (
    <div className="min-h-screen bg-black text-white" style={{ fontFamily: '"EA Font", "Arial Black", "Impact", sans-serif' }}>
      <div className="max-w-6xl mx-auto p-8">
        {/* Header */}
        <header className="text-center mb-8">
          <h1 className="text-5xl font-black tracking-tight text-white mb-2">LIM ANALYTICS</h1>
          <p className="text-gray-400 text-sm uppercase tracking-widest font-semibold">LIVING INFLUENCE MODEL SIMULATION</p>
        </header>

        {/* Inputs (pick match/event) */}
        <div className="bg-gray-900 rounded-lg border border-gray-800 p-4 mb-6 flex flex-col md:flex-row gap-3">
          <input
            className="flex-1 px-3 py-2 bg-gray-800 border border-gray-700 rounded text-white text-sm"
            placeholder="Match base (folder/file stem)"
            value={matchBase}
            onChange={(e)=>setMatchBase(e.target.value)}
          />
          <input
            className="w-40 px-3 py-2 bg-gray-800 border border-gray-700 rounded text-white text-sm"
            placeholder="Event ID"
            value={eventId}
            onChange={(e)=>setEventId(e.target.value)}
          />
          <button onClick={()=>{ /* triggers useEffect by updating state itself */ setMatchBase(m=>m); }} className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm font-bold">LOAD</button>
        </div>

        {/* Controls */}
        <div className="bg-gray-900 rounded-lg border border-gray-800 p-6 mb-8">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <button onClick={handlePlayPause}
                className={`flex items-center gap-2 px-4 py-2 rounded font-bold text-sm transition-colors ${isPlaying ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'}`}>
                {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                {isPlaying ? 'PAUSE' : 'PLAY'}
              </button>
              <button onClick={handleReset} className="flex items-center gap-2 px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded font-bold text-sm">
                <RotateCcw className="w-4 h-4" />
                RESET
              </button>
              <div className="flex items-center gap-2">
                <button onClick={handleStepBackward} disabled={currentStep === 0}
                  className="w-8 h-8 rounded bg-gray-700 hover:bg-gray-600 disabled:opacity-50 flex items-center justify-center">
                  <ChevronLeft className="w-4 h-4" />
                </button>
                <div className="px-3 py-1 bg-gray-800 rounded font-mono text-sm text-gray-300">
                  {currentStep + 1}/{maxSteps}
                </div>
                <button onClick={handleStepForward} disabled={currentStep === maxSteps - 1}
                  className="w-8 h-8 rounded bg-gray-700 hover:bg-gray-600 disabled:opacity-50 flex items-center justify-center">
                  <ChevronRight className="w-4 h-4" />
                </button>
              </div>
            </div>

            <div className="flex items-center gap-4">
              <label className="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" checked={showTrails} onChange={(e)=>setShowTrails(e.target.checked)} className="w-4 h-4 accent-gray-600" />
                <span className="text-gray-300 text-sm font-semibold">TRAILS</span>
              </label>
              <select value={playbackSpeed} onChange={(e)=>setPlaybackSpeed(Number(e.target.value))}
                className="px-3 py-1 bg-gray-800 border border-gray-700 rounded text-white text-sm font-semibold">
                <option value={2000}>0.5x</option>
                <option value={1000}>1x</option>
                <option value={500}>2x</option>
                <option value={250}>4x</option>
              </select>
            </div>
          </div>

          <div className="w-full bg-gray-800 rounded-full h-2">
            <div className="bg-gray-600 h-2 rounded-full transition-all duration-300" style={{ width: `${((currentStep + 1) / maxSteps) * 100}%` }}/>
          </div>
        </div>

        {/* Pitches */}
        <div className="grid lg:grid-cols-2 gap-8">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-2xl font-black tracking-tight uppercase">BASELINE SCENARIO</h2>
              <div className="text-right">
                <div className="text-xl font-black text-gray-300">{baseline.totalScore.toFixed(3)}</div>
                <div className="text-xs text-gray-500 uppercase tracking-wider font-semibold">SEQUENCE SCORE</div>
              </div>
            </div>
            <FootballPitch scenario="baseline" currentStep={currentStep} showTrails={showTrails} data={baseline}/>
          </div>

          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-2xl font-black tracking-tight uppercase">COUNTERFACTUAL</h2>
              <div className="text-right">
                <div className="text-xl font-black text-gray-300">{ctrf.totalScore.toFixed(3)}</div>
                <div className="text-xs text-gray-500 uppercase tracking-wider font-semibold">SEQUENCE SCORE</div>
              </div>
            </div>
            <FootballPitch scenario="counterfactual" currentStep={currentStep} showTrails={showTrails} data={ctrf}/>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LIMFootballVisualizer;
