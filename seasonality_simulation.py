import numpy as np

S, T, D, L = 8, 13, 252, 60
SIG_SEASON = 0.0002      # True same-slot edge ~2bps
SIG_NOISE  = 0.0040      # Half-hour noise ~40bps
COST_RT    = 0.0003      # Round-trip cost 3bps
reps = 400

def run(drift_sd, drift_persist=True):
    rng = np.random.default_rng(7)
    g,n,win,instab,maxshare,dcorr,sharpe = ([] for _ in range(7))
    for _ in range(reps):
        mu    = rng.normal(0, SIG_SEASON, (S,T))
        drift = rng.normal(0, drift_sd, (S,))
        if drift_persist:
            drift_t = np.repeat(drift[:,None], D+L, axis=1)        # Trend persists forever.
        else:
            # Trend decays/flips: lookback drift does not predict the future.
            drift_t = rng.normal(0, drift_sd, (S, D+L))            # Redraw each day.
        base = mu[:,None,:] + drift_t[:,:,None]
        R = base + rng.normal(0, SIG_NOISE, (S,D+L,T))
        picks=np.zeros((D,T),int); gross=np.zeros((D,T))
        for di,d in enumerate(range(L,D+L)):
            score=R[:,d-L:d,:].mean(axis=1); pk=score.argmax(axis=0)
            picks[di]=pk; gross[di]=R[pk,d,np.arange(T)]
        net=gross-COST_RT
        g.append(gross.mean()*1e4); n.append(net.mean()*1e4)
        win.append((net>0).mean()); instab.append((picks[1:]!=picks[:-1]).mean())
        sharpe.append(net.mean()/net.std())
        cnt=np.bincount(picks.ravel(),minlength=S)
        maxshare.append(cnt.max()/cnt.sum()); dcorr.append(np.corrcoef(drift,cnt)[0,1])
    M=lambda x:float(np.mean(x))
    return dict(gross=M(g),net=M(n),win=M(win)*100,instab=M(instab)*100,
                maxshare=M(maxshare)*100,dcorr=M(dcorr),sharpe=M(sharpe))

print(f"Estimated SE SIG_NOISE/sqrt(L) = {SIG_NOISE/np.sqrt(L)*1e4:.1f} bps  vs  true signal {SIG_SEASON*1e4:.0f} bps")
print(f"=> noise/signal = {(SIG_NOISE/np.sqrt(L))/SIG_SEASON:.1f}x\n")

tr=13*252
for name,kw in [("A pure seasonality (claimed mechanism, no trend)", dict(drift_sd=0.0)),
                ("B + persistent trend (permanent drift)",          dict(drift_sd=0.0006,drift_persist=True)),
                ("C + trend that decays/flips (more realistic)",    dict(drift_sd=0.0006,drift_persist=False))]:
    r=run(**kw)
    print(f"[Scenario {name}]")
    print(f"  gross/trade {r['gross']:+.2f}bps | net/trade {r['net']:+.2f}bps | win rate {r['win']:.1f}%")
    print(f"  per-trade Sharpe {r['sharpe']:+.3f} | argmax turnover {r['instab']:.0f}%"
          f" | max single-name slot share {r['maxshare']:.0f}% | drift-selection corr {r['dcorr']:+.2f}")
    print(f"  annualized (~{tr} trades/year): gross {r['gross']*tr/100:+.0f}% / net {r['net']*tr/100:+.0f}%\n")
