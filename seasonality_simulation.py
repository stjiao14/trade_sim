import numpy as np

S, T, D, L = 8, 13, 252, 60
SIG_SEASON = 0.0002      # 真·同槽 edge ~2bps
SIG_NOISE  = 0.0040      # 半小时噪声 ~40bps
COST_RT    = 0.0003      # 往返成本 3bps
reps = 400

def run(drift_sd, drift_persist=True):
    rng = np.random.default_rng(7)
    g,n,win,instab,maxshare,dcorr,sharpe = ([] for _ in range(7))
    for _ in range(reps):
        mu    = rng.normal(0, SIG_SEASON, (S,T))
        drift = rng.normal(0, drift_sd, (S,))
        if drift_persist:
            drift_t = np.repeat(drift[:,None], D+L, axis=1)        # 趋势永久持续
        else:
            # 趋势会衰减/换向:回看期的drift与未来无关
            drift_t = rng.normal(0, drift_sd, (S, D+L))            # 每天重抽
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

print(f"估计标准误 SIG_NOISE/sqrt(L) = {SIG_NOISE/np.sqrt(L)*1e4:.1f} bps  vs  真信号 {SIG_SEASON*1e4:.0f} bps")
print(f"=> 噪声/信号 = {(SIG_NOISE/np.sqrt(L))/SIG_SEASON:.1f}x\n")

tr=13*252
for name,kw in [("A 纯seasonality(声称的机制,无趋势)", dict(drift_sd=0.0)),
                ("B +持续趋势(永久drift)",            dict(drift_sd=0.0006,drift_persist=True)),
                ("C +趋势但会衰减/换向(更现实)",       dict(drift_sd=0.0006,drift_persist=False))]:
    r=run(**kw)
    print(f"【情形 {name}】")
    print(f"  毛收益/笔 {r['gross']:+.2f}bps | 净收益/笔 {r['net']:+.2f}bps | 胜率 {r['win']:.1f}%")
    print(f"  per-trade Sharpe {r['sharpe']:+.3f} | argmax翻面率 {r['instab']:.0f}%"
          f" | 单票最多占 {r['maxshare']:.0f}%槽 | drift-中选相关 {r['dcorr']:+.2f}")
    print(f"  年化(~{tr}笔/年): 毛 {r['gross']*tr/100:+.0f}% / 净 {r['net']*tr/100:+.0f}%\n")
