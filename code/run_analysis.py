"""Tolerance budget for the fast fiber-reinforced bending actuator model.

Run:  python run_analysis.py [--n 20000] [--seed 249] [--out results]
Outputs (CSV + PNG) go to the output folder; key numbers are printed.

Definitions used throughout
  gap          = P_exp - P_model          (systematic bias of a model stage, kPa)
  P scatter    = 1-sigma std of P(360 deg) from geometric tolerance (random, kPa)
  equiv. shift = Neo-Hookean modulus change (MPa) giving the same pressure change
                 at the same angle, at the nominal geometry: dmu = dP / (dP/dmu)_NH
  crossover    = sigma_t* at which P scatter == |gap|
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator
import fiber_model as fm

P_EXP = 207.0                       # kPa, measured baseline read from Fig. 9 of the paper
P_EXP_ALT = 220.0                   # kPa, value labelled in Fig. 9 for the analytical curve
ANGLES = (180, 270, 360)
TH360 = 2 * np.pi

# Model stages (the "ladder"). Sheath = added 1 mm Ecoflex outer layer on the top wall.
# The middle sheath modulus is solved so that the Quad-Chart 'NH conversion' point (211.7 kPa)
# is reproduced (~39 kPa); its physical origin is NOT verified here.
def _mu_sheath_mid(target=211.7):
    from scipy.optimize import brentq
    return brentq(lambda m: fm.pressure(TH360, wall=fm.yeoh(), sheath=fm.neo_hookean(m))[0] - target, 0.025, 0.092)


def stages():
    nh = fm.neo_hookean
    mid = _mu_sheath_mid()
    return {
        "M0 NH 0.22":            dict(wall=nh(0.22)),
        "M1 Yeoh":               dict(wall=fm.yeoh()),
        "M2a Yeoh+sheath 25kPa": dict(wall=fm.yeoh(), sheath=nh(0.025)),
        "M2b Yeoh+sheath 39kPa":  dict(wall=fm.yeoh(), sheath=nh(mid)),
        "M2c Yeoh+sheath 92kPa": dict(wall=fm.yeoh(), sheath=nh(0.092)),
        "NH fitted 0.314":       dict(wall=nh(0.314)),
    }


def P(theta, model, **geom):
    return fm.pressure(theta, wall=model["wall"], sheath=model.get("sheath"), **geom)


# ------------------------------------------------------------------ 1. ladder
def ladder():
    rows = []
    for name, m in stages().items():
        rows.append([name] + [float(P(np.radians(d), m)[0]) for d in ANGLES])
    return rows


# ------------------------------------------------------------------ 2. sensitivity


def sens_nh():
    """Elasticities for the Neo-Hookean wall (mu = 0.314); modulus elasticity is exactly 1."""
    h = 0.01
    res = {}
    for d in ANGLES:
        th = np.radians(d)
        m = dict(wall=fm.neo_hookean(fm.MU_FIT))
        for key in ("a", "b", "t", "L"):
            hi, lo = dict(fm.BASE), dict(fm.BASE)
            hi[key] *= 1 + h; lo[key] *= 1 - h
            res[(key, d)] = (np.log(P(th, m, **hi)[0]) - np.log(P(th, m, **lo)[0])) / (np.log(1 + h) - np.log(1 - h))
        res[("mu", d)] = 1.0
    return res


def sens_yeoh():
    """Elasticities for the Yeoh wall; modulus elasticity by scaling C1, C2 together."""
    h = 0.01
    res = {}
    for d in ANGLES:
        th = np.radians(d)
        m = dict(wall=fm.yeoh())
        for key in ("a", "b", "t", "L"):
            hi, lo = dict(fm.BASE), dict(fm.BASE)
            hi[key] *= 1 + h; lo[key] *= 1 - h
            res[(key, d)] = (np.log(P(th, m, **hi)[0]) - np.log(P(th, m, **lo)[0])) / (np.log(1 + h) - np.log(1 - h))
        hi = dict(wall=fm.yeoh(scale=1 + h)); lo = dict(wall=fm.yeoh(scale=1 - h))
        res[("mu", d)] = (np.log(P(th, hi)[0]) - np.log(P(th, lo)[0])) / (np.log(1 + h) - np.log(1 - h))
    return res


# ------------------------------------------------------------------ 3. Monte Carlo
def sample_geometry(rng, n, sigma_t, mode, sig_a=0.1, sig_b=0.1, sig_L=1.0):
    """mode: 't' (only t), 'corr' (t varies, outer profile fixed -> a shifts by -dt),
             'full' ('corr' plus independent a, b, L tolerances)."""
    B = fm.BASE
    dt = rng.normal(0.0, sigma_t, n)
    g = dict(a=np.full(n, B["a"]), b=np.full(n, B["b"]), t=B["t"] + dt, L=np.full(n, B["L"]))
    if mode in ("corr", "full"):
        g["a"] = B["a"] - dt
    if mode == "full":
        g["a"] = g["a"] + rng.normal(0, sig_a, n)
        g["b"] = g["b"] + rng.normal(0, sig_b, n)
        g["L"] = g["L"] + rng.normal(0, sig_L, n)
    return g


def mc_std(model, sigma_t, mode, n, seed):
    rng = np.random.default_rng(seed)
    g = sample_geometry(rng, n, sigma_t, mode)
    p = P(TH360, model, **g)
    return float(np.std(p, ddof=1)), float(np.mean(p)), p


def crossover(sig, sd, gap):
    """sigma_t where sd(sigma_t) == |gap| (log-log interpolation); None if outside the sweep."""
    gap = abs(gap)
    if gap < sd[0]:
        return "<%.3f" % sig[0]
    if gap > sd[-1]:
        return ">%.3f" % sig[-1]
    return float(np.exp(np.interp(np.log(gap), np.log(sd), np.log(sig))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=249)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ---- validation against the notebook's scipy.quad implementation
    worst = max(abs(fm.pressure(np.radians(d))[0] - fm.pressure_quad_reference(np.radians(d))) for d in (30, 90, 180, 270, 360))
    print(f"[check] vectorised vs scipy.quad notebook model, max |dP| = {worst:.2e} kPa")

    fm.set_quadrature(12)   # exact to ~1e-13 kPa for these smooth integrands; 5x faster for Monte Carlo

    # ---- ladder
    rows = ladder()
    print("\n[ladder] pressure (kPa) at 180/270/360 deg; gap = 207 - P(360)")
    with open(f"{args.out}/ladder.csv", "w") as f:
        f.write("stage,P180,P270,P360,gap_to_207_kPa,gap_pct_of_model\n")
        for r in rows:
            gap = P_EXP - r[3]
            f.write(f"{r[0]},{r[1]:.2f},{r[2]:.2f},{r[3]:.2f},{gap:.2f},{100*gap/r[3]:.2f}\n")
            print(f"  {r[0]:<24} {r[1]:7.1f} {r[2]:7.1f} {r[3]:7.1f}   gap {gap:+6.1f} kPa ({100*gap/r[3]:+.1f}%)")

    # ---- sensitivity (Yeoh)
    sens = sens_yeoh()
    sens_n = sens_nh()
    print("\n[sensitivity] elasticity d ln P / d ln x")
    with open(f"{args.out}/sensitivity.csv", "w") as f:
        f.write("material,angle,a,b,t,L,mu\n")
        for lab_, sv in (("Yeoh", sens), ("NeoHookean_0.314", sens_n)):
            print(f"  {lab_}:   angle " + "   ".join(f"{k:>6}" for k in ("a", "b", "t", "L", "mu")))
            for d in ANGLES:
                v = [sv[(k, d)] for k in ("a", "b", "t", "L", "mu")]
                f.write(f"{lab_},{d}," + ",".join(f"{x:.4f}" for x in v) + "\n")
                print(f"           {d:>4}    " + "   ".join(f"{x:+6.3f}" for x in v))

    # ---- sigma_t sweep
    sigs = np.logspace(np.log10(0.01), np.log10(0.40), 25)
    modes = ("t", "corr", "full")
    models = {k: v for k, v in stages().items() if k.startswith(("M0", "M1", "M2"))}
    sd = {(mn, mo): np.array([mc_std(m, s, mo, args.n, args.seed)[0] for s in sigs])
          for mn, m in models.items() for mo in modes}
    with open(f"{args.out}/mc_sweep.csv", "w") as f:
        f.write("sigma_t_mm," + ",".join(f"sd_kPa[{mn}|{mo}]" for mn in models for mo in modes) + "\n")
        for i, s in enumerate(sigs):
            f.write(f"{s:.4f}," + ",".join(f"{sd[(mn, mo)][i]:.3f}" for mn in models for mo in modes) + "\n")

    # ---- crossover table
    print(f"\n[crossover] sigma_t* (mm) where 1-sigma P(360) scatter = |gap|   (N={args.n}, seed={args.seed})")
    cross_rows = []
    for pexp in (P_EXP, P_EXP_ALT):
        for mn, m in models.items():
            gap = pexp - float(P(TH360, m)[0])
            cs = [crossover(sigs, sd[(mn, mo)], gap) for mo in modes]
            cross_rows.append((pexp, mn, gap, *cs))
            fmt = lambda c: f"{c:.3f}" if isinstance(c, float) else c
            print(f"  P_exp={pexp:.0f}  {mn:<24} gap {gap:+7.1f} kPa   t-only {fmt(cs[0]):>7}  corr {fmt(cs[1]):>7}  full {fmt(cs[2]):>7}")
    with open(f"{args.out}/crossover.csv", "w") as f:
        f.write("P_exp,stage,gap_kPa,sigma_t*_t_only,sigma_t*_corr,sigma_t*_full\n")
        for r in cross_rows:
            f.write(f"{r[0]:.0f},{r[1]},{r[2]:.2f},{r[3]},{r[4]},{r[5]}\n")

    # ---- sigma_t*(mu_s): linearised scatter  sd = |dP/dt| * sigma_t  (checked against the Monte Carlo below)
    K_OFF = 220.0 / 237.8   # uniform scale if the paper's analytical curve (220 kPa) is the correct model value
    def sig_star_lin(mu_s, k=1.0):
        mdl = dict(wall=fm.yeoh(), sheath=fm.neo_hookean(mu_s)) if mu_s > 0 else dict(wall=fm.yeoh())
        p0 = float(P(TH360, mdl)[0])
        dpdt = (float(P(TH360, mdl, t=2.02)[0]) - float(P(TH360, mdl, t=1.98)[0])) / 0.04
        return abs(P_EXP - k * p0) / (k * abs(dpdt)), P_EXP - k * p0
    mus = np.linspace(0.0, 0.100, 201)
    ss0 = np.array([sig_star_lin(m)[0] for m in mus]); ss1 = np.array([sig_star_lin(m, K_OFF)[0] for m in mus])
    with open(f"{args.out}/sigma_star_vs_mus.csv", "w") as f:
        f.write("mu_s_kPa,sigma_star_mm_k1,gap_k1,sigma_star_mm_k0.925,gap_k0.925\n")
        for m_, a_, b_ in zip(mus, ss0, ss1):
            f.write(f"{1e3*m_:.1f},{a_:.4f},{sig_star_lin(m_)[1]:.2f},{b_:.4f},{sig_star_lin(m_, K_OFF)[1]:.2f}\n")
    print("\n[sigma*(mu_s), linearised]  mu_s(kPa): k=1 / k=%.3f" % K_OFF)
    for m_ in (0.0, 0.025, 0.030, 0.0392, 0.060, 0.092):
        print(f"  {1e3*m_:5.1f}: {sig_star_lin(m_)[0]:.3f} / {sig_star_lin(m_, K_OFF)[0]:.3f}")
    mu_zero = mus[np.argmin(ss0[1:]) + 1]
    print(f"  gap = 0 near mu_s = {1e3*mu_zero:.1f} kPa (k=1);  slope check: linear sigma*(M1)={sig_star_lin(0.0)[0]:.4f} vs MC t-only {crossover(sigs, sd[('M1 Yeoh','t')], P_EXP-float(P(TH360, models['M1 Yeoh'])[0])):.4f}")

    # ================= figures =================
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "legend.fontsize": 9,
                         "xtick.labelsize": 10, "ytick.labelsize": 10, "font.family": "serif"})
    # ---- main multi-panel figure (all text >= 10 pt at 6.5 in width)
    fig = plt.figure(figsize=(6.5, 2.75))
    gsm = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], wspace=0.36)
    ax = fig.add_subplot(gsm[0, 0])
    colors = {"t": "#1f77b4", "corr": "#d62728", "full": "#2ca02c"}
    lab = {"t": r"$t$ only", "corr": "corr", "full": "full"}
    for mo in modes:
        ax.plot(sigs, sd[("M1 Yeoh", mo)], color=colors[mo], lw=1.8, label=lab[mo])
    gaps = {mn: abs(P_EXP - float(P(TH360, m)[0])) for mn, m in models.items() if not mn.startswith("M0")}
    lo_g, hi_g = min(gaps.values()), max(gaps.values())
    ax.axhspan(lo_g, hi_g, color="0.8", alpha=0.7)
    ax.axhline(gaps["M1 Yeoh"], color="k", ls="--", lw=1.3)
    # Reference fabrication tolerance: McCall et al. 2024 (caliper, 3D-printed-mold DS20 chambers)
    ax.axvspan(0.04, 0.06, color="tab:purple", alpha=0.30)
    ax.axvspan(0.06, 0.128, color="tab:purple", alpha=0.12, hatch="//", lw=0)
    ax.text(0.39, 1.25, r"$|$gap$|$, sheath range", fontsize=10, va="bottom", ha="right")
    ax.text(0.0105, gaps["M1 Yeoh"] * 1.06, r"$|$gap$|$ after Yeoh", fontsize=10, va="bottom")
    c = crossover(sigs, sd[("M1 Yeoh", "t")], gaps["M1 Yeoh"])
    if isinstance(c, float):
        ax.plot([c], [gaps["M1 Yeoh"]], "ko", ms=6)
        ax.annotate(f"$\\sigma_t^*$={c:.2f} mm", (c, gaps["M1 Yeoh"]), xytext=(0.34, 6.0), ha="right",
                    fontsize=10, arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set(xscale="log", yscale="log", xlabel=r"tolerance $\sigma_t$ (mm)",
           ylabel=r"1$\sigma$ scatter of $P_{360^\circ}$ (kPa)", ylim=(1.1, 110))
    ax.text(0.0, 1.02, "(a)", fontsize=10, va="bottom", fontweight="bold", transform=ax.transAxes)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", frameon=False, fontsize=10, handlelength=1.4, labelspacing=0.25)
    fmt_g = FuncFormatter(lambda v, _: f"{v:g}")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        axis.set_major_formatter(fmt_g); axis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    # panel (b): crossover as a function of sheath stiffness
    axs = fig.add_subplot(gsm[0, 1])
    axs.axhspan(0.04, 0.06, color="tab:purple", alpha=0.30)
    axs.axhspan(0.06, 0.128, color="tab:purple", alpha=0.12, hatch="//", lw=0)
    axs.plot(1e3 * mus, ss0, color="k", lw=1.8, label="model as is")
    axs.plot(1e3 * mus, ss1, color="0.45", lw=1.6, ls="--", label=r"offset $\times0.925$")
    axs.set(yscale="log", ylim=(0.003, 1.0), xlim=(0, 100), xlabel=r"sheath modulus $\mu_s$ (kPa)",
            ylabel=r"crossover $\sigma_t^*$ (mm)")
    axs.text(0.0, 1.02, "(b)", fontsize=10, va="bottom", fontweight="bold", transform=axs.transAxes)
    axs.legend(loc="upper center", frameon=False, fontsize=10, handlelength=1.4, labelspacing=0.25)
    axs.grid(True, which="both", alpha=0.3)
    axs.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    axs.yaxis.set_major_formatter(fmt_g); axs.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    fig.savefig(f"{args.out}/fig_main.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # supplementary panels (elasticities, MC distribution), not used in the 3-page report
    fig = plt.figure(figsize=(6.5, 2.2))
    gs = fig.add_gridspec(1, 2, wspace=0.42)

    axb = fig.add_subplot(gs[0, 0])
    keys = ("a", "b", "t", "L", "mu")
    xlab = ("a", "b", "t", "L", r"$\mu$")
    w = 0.26
    for i_, d in enumerate(ANGLES):
        axb.bar(np.arange(5) + (i_ - 1) * w, [sens[(k, d)] for k in keys], w, label=f"{d}°")
    axb.axhline(0, color="k", lw=0.6)
    axb.set_xticks(range(5)); axb.set_xticklabels(xlab)
    axb.set(ylabel=r"$d\ln P/d\ln x$", ylim=(-1.2, 2.15))
    axb.legend(frameon=False, fontsize=10, ncol=3, loc="upper center", columnspacing=0.6, handlelength=0.8, handletextpad=0.3)
    axb.set_title("(b) Elasticities (Yeoh)", fontsize=10)

    axh = fig.add_subplot(gs[0, 1])
    for s_, col in ((0.1, "#1f77b4"), (0.25, "#ff7f0e")):
        _, mean, p = mc_std(models["M1 Yeoh"], s_, "t", args.n, args.seed)
        axh.hist(p, bins=60, density=True, alpha=0.6, color=col, label=rf"$\sigma_t$={s_}")
    axh.axvline(P_EXP, color="k", ls="--", lw=1.2, label="exp.")
    axh.set(xlabel=r"$P_{360^\circ}$ (kPa)", ylabel="density", xlim=(80, 400), ylim=(0, 0.036))
    axh.legend(frameon=False, fontsize=10, loc="upper right", labelspacing=0.2, handlelength=1.2)
    axh.set_title("(c) MC distribution", fontsize=10)
    fig.savefig(f"{args.out}/fig_extra.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- ladder ("gap budget") figure
    fig, ax = plt.subplots(figsize=(6.5, 2.75))
    names = [r[0] for r in rows]
    vals = [r[3] for r in rows]
    ax.bar(range(len(rows)), vals, color="#8da0cb")
    ax.axhline(P_EXP, color="k", ls="--", lw=1.2, label="measured 207 kPa")
    ax.set_xticks(range(len(rows)))
    short = ["M0\nNH 0.22", "M1\nYeoh", "M2a\n+sheath\n25 kPa", "M2b\n+sheath\n39 kPa", "M2c\n+sheath\n92 kPa", "NH fit\n0.314"]
    ax.set_xticklabels(short, fontsize=9)
    for i, v in enumerate(vals):
        ax.text(i, v + 3, f"{v:.0f}", ha="center", fontsize=9)
    ax.set(ylabel=r"$P_{360^\circ}$ (kPa)", ylim=(0, 280))
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(f"{args.out}/fig_ladder.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote CSV + figures to {args.out}/")


if __name__ == "__main__":
    main()
