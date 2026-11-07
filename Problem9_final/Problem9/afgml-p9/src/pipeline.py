"""Core experiment pipeline: exact reference, deterministic op budgets, fallbacks,
fairness-aware fallback selection (validation only), baselines and metrics."""
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from src.bn import predict, fit_bn
from src.fallbacks import candidate_fallbacks
from src.fairness import fit_eopp_thresholds, apply_thresholds
from src.metrics import fairness_summary, utility, prob_metrics, group_rates


def make_mask(frame, cfg, band_labels, workload, seed, split_id):
    """SYNTHETIC extra masking of maskable variables (True = hidden)."""
    if workload == "natural": return None
    rng = np.random.default_rng([seed, 7, split_id])
    rates = np.array(cfg["workloads"]["dependent"]["rates_by_band"], float)
    idx = frame["age_band"].map({b: i for i, b in enumerate(band_labels)}).to_numpy()
    row_rate = rates[idx]
    if workload == "independent": row_rate = np.full(len(frame), row_rate.mean())
    return pd.DataFrame({v: rng.random(len(frame)) < row_rate for v in cfg["mask_vars"]}, index=frame.index)


def eval_method(y, score, d, g, ref_d, ref_score, used_fb, ms):
    fs, fsu = fairness_summary(y, d, g, ms), fairness_summary(y, d, g, ms, exclude_exploratory=True)
    out = dict(dEOpp=fs["dEOpp"], dEOpp_supported=fsu["dEOpp"], dDP=fs["dDP"], dEOdds=fs["dEOdds"],
               sel_ratio=fs["sel_ratio"], worst_group_tpr=fs["worst_group_tpr"],
               n_groups=fs["n_groups"], n_exploratory_tpr=fs["n_exploratory_tpr"])
    out.update(utility(y, score, d)); out.update(prob_metrics(y, score))
    sg = pd.Series(g); n_by = sg.value_counts(); sup = n_by[n_by >= ms].index
    flip = pd.Series(np.asarray(d) != np.asarray(ref_d)).groupby(g).mean()
    fb = pd.Series(np.asarray(used_fb, float)).groupby(g).mean()
    out.update(agree_exact=float(np.mean(np.asarray(d) == np.asarray(ref_d))),
               worst_group_flip=float(flip[sup].max()) if len(sup) else np.nan,
               fallback_rate=float(np.mean(used_fb)),
               fallback_gap=float(fb[sup].max() - fb[sup].min()) if len(sup) else np.nan,
               tv_mean=float(np.mean(np.abs(score - ref_score))), tv_max=float(np.max(np.abs(score - ref_score))))
    return out


def _cand_val_stats(y, score, d, g, ref_d, ref_score, ms, ap_exact):
    n_by = pd.Series(g).value_counts(); sup = n_by[n_by >= ms].index
    flip = pd.Series(np.asarray(d) != np.asarray(ref_d)).groupby(g).mean()
    return dict(ap_loss=float(ap_exact - average_precision_score(y, score)),
                worst_flip=float(flip[sup].max()) if len(sup) else float(flip.max()),
                logloss=prob_metrics(y, score)["log_loss"])


def select_fallback(stats: dict, tol, mode):
    names = list(stats)                                     # insertion order = deterministic tie-break
    assert names, "no candidate fallbacks to select from"
    if mode == "fair":
        ok = [n for n in names if stats[n]["ap_loss"] <= tol]
        pool, key = (ok, lambda n: (stats[n]["worst_flip"], stats[n]["ap_loss"], names.index(n))) if ok else \
                    (names, lambda n: (stats[n]["ap_loss"], names.index(n)))
    else:
        pool, key = names, lambda n: (stats[n]["logloss"], names.index(n))
    return min(pool, key=key)


def natural_budgets(factors, states, val, heuristic, qs):
    ops = predict(val, factors, states, heuristic=heuristic)["ops"].to_numpy()
    return {q: int(np.ceil(np.quantile(ops, q))) for q in qs}


def run_core(splits, factors, states, cfg, band_labels, *, heuristic="min_fill", capacity=None,
             simplify=True, workloads=("natural", "dependent", "independent"), seeds=(0,),
             budgets: dict, collect_q=None, tag="main"):
    capacity = cfg["capacity"] if capacity is None else capacity
    ms, tol = cfg["min_group_support"], cfg["utility_loss_tol"]
    val, te = splits["val"], splits["test"]
    yv, yt, gv, gt = val.y.to_numpy(), te.y.to_numpy(), val.group.to_numpy(), te.group.to_numpy()
    fbs = candidate_fallbacks(factors, states, cfg["fallbacks"])
    rows, grows, chosen, arts = [], [], [], {}
    for w in workloads:
        for seed in (seeds if w != "natural" else (0,)):
            mv, mt = make_mask(val, cfg, band_labels, w, seed, 0), make_mask(te, cfg, band_labels, w, seed, 1)
            exv = predict(val, factors, states, heuristic=heuristic, extra_mask=mv)
            ext = predict(te, factors, states, heuristic=heuristic, extra_mask=mt)
            thr = float(np.quantile(exv.p1, 1 - capacity))
            dv_ex, dt_ex = (exv.p1.to_numpy() >= thr).astype(int), (ext.p1.to_numpy() >= thr).astype(int)
            ap_ex = average_precision_score(yv, exv.p1)
            fair_thr, fair_glob = fit_eopp_thresholds(exv.p1, yv, gv, capacity, ms)
            d_fairex = apply_thresholds(ext.p1, gt, fair_thr, fair_glob)
            for q, B in budgets.items():
                pv, pt = {}, {}
                for n, fb in fbs.items():
                    pv[n] = predict(val, factors, states, B, heuristic, simplify, fb, mv)
                    pt[n] = predict(te, factors, states, B, heuristic, simplify, fb, mt)
                stats = {n: _cand_val_stats(yv, pv[n].p1.to_numpy(), (pv[n].p1.to_numpy() >= thr).astype(int),
                                            gv, dv_ex, exv.p1.to_numpy(), ms, ap_ex) for n in fbs}
                pick_f, pick_u = select_fallback(stats, tol, "fair"), select_fallback(stats, tol, "util")
                # group-UNAWARE routing comparison (required by the brief when a protected attribute
                # can be used in routing): same fairness-aware rule, but candidates that condition on
                # the protected attribute are excluded from the pool.
                unaware = {n: st for n, st in stats.items() if not cfg["fallbacks"][n].get("uses_protected", False)}
                pick_g = select_fallback(unaware, tol, "fair")
                gp = pv["prior"]
                thr_pb, glob_pb = fit_eopp_thresholds(gp.p1, yv, gv, capacity, ms)
                scores = {"exact": (ext.p1, pd.Series(np.zeros(len(te), bool))),
                          "global_prior": (pt["prior"].p1, pt["prior"].used_fallback),
                          "proposed": (pt[pick_f].p1, pt[pick_f].used_fallback),
                          "util_sel": (pt[pick_u].p1, pt[pick_u].used_fallback),
                          "proposed_unaware": (pt[pick_g].p1, pt[pick_g].used_fallback),
                          **{f"cand_{n}": (pt[n].p1, pt[n].used_fallback) for n in fbs if n != "prior"}}
                decs = {m: (s.to_numpy() >= thr).astype(int) for m, (s, _) in scores.items()}
                decs["thr_adj_post_budget"] = apply_thresholds(pt["prior"].p1.to_numpy(), gt, thr_pb, glob_pb)
                decs["fair_exact"] = d_fairex
                scores["thr_adj_post_budget"] = scores["global_prior"]; scores["fair_exact"] = scores["exact"]
                chosen.append(dict(tag=tag, workload=w, seed=seed, q=q, budget_ops=B, heuristic=heuristic,
                                   simplify=simplify, capacity=capacity, proposed=pick_f, util_sel=pick_u, proposed_unaware=pick_g,
                                   **{f"val_{n}_{k}": v for n, st in stats.items() for k, v in st.items()}))
                for m, d in decs.items():
                    sc, ufb = scores[m]
                    ref_d = d_fairex if m in ("fair_exact", "thr_adj_post_budget") else dt_ex
                    r = eval_method(yt, sc.to_numpy(), d, gt, ref_d, ext.p1.to_numpy(), ufb.to_numpy(), ms)
                    uses_a = (m in ("thr_adj_post_budget", "fair_exact") or
                              (m == "proposed" and cfg["fallbacks"][pick_f].get("uses_protected", False)) or
                              (m == "util_sel" and cfg["fallbacks"][pick_u].get("uses_protected", False)) or
                              (m == "cand_age_prior"))
                    rows.append(dict(tag=tag, workload=w, seed=seed, q=q, budget_ops=B, heuristic=heuristic,
                                     simplify=simplify, capacity=capacity, method=m,
                                     group_label_at_decision=bool(uses_a),
                                     mean_ops=float(np.minimum(ext.ops, B).mean()) if m != "exact" else float(ext.ops.mean()),
                                     **r))
                if collect_q is not None and q == collect_q and seed == 0 and w in ("natural", "dependent"):
                    arts[w] = dict(y=yt, group=gt, **{f"d_{m}": d for m, d in decs.items()},
                                   **{f"score_{m}": sc.to_numpy() for m, (sc, _) in scores.items()},
                                   **{f"fb_{m}": u.to_numpy() for m, (_, u) in scores.items()})
                    for m, d in decs.items():
                        gr = group_rates(yt, d, gt, ms)
                        gr["flip"] = [float(np.mean(d[gt == k] != dt_ex[gt == k])) for k in gr.group]
                        gr["fallback_rate"] = [float(scores[m][1].to_numpy()[gt == k].mean()) for k in gr.group]
                        gr.insert(0, "method", m); gr.insert(0, "workload", w); grows.append(gr)
    return rows, grows, chosen, arts
