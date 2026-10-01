"""Cross-fold-consensus backward feature elimination with shadow-decoy
stopping. Every decision - which feature to drop, when to stop - is made
from out-of-fold CV only; the fixed holdout is never touched by this
module.

Algorithm, per model, fixed hyperparameters throughout:
  1. Start from a candidate set (already correlation-pruned).
  2. Add one shadow feature per real candidate - a row-permuted copy with
     the same marginal distribution but no relationship to the target,
     re-randomized every step. A trained model's own permutation
     importance for a shadow column is the noise floor: any real feature
     that doesn't clearly beat it is behaving like noise to this model.
  3. Train all fold models fresh, compute every feature's (real + shadow)
     permutation importance on that fold's own held-out validation split
     (never train, never holdout), averaged across folds.
  4. Drop the worst real feature if its importance is at or below the
     shadow floor, and repeat from step 2.
  5. Stop when either (a) the worst remaining real feature already beats
     the shadow floor, or (b) mean CV RMSE has degraded by more than
     `tol` (relative) from the best point seen on the path - a safety net
     for cases where importance ranking and realized CV error disagree.
"""
import numpy as np

import metrics as M
from spatial_cv import Fold

MODEL_MODULES = {}


def register_model(name, module):
    MODEL_MODULES[name] = module


def _fold_with_shadows(fold, real_feats, rng):
    train_s = fold.train.copy()
    val_s = fold.val.copy()
    shadow_feats = []
    for f in real_feats:
        sf = f"shadow__{f}"
        train_s[sf] = rng.permutation(train_s[f].values)
        val_s[sf] = rng.permutation(val_s[f].values)
        shadow_feats.append(sf)
    return Fold(part=fold.part, train=train_s, val=val_s), shadow_feats


def _eval_and_importance_one_seed(mod_name, folds, real_feats, target, params, weight_fn, seed):
    mod = MODEL_MODULES[mod_name]
    rng = np.random.default_rng(seed)

    fold_val_rmse = []
    importance_accum = None
    for fold in folds:
        shadow_fold, shadow_feats = _fold_with_shadows(fold, real_feats, rng)
        all_feats = real_feats + shadow_feats
        if importance_accum is None:
            importance_accum = {f: [] for f in all_feats}

        models = mod.train_fold_models([shadow_fold], all_feats, target, params, weight_fn=weight_fn)
        model = models[0]
        y = shadow_fold.val[target].values
        base_pred = mod.predict_ensemble([model], shadow_fold.val[all_feats])
        base_rmse = M.rmse(y, base_pred)
        fold_val_rmse.append(base_rmse)

        for f in all_feats:
            permuted = shadow_fold.val.copy()
            permuted[f] = rng.permutation(permuted[f].values)
            pred = mod.predict_ensemble([model], permuted[all_feats])
            importance_accum[f].append(M.rmse(y, pred) - base_rmse)

    mean_importance = {f: float(np.mean(v)) for f, v in importance_accum.items()}
    return float(np.mean(fold_val_rmse)), mean_importance


def eval_and_importance(mod_name, folds, real_feats, target, params, weight_fn=None, seed=47, n_seeds=3):
    """Train fresh fold models on real_feats + shadow copies, repeated over
    n_seeds independent shadow-permutation draws and averaged - a single
    seed's shadow re-randomization is itself a source of step-to-step
    training noise, comparable in size to the 0.5% stop tolerance, so
    averaging over multiple seeds is what keeps the elimination and
    stop-trigger decisions from reacting to that noise instead of real
    feature effects. Returns (mean_cv_rmse, importance dict for every
    real+shadow feature)."""
    rmse_vals, importance_vals = [], []
    for i in range(n_seeds):
        r, imp = _eval_and_importance_one_seed(mod_name, folds, real_feats, target, params,
                                                weight_fn, seed=seed * 1000 + i)
        rmse_vals.append(r)
        importance_vals.append(imp)
    mean_importance = {f: float(np.mean([imp[f] for imp in importance_vals])) for f in importance_vals[0]}
    return float(np.mean(rmse_vals)), mean_importance


def backward_eliminate(mod_name, folds, target, params, real_feats, weight_fn=None,
                        tol=0.005, seed=47, log=print):
    remaining = list(real_feats)
    path = []
    best_rmse, best_step = np.inf, None
    step = 0

    while len(remaining) > 1:
        mean_rmse, importance = eval_and_importance(mod_name, folds, remaining, target, params,
                                                      weight_fn=weight_fn, seed=seed + step)
        real_imp = {f: importance[f] for f in remaining}
        shadow_floor = max(importance[f] for f in importance if f.startswith("shadow__"))

        path.append(dict(step=step, n_features=len(remaining), mean_cv_rmse=mean_rmse,
                          shadow_floor=shadow_floor, features=list(remaining),
                          real_importance=real_imp))
        log(f"  [{mod_name}] step {step}: n={len(remaining)} cv_rmse={mean_rmse:.4f} "
            f"shadow_floor={shadow_floor:.4f}")

        if mean_rmse < best_rmse:
            best_rmse, best_step = mean_rmse, step

        if mean_rmse > best_rmse * (1 + tol):
            log(f"  [{mod_name}] stop: cv_rmse degraded >{tol:.1%} from best ({best_rmse:.4f})")
            break

        worst_feat = min(real_imp, key=real_imp.get)
        if real_imp[worst_feat] > shadow_floor:
            log(f"  [{mod_name}] stop: worst real feature '{worst_feat}' "
                f"({real_imp[worst_feat]:.4f}) beats shadow floor ({shadow_floor:.4f})")
            break

        log(f"  [{mod_name}] drop '{worst_feat}' (importance {real_imp[worst_feat]:.4f} "
            f"<= shadow floor {shadow_floor:.4f})")
        remaining.remove(worst_feat)
        step += 1

    if not path:
        return dict(path=path, final_features=list(remaining), best_step=None)
    final_features = path[best_step]["features"] if best_step is not None else path[-1]["features"]
    return dict(path=path, final_features=final_features, best_step=best_step)


def joint_backward_eliminate(model_names, folds_by_model, target, params_by_model, real_feats,
                              weight_fn=None, tol=0.005, seed=47, log=print):
    """Same elimination loop, but a feature's importance is the mean of its
    per-model importance normalized by that model's own shadow floor (so
    models with different RMSE scales contribute comparably), and the
    safety stop fires if ANY individual model's CV RMSE degrades too much
    - the joint set should stay safe for all models, not just on average."""
    remaining = list(real_feats)
    path = []
    best_rmse_by_model = {m: np.inf for m in model_names}
    step = 0

    while len(remaining) > 1:
        per_model = {}
        for m in model_names:
            mean_rmse, importance = eval_and_importance(
                m, folds_by_model[m], remaining, target, params_by_model[m],
                weight_fn=weight_fn, seed=seed + step)
            shadow_floor = max(importance[f] for f in importance if f.startswith("shadow__"))
            per_model[m] = dict(mean_rmse=mean_rmse, importance=importance, shadow_floor=shadow_floor)

        joint_norm_importance = {}
        for f in remaining:
            ratios = [per_model[m]["importance"][f] / max(per_model[m]["shadow_floor"], 1e-9)
                      for m in model_names]
            joint_norm_importance[f] = float(np.mean(ratios))

        # degradation check compares against the best seen on PRIOR steps only
        degraded = [m for m in model_names
                    if per_model[m]["mean_rmse"] > best_rmse_by_model[m] * (1 + tol)]
        for m in model_names:
            if per_model[m]["mean_rmse"] < best_rmse_by_model[m]:
                best_rmse_by_model[m] = per_model[m]["mean_rmse"]

        path.append(dict(step=step, n_features=len(remaining), features=list(remaining),
                          per_model={m: dict(mean_cv_rmse=per_model[m]["mean_rmse"],
                                              shadow_floor=per_model[m]["shadow_floor"])
                                     for m in model_names},
                          joint_norm_importance=joint_norm_importance))
        log(f"  [joint] step {step}: n={len(remaining)} " +
            " ".join(f"{m}_rmse={per_model[m]['mean_rmse']:.4f}" for m in model_names))

        if degraded:
            log(f"  [joint] stop: {degraded} degraded >{tol:.1%} from their own best on prior steps")
            break

        worst_feat = min(joint_norm_importance, key=joint_norm_importance.get)
        if joint_norm_importance[worst_feat] > 1.0:
            log(f"  [joint] stop: worst real feature '{worst_feat}' "
                f"(norm importance {joint_norm_importance[worst_feat]:.3f}) beats the noise floor (1.0) for all models")
            break

        log(f"  [joint] drop '{worst_feat}' (norm importance {joint_norm_importance[worst_feat]:.3f} <= 1.0)")
        remaining.remove(worst_feat)
        step += 1

    # pick the step that minimizes mean(rmse_m / true_best_rmse_m), where
    # true_best_rmse_m is each model's best seen anywhere on the COMPLETED
    # path, computed once at the end (not concurrently with the comparison)
    true_best_by_model = {m: min(p["per_model"][m]["mean_cv_rmse"] for p in path) for m in model_names}
    best_joint_score, best_step = np.inf, None
    for p in path:
        score = float(np.mean([p["per_model"][m]["mean_cv_rmse"] / true_best_by_model[m] for m in model_names]))
        p["joint_score"] = score
        if score < best_joint_score:
            best_joint_score, best_step = score, p["step"]

    if not path:
        return dict(path=path, final_features=list(remaining), best_step=None)
    final_features = path[best_step]["features"] if best_step is not None else path[-1]["features"]
    return dict(path=path, final_features=final_features, best_step=best_step)
