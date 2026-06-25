#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K-Risk dataset statistics recomputation.

Produces (all written into KHHRED/):
  - K-Risk_statistics.md                       (Task 1/2/3 + Task 4 summary)
  - task4_ttc_distribution.csv                 (TTC histogram: K-Risk vs Original)
  - task4_speed_distribution_by_scenario.csv   (Speed histogram per scenario)
  - task4_ttc_speed_joint_distribution.csv     (TTC x Speed 2D histogram)

Methodology decisions (confirmed with user):
  - De-dup unit = file basename. extreme/ files are merged into expresswayA/freewayB
    (they originate from those sources) and never counted as a standalone source.
  - Behavior distribution: only highD / expresswayA / freewayB carry per-frame
    behavior labels; ind/round/ngsim are excluded from behavior counts.
  - TTC / speed distributions: per-ego, per-frame values.
  - Original Data (trajectory_data): sampled estimate (subset of CSVs + row sampling).
  - Risk levels are mutually exclusive: Extreme = dedup(extreme/);
    High = high_risk folders minus extreme; Moderate = normal_risk minus extreme.
"""

import os
import json
import glob
import math
import random
from collections import defaultdict, Counter

import numpy as np

random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
EVENT = os.path.join(HERE, "event_annotations")
HV = os.path.join(EVENT, "HV")
AV = os.path.join(EVENT, "AV")
TRAJ_HV = os.path.join(HERE, "trajectory_data", "HV")

OUT_MD = os.path.join(HERE, "K-Risk_statistics.md")
OUT_TTC = os.path.join(HERE, "task4_ttc_distribution.csv")
OUT_SPEED = os.path.join(HERE, "task4_speed_distribution.csv")
OUT_JOINT = os.path.join(HERE, "task4_ttc_speed_joint_distribution.csv")
# legacy output (per-scenario speed) superseded by HighD K-Risk vs Original
OUT_SPEED_LEGACY = os.path.join(HERE, "task4_speed_distribution_by_scenario.csv")

# Task 5: per-dataset speed distribution across the 6 HV sources
OUT_T5_HIST = os.path.join(HERE, "task5_speed_distribution_by_dataset_hist.csv")
OUT_T5_SUMMARY = os.path.join(HERE, "task5_speed_distribution_by_dataset_summary.csv")
OUT_T5_SAMPLES = os.path.join(HERE, "task5_speed_distribution_by_dataset_samples.csv")

HV_SOURCES = ["expresswayA", "freewayB", "highd", "ind", "ngsim", "round"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def json_basenames(folder):
    """Set of *.json basenames directly under `folder`."""
    return set(os.path.basename(p) for p in glob.glob(os.path.join(folder, "*.json")))


def load_json(path):
    with open(path) as f:
        return json.load(f)


def iter_vehicles(d):
    """Yield per-vehicle dicts from either list-of-records, nested-frames,
    or list-of-{frame_id,vehicles} structures."""
    if isinstance(d, dict) and "frames" in d:
        for fr in d["frames"]:
            for v in fr.get("vehicles", []):
                yield v
    elif isinstance(d, list):
        for x in d:
            if isinstance(x, dict) and "vehicles" in x:
                for v in x["vehicles"]:
                    yield v
            elif isinstance(x, dict):
                yield x


def iter_frames(d):
    """Yield lists-of-vehicles grouped per frame (for ego-per-frame extraction)."""
    if isinstance(d, dict) and "frames" in d:
        for fr in d["frames"]:
            yield fr.get("vehicles", [])
    elif isinstance(d, list):
        # nested {frame_id, vehicles}
        if d and isinstance(d[0], dict) and "vehicles" in d[0]:
            for fr in d:
                yield fr.get("vehicles", [])
        else:
            # flat list of per-frame-per-vehicle records: group by frame field
            buckets = defaultdict(list)
            order = []
            for rec in d:
                if not isinstance(rec, dict):
                    continue
                fk = None
                for k in ("frame", "frame_id", "Frame_ID"):
                    if k in rec:
                        fk = rec[k]
                        break
                if fk not in buckets:
                    order.append(fk)
                buckets[fk].append(rec)
            for fk in order:
                yield buckets[fk]


def ego_id_from_filename(source, fname):
    """Extract ego vehicle id from filename to pick the ego record within a frame."""
    base = fname[:-5] if fname.endswith(".json") else fname
    parts = base.split("_")
    try:
        if source == "highd":
            # highd_<track>_<egoId>_<relation>_...
            return ("highd", float(parts[2]))
        if source in ("expresswayA", "freewayB"):
            # <src>_track_<n>_car_<id>_frame_...
            i = parts.index("car")
            return (source, float(parts[i + 1]))
        if source in ("ind", "round"):
            # recording_<n>_ego_<id>_frame_...
            i = parts.index("ego")
            return (source, float(parts[i + 1]))
        if source == "ngsim":
            # ngsim_car<ID>_frame_...
            for p in parts:
                if p.startswith("car"):
                    return ("ngsim", float(p[3:]))
    except (ValueError, IndexError):
        return None
    return None


def vehicle_id(source, v):
    for k in ("id", "car_id", "Vehicle_ID", "trackId"):
        if k in v:
            try:
                return float(v[k])
            except (TypeError, ValueError):
                return v[k]
    return None


def vehicle_speed(source, v):
    """Return speed in m/s for a vehicle record, or None."""
    if "speed" in v and v["speed"] is not None:
        try:
            return float(v["speed"])
        except (TypeError, ValueError):
            pass
    if source == "ngsim" and v.get("v_Vel") is not None:
        try:
            return float(v["v_Vel"])
        except (TypeError, ValueError):
            pass
    # ind / round: compose from velocity components
    vx = v.get("xVelocity")
    vy = v.get("yVelocity")
    if vx is not None and vy is not None:
        try:
            return math.hypot(float(vx), float(vy))
        except (TypeError, ValueError):
            pass
    if v.get("lonVelocity") is not None:
        try:
            return abs(float(v["lonVelocity"]))
        except (TypeError, ValueError):
            pass
    return None


def vehicle_ttc(source, v):
    """Return TTC in seconds for a vehicle record, or None."""
    for k in ("ttc", "TTC", "Time to Collision"):
        if k in v and v[k] is not None:
            try:
                t = float(v[k])
                return t
            except (TypeError, ValueError):
                continue
    return None


def vehicle_class_bucket(source, v):
    """Map a vehicle record to a unified class bucket, or None if unknown."""
    raw = None
    if source == "highd":
        raw = v.get("vehicle_class")
        if raw is not None:
            r = str(raw).strip().lower()
            if r == "car":
                return "Car"
            if r == "truck":
                return "Truck/Bus"
    elif source == "ngsim":
        raw = v.get("v_Class")
        if raw is not None:
            r = str(raw).strip()
            # NGSIM coding: 1=motorcycle, 2=auto/car, 3=truck
            return {"1": "Motorcycle", "2": "Car", "3": "Truck/Bus"}.get(r)
    elif source in ("ind", "round"):
        raw = v.get("class")
        if raw is not None:
            r = str(raw).strip().lower()
            mapping = {
                "car": "Car",
                "van": "Van",
                "truck": "Truck/Bus",
                "truck_bus": "Truck/Bus",
                "bus": "Truck/Bus",
                "trailer": "Trailer",
                "motorcycle": "Motorcycle",
                "bicycle": "Bicycle",
                "pedestrian": "Pedestrian",
            }
            return mapping.get(r)
    # expresswayA / freewayB: no class field
    return None


# ---------------------------------------------------------------------------
# Build the de-dup sample sets shared by Task 1 / 2 / 3
# ---------------------------------------------------------------------------
def build_sets():
    sets = {}
    for s in HV_SOURCES:
        hi = json_basenames(os.path.join(HV, s, f"{s}_high_risk"))
        no = json_basenames(os.path.join(HV, s, f"{s}_normal_risk"))
        sets[s] = {"high": hi, "normal": no}

    ex1 = json_basenames(os.path.join(HV, "extreme", "ttc_1s"))
    ex2 = json_basenames(os.path.join(HV, "extreme", "ttc_2s"))
    extreme = ex1 | ex2
    extreme_eA = set(n for n in extreme if n.startswith("expresswayA"))
    extreme_fB = set(n for n in extreme if n.startswith("freewayB"))
    sets["_extreme"] = {
        "all": extreme, "ttc_1s": ex1, "ttc_2s": ex2,
        "expresswayA": extreme_eA, "freewayB": extreme_fB,
    }
    return sets


# ===========================================================================
# TASK 1 — data source distribution (7 items, exact)
# ===========================================================================
def task1(sets):
    rows = []
    total = 0
    src_union = {}
    for s in HV_SOURCES:
        union = sets[s]["high"] | sets[s]["normal"]
        if s == "expresswayA":
            union = union | sets["_extreme"]["expresswayA"]
        if s == "freewayB":
            union = union | sets["_extreme"]["freewayB"]
        src_union[s] = union
        rows.append((s, len(union)))
        total += len(union)

    # AV: dedup by (sub-dataset, basename)
    av_files = glob.glob(os.path.join(AV, "**", "*.json"), recursive=True)
    av_all = len(av_files)
    av_keys = set()
    av_basenames = set()
    for p in av_files:
        rel = os.path.relpath(p, AV)
        sub = rel.split(os.sep)[0]
        bn = os.path.basename(p)
        av_keys.add((sub, bn))
        av_basenames.add(bn)
    av_dedup = len(av_keys)
    rows.append(("AV", av_dedup))
    total += av_dedup

    return {
        "rows": rows, "total": total, "src_union": src_union,
        "av_all": av_all, "av_dedup": av_dedup, "av_unique_basenames": len(av_basenames),
    }


# ===========================================================================
# TASK 2 — HV risk-level distribution (mutually exclusive)
# ===========================================================================
def task2(sets):
    extreme = sets["_extreme"]["all"]
    # extreme membership keyed by basename, split per owning source already.
    extreme_by_src = {"expresswayA": sets["_extreme"]["expresswayA"],
                      "freewayB": sets["_extreme"]["freewayB"]}

    per_src = []
    tot_mod = tot_high = 0
    for s in HV_SOURCES:
        ex_here = extreme_by_src.get(s, set())
        # priority: Extreme > High > Moderate. A file appearing in both the
        # high and normal folders (NGSIM has 92 such) is counted as High only;
        # remove extreme members from both, then remove high members from
        # normal so the three levels stay mutually exclusive.
        high = sets[s]["high"] - ex_here
        normal = sets[s]["normal"] - ex_here - high
        per_src.append((s, len(normal), len(high)))
        tot_mod += len(normal)
        tot_high += len(high)

    tot_ext = len(extreme)
    return {
        "per_src": per_src,
        "moderate": tot_mod, "high": tot_high, "extreme": tot_ext,
        "total": tot_mod + tot_high + tot_ext,
        "extreme_eA": len(sets["_extreme"]["expresswayA"]),
        "extreme_fB": len(sets["_extreme"]["freewayB"]),
    }


# ===========================================================================
# TASK 3 — Figure 6 four panels
# ===========================================================================
ENV_MAP = {
    "Urban Freeway": ["expresswayA", "freewayB"],
    "High-Speed Highway": ["highd", "ngsim"],
    "Roundabout": ["round"],
    "Urban Intersection": ["ind"],
}
PAPER_ENV = {"Urban Freeway": 11704, "High-Speed Highway": 11660,
             "Roundabout": 2753, "Urban Intersection": 4684}
PAPER_BEHAVIOR = {"Brake": 632754, "Acceleration": 501348, "Change_lane": 375929}
PAPER_CLASS = {"Pedestrian": 2318, "Bicycle": 1796, "Truck/Bus": 10668,
               "Van": 413, "Motorcycle": 85, "Trailer": 189}


def task3_environment(t1):
    res = {}
    for env, srcs in ENV_MAP.items():
        res[env] = sum(len(t1["src_union"][s]) for s in srcs)
    total = sum(res.values())
    return res, total


def task3_behavior():
    """Per-frame behavior label counts over highd / expresswayA / freewayB.

    Returns aggregate counts plus a per-source breakdown. The lane-change
    label semantics are NOT consistent across sources: highD uses yaw flags
    (~7% of frames) and expresswayA's turn labels are sparse (~0.2%), but
    freewayB's turn labels fire on ~20% of frames (a looser lateral-speed
    threshold), inflating the aggregate. We therefore also report a
    "consistent" lane-change figure that excludes freewayB.
    """
    folders = {
        "highd": (["acc_high"], ["brake_high"], ["yaw_left", "yaw_right"]),
        "expresswayA": (["acc_label"], ["brake_label"], ["left_turn_label", "right_turn_label"]),
        "freewayB": (["acc_label"], ["brake_label"], ["left_turn_label", "right_turn_label"]),
    }
    per_src = {}
    for s, (acc_k, brk_k, lane_k) in folders.items():
        b = a = l = 0
        for sub in (f"{s}_high_risk", f"{s}_normal_risk"):
            for p in glob.glob(os.path.join(HV, s, sub, "*.json")):
                try:
                    d = load_json(p)
                except Exception:
                    continue
                for v in iter_vehicles(d):
                    if not isinstance(v, dict):
                        continue
                    if any(bool(v.get(k)) for k in acc_k):
                        a += 1
                    if any(bool(v.get(k)) for k in brk_k):
                        b += 1
                    if any(bool(v.get(k)) for k in lane_k):
                        l += 1
        per_src[s] = {"Brake": b, "Acceleration": a, "Change_lane": l}

    brake = sum(v["Brake"] for v in per_src.values())
    accel = sum(v["Acceleration"] for v in per_src.values())
    lane = sum(v["Change_lane"] for v in per_src.values())
    lane_consistent = per_src["highd"]["Change_lane"] + per_src["expresswayA"]["Change_lane"]
    return {
        "Brake": brake, "Acceleration": accel, "Change_lane": lane,
        "Change_lane_consistent": lane_consistent,
        "per_src": per_src,
    }


def task3_agents():
    """Unique-agent class distribution over all HV event vehicles.

    Agent identity = (source, recording/track, vehicle_id). For each agent we
    take the majority (most frequent non-None) class bucket observed.
    expresswayA / freewayB carry no class -> bucket 'Unlabeled (eA/fB)'.
    """
    agent_class = {}  # key -> Counter of buckets
    agent_unlabeled = set()
    for s in HV_SOURCES:
        for sub in (f"{s}_high_risk", f"{s}_normal_risk"):
            folder = os.path.join(HV, s, sub)
            for p in glob.glob(os.path.join(folder, "*.json")):
                fname = os.path.basename(p)
                # recording/track key from filename to disambiguate ids across files
                base = fname[:-5]
                parts = base.split("_")
                rec_key = None
                try:
                    if s == "highd":
                        rec_key = parts[1]
                    elif s in ("expresswayA", "freewayB"):
                        rec_key = parts[parts.index("track") + 1]
                    elif s in ("ind", "round"):
                        rec_key = parts[parts.index("recording") + 1]
                    elif s == "ngsim":
                        rec_key = "0"
                except (ValueError, IndexError):
                    rec_key = base
                try:
                    d = load_json(p)
                except Exception:
                    continue
                for v in iter_vehicles(d):
                    if not isinstance(v, dict):
                        continue
                    vid = vehicle_id(s, v)
                    if vid is None:
                        continue
                    key = (s, rec_key, vid)
                    b = vehicle_class_bucket(s, v)
                    if b is None:
                        if s in ("expresswayA", "freewayB"):
                            agent_unlabeled.add(key)
                        continue
                    agent_class.setdefault(key, Counter())[b] += 1

    # resolve majority bucket per labeled agent
    bucket_counts = Counter()
    for key, ctr in agent_class.items():
        bucket = ctr.most_common(1)[0][0]
        bucket_counts[bucket] += 1

    n_labeled = sum(bucket_counts.values())
    n_unlabeled = len(agent_unlabeled - set(agent_class.keys()))
    total_agents = n_labeled + n_unlabeled

    # VRU per paper = non-Car among *labeled* agents (Truck/Bus, Van, Trailer,
    # Motorcycle, Pedestrian, Bicycle). Car = Car bucket + unlabeled eA/fB (highway).
    non_car = sum(c for b, c in bucket_counts.items() if b != "Car")
    car = bucket_counts.get("Car", 0) + n_unlabeled
    # strict VRU = pedestrian + bicycle + motorcycle
    strict_vru = sum(bucket_counts.get(b, 0) for b in ("Pedestrian", "Bicycle", "Motorcycle"))

    return {
        "bucket_counts": dict(bucket_counts),
        "n_labeled": n_labeled,
        "n_unlabeled": n_unlabeled,
        "total_agents": total_agents,
        "non_car": non_car,
        "car": car,
        "strict_vru": strict_vru,
    }


# ===========================================================================
# TASK 4 — Figure 7 + joint distribution (sampled)
# ===========================================================================
# Task 4 is scoped entirely to HighD so that K-Risk (the curated high-risk
# event slices) and Original (the full raw trajectories) are compared on the
# SAME source, the SAME variables (ttc, speed) and the SAME filters. Both
# sides are computed in FULL (no sampling).
TTC_SRCS = ["highd"]                                   # HighD only for Task 4
SPEED_SRCS = ["highd"]

TTC_BINS = np.arange(0, 45.0 + 1e-9, 1.0)              # 0..45 s, 1 s bins
SPEED_BINS = np.arange(0, 55.0 + 1e-9, 1.0)            # 0..55 m/s, 1 m/s bins
JOINT_TTC_BINS = np.arange(0, 45.0 + 1e-9, 2.5)
JOINT_SPEED_BINS = np.arange(0, 55.0 + 1e-9, 2.5)

TTC_LO, TTC_HI = 0.0, 45.0
SPEED_LO, SPEED_HI = 0.0, 55.0


def collect_krisk_ego_values():
    """K-Risk side: extract ego per-frame (ttc, speed) from HighD event jsons.

    Full scan of every highd_high_risk / highd_normal_risk json (no sampling).
    Returns {'highd': {'ttc':[...], 'speed':[...], 'pairs':[(ttc,speed)...]}}.
    """
    s = "highd"
    out = {s: {"ttc": [], "speed": [], "pairs": []}}
    for sub in (f"{s}_high_risk", f"{s}_normal_risk"):
        folder = os.path.join(HV, s, sub)
        for p in glob.glob(os.path.join(folder, "*.json")):
            fname = os.path.basename(p)
            ego = ego_id_from_filename(s, fname)
            try:
                d = load_json(p)
            except Exception:
                continue
            for vehicles in iter_frames(d):
                rec = None
                if ego is not None:
                    for v in vehicles:
                        if not isinstance(v, dict):
                            continue
                        if vehicle_id(s, v) == ego[1]:
                            rec = v
                            break
                if rec is None and vehicles:
                    rec = vehicles[0]  # fallback: first vehicle in frame
                if not isinstance(rec, dict):
                    continue
                t = vehicle_ttc(s, rec)
                sp = vehicle_speed(s, rec)
                t_ok = t is not None and TTC_LO < t <= TTC_HI and not math.isinf(t)
                sp_ok = sp is not None and SPEED_LO <= sp <= SPEED_HI
                if t_ok:
                    out[s]["ttc"].append(t)
                if sp_ok:
                    out[s]["speed"].append(sp)
                if t_ok and sp_ok:
                    out[s]["pairs"].append((t, sp))
    return out


def collect_original_full():
    """Original side: full read of every HighD raw trajectory CSV.

    Reads only the `ttc` and `speed` columns from all 61 highd_*.csv files
    (no sampling). HighD's `ttc` column is the only fully continuous TTC in
    trajectory_data, and `speed` is in m/s. Returns
    {'ttc':[...], 'speed':[...], 'pairs':[(ttc,speed)...]}.
    """
    import pandas as pd
    out = {"ttc": [], "speed": [], "pairs": []}
    folder = os.path.join(TRAJ_HV, "highd")
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    for f in files:
        try:
            head = pd.read_csv(f, nrows=0)
        except Exception:
            continue
        cols = set(head.columns)
        ttc_col = "ttc" if "ttc" in cols else ("TTC" if "TTC" in cols else None)
        speed_col = "speed" if "speed" in cols else None
        usecols = [c for c in (ttc_col, speed_col) if c]
        if not usecols:
            continue
        try:
            df = pd.read_csv(f, usecols=usecols)
        except Exception:
            continue
        tt = pd.to_numeric(df[ttc_col], errors="coerce") if ttc_col else None
        sp = pd.to_numeric(df[speed_col], errors="coerce") if speed_col else None
        if tt is not None:
            vals = tt[(tt > TTC_LO) & (tt <= TTC_HI) & np.isfinite(tt)]
            out["ttc"].extend(vals.tolist())
        if sp is not None:
            vals = sp[(sp >= SPEED_LO) & (sp <= SPEED_HI) & np.isfinite(sp)]
            out["speed"].extend(vals.tolist())
        if tt is not None and sp is not None:
            m = ((tt > TTC_LO) & (tt <= TTC_HI) & np.isfinite(tt)
                 & (sp >= SPEED_LO) & (sp <= SPEED_HI) & np.isfinite(sp))
            out["pairs"].extend(list(zip(tt[m].tolist(), sp[m].tolist())))
    return out


def collect_speed_by_dataset():
    """Task 5: ego per-frame speed for each of the 6 HV sources (full scan).

    Same granularity/filter as Task 4 (ego vehicle, per frame, speed in
    [0,55] m/s), but kept per-source so the violin/ridgeline of speed by
    dataset (paper Fig.7 right) can be reproduced. Returns
    {source: numpy_array_of_speeds}.
    """
    out = {}
    for s in HV_SOURCES:
        vals = []
        for sub in (f"{s}_high_risk", f"{s}_normal_risk"):
            folder = os.path.join(HV, s, sub)
            for p in glob.glob(os.path.join(folder, "*.json")):
                fname = os.path.basename(p)
                ego = ego_id_from_filename(s, fname)
                try:
                    d = load_json(p)
                except Exception:
                    continue
                for vehicles in iter_frames(d):
                    rec = None
                    if ego is not None:
                        for v in vehicles:
                            if not isinstance(v, dict):
                                continue
                            if vehicle_id(s, v) == ego[1]:
                                rec = v
                                break
                    if rec is None and vehicles:
                        rec = vehicles[0]
                    if not isinstance(rec, dict):
                        continue
                    sp = vehicle_speed(s, rec)
                    if sp is not None and SPEED_LO <= sp <= SPEED_HI:
                        vals.append(sp)
        out[s] = np.asarray(vals, dtype=float)
    return out


def write_task5_csvs(speed_by_ds, sample_cap=20000):
    """Write the three Task 5 CSVs (histogram / summary / capped samples)."""
    import csv
    scen_label = {"expresswayA": "ExpresswayA", "freewayB": "FreewayB",
                  "highd": "HighD", "ind": "InD", "ngsim": "NGSIM", "round": "RounD"}

    # 1) histogram per dataset (for ridgeline / density curves)
    with open(OUT_T5_HIST, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "bin_left_mps", "bin_right_mps", "count", "density"])
        for s in HV_SOURCES:
            c, e, dd = hist_density(speed_by_ds[s], SPEED_BINS)
            for i in range(len(c)):
                w.writerow([scen_label[s], f"{e[i]:.1f}", f"{e[i+1]:.1f}",
                            int(c[i]), f"{dd[i]:.6f}"])

    # 2) summary statistics per dataset (for box/violin whiskers)
    stats = {}
    with open(OUT_T5_SUMMARY, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "n", "mean", "std", "min",
                    "p5", "p25", "median", "p75", "p95", "max"])
        for s in HV_SOURCES:
            a = speed_by_ds[s]
            if a.size:
                row = [scen_label[s], int(a.size),
                       f"{a.mean():.3f}", f"{a.std():.3f}", f"{a.min():.3f}",
                       f"{np.percentile(a,5):.3f}", f"{np.percentile(a,25):.3f}",
                       f"{np.median(a):.3f}", f"{np.percentile(a,75):.3f}",
                       f"{np.percentile(a,95):.3f}", f"{a.max():.3f}"]
            else:
                row = [scen_label[s], 0] + ["nan"] * 9
            stats[s] = row
            w.writerow(row)

    # 3) capped random samples per dataset (for seaborn violin/KDE directly)
    rng = np.random.default_rng(42)
    with open(OUT_T5_SAMPLES, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "speed_mps"])
        for s in HV_SOURCES:
            a = speed_by_ds[s]
            if a.size > sample_cap:
                idx = rng.choice(a.size, size=sample_cap, replace=False)
                a = a[idx]
            for v in a:
                w.writerow([scen_label[s], f"{v:.3f}"])

    return {s: {"n": int(speed_by_ds[s].size)} for s in HV_SOURCES}, scen_label


def summary_stats(arr):
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return dict(n=0, mean=float("nan"), median=float("nan"),
                    p25=float("nan"), p75=float("nan"), min=float("nan"), max=float("nan"))
    return dict(
        n=int(a.size), mean=float(a.mean()), median=float(np.median(a)),
        p25=float(np.percentile(a, 25)), p75=float(np.percentile(a, 75)),
        min=float(a.min()), max=float(a.max()),
    )


def hist_density(values, bins):
    counts, edges = np.histogram(np.asarray(values, dtype=float), bins=bins)
    widths = np.diff(edges)
    total = counts.sum()
    density = counts / (total * widths) if total > 0 else np.zeros_like(counts, dtype=float)
    return counts, edges, density


# ---------------------------------------------------------------------------
def write_csvs(krisk, original):
    import csv

    # 1) TTC distribution: K-Risk vs Original
    kr_ttc = [t for s in TTC_SRCS for t in krisk[s]["ttc"]]
    or_ttc = original["ttc"]
    kc, kedges, kd = hist_density(kr_ttc, TTC_BINS)
    oc, oedges, od = hist_density(or_ttc, TTC_BINS)
    with open(OUT_TTC, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bin_left_s", "bin_right_s",
                    "krisk_count", "krisk_density",
                    "original_count", "original_density"])
        for i in range(len(kc)):
            w.writerow([f"{kedges[i]:.1f}", f"{kedges[i+1]:.1f}",
                        int(kc[i]), f"{kd[i]:.6f}",
                        int(oc[i]), f"{od[i]:.6f}"])

    # 2) Speed distribution: K-Risk vs Original (HighD)
    kr_speed = [v for s in SPEED_SRCS for v in krisk[s]["speed"]]
    or_speed = original["speed"]
    ksc, ksedges, ksd = hist_density(kr_speed, SPEED_BINS)
    osc, osedges, osd = hist_density(or_speed, SPEED_BINS)
    with open(OUT_SPEED, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bin_left_mps", "bin_right_mps",
                    "krisk_count", "krisk_density",
                    "original_count", "original_density"])
        for i in range(len(ksc)):
            w.writerow([f"{ksedges[i]:.1f}", f"{ksedges[i+1]:.1f}",
                        int(ksc[i]), f"{ksd[i]:.6f}",
                        int(osc[i]), f"{osd[i]:.6f}"])

    # 3) TTC x Speed joint 2D histogram (K-Risk and Original), HighD only
    def joint(pairs):
        if not pairs:
            return np.zeros((len(JOINT_TTC_BINS) - 1, len(JOINT_SPEED_BINS) - 1))
        arr = np.asarray(pairs, dtype=float)
        H, _, _ = np.histogram2d(arr[:, 0], arr[:, 1],
                                 bins=[JOINT_TTC_BINS, JOINT_SPEED_BINS])
        return H
    kr_pairs = [pr for s in TTC_SRCS for pr in krisk[s]["pairs"]]
    Hk = joint(kr_pairs)
    Ho = joint(original["pairs"])
    nk = Hk.sum()
    no = Ho.sum()
    with open(OUT_JOINT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "ttc_left_s", "ttc_right_s",
                    "speed_left_mps", "speed_right_mps", "count", "joint_density"])
        for label, H, ntot in (("K-Risk", Hk, nk), ("Original", Ho, no)):
            for i in range(H.shape[0]):
                for j in range(H.shape[1]):
                    cnt = H[i, j]
                    area = (JOINT_TTC_BINS[i+1]-JOINT_TTC_BINS[i]) * \
                           (JOINT_SPEED_BINS[j+1]-JOINT_SPEED_BINS[j])
                    dens = cnt / (ntot * area) if ntot > 0 else 0.0
                    w.writerow([label, f"{JOINT_TTC_BINS[i]:.1f}", f"{JOINT_TTC_BINS[i+1]:.1f}",
                                f"{JOINT_SPEED_BINS[j]:.1f}", f"{JOINT_SPEED_BINS[j+1]:.1f}",
                                int(cnt), f"{dens:.6f}"])

    return {
        "kr_ttc": summary_stats(kr_ttc),
        "or_ttc": summary_stats(or_ttc),
        "kr_speed": summary_stats(kr_speed),
        "or_speed": summary_stats(or_speed),
        "kr_pairs_n": int(nk),
        "or_pairs_n": int(no),
    }


# ---------------------------------------------------------------------------
def fmt_pct(part, whole):
    return f"{100.0*part/whole:.1f}%" if whole else "—"


def write_md(t1, t2, env, env_total, beh, agents, t4, speed_by_ds=None, t5_label=None):
    L = []
    A = L.append
    A("# K-Risk 数据集统计结果\n")
    A("> 数据来源：`KHHRED/event_annotations`、`KHHRED/trajectory_data`。"
      "去重单位为文件名 basename；`extreme/` 样本来自 expresswayA/freewayB，"
      "已合并入对应源、不单独成项、不重复计数。图6/图7 为基于**当前数据**的重算，"
      "与论文 `K_Risk-1025.pdf` 历史数字存在偏差属正常。\n")

    # ---- Task 1 ----
    A("## 1. 数据源分布（Distribution of Data Sources）\n")
    A("HV 各源 = `_high_risk ∪ _normal_risk` 按 basename 去重；expresswayA/freewayB "
      "再并入对应 extreme 文件。AV 按 `(子数据集, 文件名)` 去重。\n")
    A("| Data Source | # JSON files |")
    A("|---|---:|")
    name_disp = {"expresswayA": "ExpresswayA", "freewayB": "FreewayB", "highd": "HighD",
                 "ind": "InD", "ngsim": "NGSIM", "round": "RounD", "AV": "AV"}
    for s, c in t1["rows"]:
        A(f"| {name_disp.get(s, s)} | {c} |")
    A(f"| **合计 (Total)** | **{t1['total']}** |\n")
    A(f"- AV 明细：全量文件 {t1['av_all']} / 去重后 {t1['av_dedup']} "
      f"(其中 23 处同名为同数据集内 acc/brake 切片、2 处为跨数据集巧合同名)。"
      f"七项汇总采用去重值 {t1['av_dedup']}。\n")

    # ---- Task 2 ----
    A("## 2. HV 风险层级分布（Risk Level Distribution）\n")
    A("三层互斥，优先级 Extreme > High > Moderate：Extreme = 去重后 `extreme/`；"
      "High = `_high_risk` 去除 extreme 成员；Moderate = `_normal_risk` 去除 extreme "
      "及 high 成员（NGSIM 有 92 个文件同时位于 high/normal 文件夹，按 High 计、"
      "不在 Moderate 重复计数）。\n")
    A("| Source | Moderate Risk | High Risk | (extreme 已剥离) |")
    A("|---|---:|---:|---|")
    for s, mod, high in t2["per_src"]:
        note = ""
        if s == "expresswayA":
            note = f"extreme={t2['extreme_eA']}"
        elif s == "freewayB":
            note = f"extreme={t2['extreme_fB']}"
        A(f"| {name_disp[s]} | {mod} | {high} | {note} |")
    A(f"| **小计** | **{t2['moderate']}** | **{t2['high']}** | |\n")
    A("| Risk Level | Count |")
    A("|---|---:|")
    A(f"| Moderate Risk (`_normal_risk` − extreme) | {t2['moderate']} |")
    A(f"| High Risk (`_high_risk` − extreme) | {t2['high']} |")
    A(f"| Extreme Risk (`extreme/`) | {t2['extreme']} |")
    A(f"| **合计 (互斥总样本)** | **{t2['total']}** |\n")

    # ---- Task 3 ----
    A("## 3. 图6 复现（Figure 6, 基于 event_annotations/HV）\n")

    A("### 3.1 驾驶环境分布（Distribution by Driving Environment）\n")
    A("映射：Urban Freeway = ExpresswayA+FreewayB；High-Speed Highway = HighD+NGSIM；"
      "Roundabout = RounD；Urban Intersection = InD。\n")
    A("| Environment | Count | 占比 | 论文原值 |")
    A("|---|---:|---:|---:|")
    for env_name in ["High-Speed Highway", "Urban Freeway", "Urban Intersection", "Roundabout"]:
        A(f"| {env_name} | {env[env_name]} | {fmt_pct(env[env_name], env_total)} | "
          f"{PAPER_ENV[env_name]} |")
    A(f"| **Total Cases** | **{env_total}** | 100% | 30801 |\n")

    A("### 3.2 驾驶行为分布（Driving Behavior Distribution）\n")
    A("口径：仅 HighD / ExpresswayA / FreewayB 三源逐帧累计行为标签为 True 的帧数"
      "（其余源无行为标签字段，不计入）；Change_lane = 左转+右转标签合并。\n")
    beh_total = beh["Brake"] + beh["Acceleration"] + beh["Change_lane"]
    A("| Behavior | Frame count | 占比 | 论文原值 |")
    A("|---|---:|---:|---:|")
    for b in ["Brake", "Acceleration", "Change_lane"]:
        A(f"| {b} | {beh[b]} | {fmt_pct(beh[b], beh_total)} | {PAPER_BEHAVIOR[b]} |")
    A(f"| **Total Events** | **{beh_total}** | 100% | 1510031 |\n")
    A("> ⚠️ **Change_lane 口径警示**：Brake / Acceleration 与论文高度吻合（误差 <2%），"
      "印证逐帧标签计数口径正确。但三源的「变道/转向」标签语义不一致——"
      "FreewayB 的 `left/right_turn_label` 约 20% 帧触发（横向速度超均值的松阈值），"
      "使 Change_lane 聚合值虚高。**剔除 FreewayB 的一致口径 Change_lane = "
      f"{beh['Change_lane_consistent']}**（仅 HighD 偏航 + ExpresswayA 转向）。\n")
    ps = beh["per_src"]
    A("各源行为帧数明细：\n")
    A("| Source | Brake | Acceleration | Change_lane | 变道字段 |")
    A("|---|---:|---:|---:|---|")
    field_note = {"highd": "yaw_left+yaw_right", "expresswayA": "left+right_turn_label",
                  "freewayB": "left+right_turn_label (阈值偏松)"}
    for s in ["highd", "expresswayA", "freewayB"]:
        A(f"| {name_disp[s]} | {ps[s]['Brake']} | {ps[s]['Acceleration']} | "
          f"{ps[s]['Change_lane']} | {field_note[s]} |")
    A("")

    A("### 3.3 脆弱道路使用者占比（VRU Percentage）\n")
    A("Unique agent = `(源, recording/track, vehicle_id)` 去重；按论文口径 "
      "non-Car 计为 VRU（含 Truck/Bus、Van、Trailer 等重型/特殊车），"
      "ExpresswayA/FreewayB 无 class 字段、按高速场景归为 Car。\n")
    A("| Category | Count | 占比 |")
    A("|---|---:|---:|")
    A(f"| VRU (non-Car, 论文口径) | {agents['non_car']} | "
      f"{fmt_pct(agents['non_car'], agents['total_agents'])} |")
    A(f"| Car | {agents['car']} | {fmt_pct(agents['car'], agents['total_agents'])} |")
    A(f"| **Total Users** | **{agents['total_agents']}** | 100% |\n")
    A(f"- 对照：**严格 VRU**（Pedestrian+Bicycle+Motorcycle）= {agents['strict_vru']} "
      f"({fmt_pct(agents['strict_vru'], agents['total_agents'])})。\n")
    A(f"- 其中无 class 标注的 ExpresswayA/FreewayB agent 数 = {agents['n_unlabeled']}"
      f"（已并入 Car）。\n")

    A("### 3.4 车辆类别分布（Class Distribution）\n")
    A("统一类别桶；NGSIM 1→Motorcycle、2→Car、3→Truck/Bus；HighD Truck→Truck/Bus。"
      "每个 agent 取出现最多的类别。\n")
    A("| Class | Count |")
    A("|---|---:|")
    order = ["Car", "Pedestrian", "Bicycle", "Truck/Bus", "Van", "Motorcycle", "Trailer"]
    bc = agents["bucket_counts"]
    for c in order:
        if c in bc:
            A(f"| {c} | {bc[c]} |")
    if agents["n_unlabeled"]:
        A(f"| Unlabeled (ExpresswayA/FreewayB) | {agents['n_unlabeled']} |")
    A(f"| **Total labeled agents** | **{agents['n_labeled']}** |\n")

    # ---- Task 4 ----
    A("## 4. 图7 复现 + TTC/Speed 联合分布（HighD，全量；明细见 CSV）\n")
    A("**口径**：Task 4 完全限定 **HighD**，使 K-Risk 与 Original Data 在"
      "**同一数据源、同一变量（ttc / speed）、同一过滤条件**下可比。"
      "K-Risk = `event_annotations/HV/highd` 的 ego 车逐帧值（高风险事件切片）；"
      "Original = `trajectory_data/HV/highd` 全部 61 个轨迹 CSV 的逐帧值（原始全量）。"
      "**两端均为全量统计，无抽样。** HighD 的 `ttc` 是唯一全量连续的 TTC 列，"
      "`speed` 单位 m/s。过滤无效值（ttc∈(0,45]、speed∈[0,55]）。\n")
    A("> 说明：上一版速度对比把 K-Risk 设为 4 源、Original 仅 HighD，口径不一致导致"
      "Original 反而比 K-Risk 更集中；本版两端统一为 HighD 全量后，曲线方可正确对比"
      "（Original 覆盖全部常规行驶、分布更宽；K-Risk 为高风险子集，向低 TTC 富集）。\n")

    A("### 4.1 TTC 分布（左图）— 汇总统计\n")
    A("| Dataset | n | mean | median | p25 | p75 | min | max |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|")
    for label, st in (("K-Risk", t4["kr_ttc"]), ("Original", t4["or_ttc"])):
        A(f"| {label} | {st['n']} | {st['mean']:.2f} | {st['median']:.2f} | "
          f"{st['p25']:.2f} | {st['p75']:.2f} | {st['min']:.2f} | {st['max']:.2f} |")
    A("\n→ 明细：`task4_ttc_distribution.csv`（K-Risk vs Original，1s 分箱）\n")

    A("### 4.2 速度分布（右图）— 汇总统计\n")
    A("| Dataset | n | mean | median | p25 | p75 | min | max |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|")
    for label, st in (("K-Risk", t4["kr_speed"]), ("Original", t4["or_speed"])):
        A(f"| {label} | {st['n']} | {st['mean']:.2f} | {st['median']:.2f} | "
          f"{st['p25']:.2f} | {st['p75']:.2f} | {st['min']:.2f} | {st['max']:.2f} |")
    A("\n→ 明细：`task4_speed_distribution.csv`（K-Risk vs Original，1m/s 分箱）\n")

    A("### 4.3 TTC × Speed 联合分布 — 规模\n")
    A(f"- K-Risk 同帧 (TTC, speed) 样本对：{t4['kr_pairs_n']}（HighD 事件，全量）\n")
    A(f"- Original 同帧 (TTC, speed) 样本对：{t4['or_pairs_n']}（HighD 轨迹，全量）\n")
    A("→ 2D 直方图明细：`task4_ttc_speed_joint_distribution.csv`"
      "（含 K-Risk 与 Original 两组，2.5s × 2.5m/s 分箱）\n")

    # ---- Task 5 ----
    if speed_by_ds is not None:
        A("## 5. 各数据集速度分布（Speed Distribution by Dataset，event_annotations/HV）\n")
        A("数据：`event_annotations/HV` 全部 6 个源的 ego 车·逐帧 speed，**全量统计、无抽样**，"
          "过滤 speed∈[0,55] m/s（速度口径与 Task 4 一致）。用于复现论文图7右图"
          "（各场景速度的 violin/ridgeline 对比）。NGSIM 用 `v_Vel`、InD/RounD 由"
          "速度分量合成 √(vx²+vy²)，其余源用 `speed` 字段。\n")
        A("### 5.1 各数据集速度汇总统计\n")
        A("| Dataset | n | mean | std | p5 | p25 | median | p75 | p95 | max |")
        A("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for s in HV_SOURCES:
            a = speed_by_ds[s]
            lab = t5_label[s] if t5_label else s
            if a.size:
                A(f"| {lab} | {a.size} | {a.mean():.2f} | {a.std():.2f} | "
                  f"{np.percentile(a,5):.2f} | {np.percentile(a,25):.2f} | "
                  f"{np.median(a):.2f} | {np.percentile(a,75):.2f} | "
                  f"{np.percentile(a,95):.2f} | {a.max():.2f} |")
            else:
                A(f"| {lab} | 0 | — | — | — | — | — | — | — | — |")
        A("")
        A("### 5.2 绘图用 CSV\n")
        A("| 文件 | 用途 |")
        A("|---|---|")
        A("| `task5_speed_distribution_by_dataset_hist.csv` | 各数据集速度直方图（1 m/s 分箱，"
          "含 count 与 density）——画 ridgeline / 密度曲线 |")
        A("| `task5_speed_distribution_by_dataset_summary.csv` | 各数据集 n/mean/std/分位数"
          "（p5–p95）——画 box / violin 须线 |")
        A("| `task5_speed_distribution_by_dataset_samples.csv` | 各数据集随机抽样原始速度值"
          "（每源最多 20000 条）——供 seaborn 直接画 violin / KDE |")
        A("")

    with open(OUT_MD, "w") as f:
        f.write("\n".join(L))


# ---------------------------------------------------------------------------
def main():
    print("[1/7] building dedup sets ...")
    sets = build_sets()

    print("[2/7] Task 1: data source distribution ...")
    t1 = task1(sets)

    print("[3/7] Task 2: risk level distribution ...")
    t2 = task2(sets)

    print("[4/7] Task 3: figure 6 panels (environment / behavior / agents) ...")
    env, env_total = task3_environment(t1)
    beh = task3_behavior()
    agents = task3_agents()

    print("[5/7] Task 4: HighD K-Risk (events) + Original (full trajectories) ...")
    krisk = collect_krisk_ego_values()
    print("       -> reading full HighD raw trajectories (61 CSVs) ...")
    original = collect_original_full()
    t4 = write_csvs(krisk, original)

    # remove the superseded per-scenario speed CSV so outputs stay consistent
    if os.path.exists(OUT_SPEED_LEGACY):
        try:
            os.remove(OUT_SPEED_LEGACY)
            print(f"       -> removed legacy file {os.path.basename(OUT_SPEED_LEGACY)}")
        except OSError:
            pass

    print("[6/7] Task 5: per-dataset speed distribution (6 HV sources) ...")
    speed_by_ds = collect_speed_by_dataset()
    t5_n, t5_label = write_task5_csvs(speed_by_ds)

    print("[7/7] writing markdown ...")
    write_md(t1, t2, env, env_total, beh, agents, t4, speed_by_ds, t5_label)

    # console cross-checks
    hv_dedup = sum(len(t1["src_union"][s]) for s in HV_SOURCES)
    print("\n=== cross-checks ===")
    print(f"Task1 HV six-source dedup sum = {hv_dedup}")
    print(f"Task2 mutually-exclusive total = {t2['total']}  (should equal HV dedup sum)")
    print(f"  -> match: {hv_dedup == t2['total']}")
    print(f"RounD = {len(t1['src_union']['round'])} (paper 2753), "
          f"InD = {len(t1['src_union']['ind'])} (paper 4684)")
    print(f"Total (7 sources, incl AV) = {t1['total']}")
    print(f"Task4 HighD TTC  : K-Risk n={t4['kr_ttc']['n']}, Original n={t4['or_ttc']['n']}")
    print(f"Task4 HighD speed: K-Risk n={t4['kr_speed']['n']}, Original n={t4['or_speed']['n']}")
    print(f"Task4 HighD pairs: K-Risk={t4['kr_pairs_n']}, Original={t4['or_pairs_n']}")
    print("Task5 per-dataset speed n:", {t5_label[s]: t5_n[s]["n"] for s in HV_SOURCES})
    print("\nOutputs written to:")
    for p in (OUT_MD, OUT_TTC, OUT_SPEED, OUT_JOINT,
              OUT_T5_HIST, OUT_T5_SUMMARY, OUT_T5_SAMPLES):
        print("  -", os.path.relpath(p, HERE))


if __name__ == "__main__":
    main()
