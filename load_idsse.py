"""Load IDSSE (DFL Bundesliga) matches via floodlight and compute Spearman pitch control.

Reusable helpers:
    load_match(match_id)                 -> dict with events/xy/possession/ballstatus/teamsheets/pitch
    build_event_frame(match)             -> one tidy events DataFrame ("Delete" dropped, coords centred)
    event_to_tracking_coords(x, y, ...)  -> event (0..L, 0..W) -> tracking (centre-origin metres)
    compute_velocities(xy)               -> (vx, vy) arrays in m/s for a floodlight XY object
    pitch_control_at_frame(...)          -> Spearman (2018) pitch control surface for one frame
    validate_event_alignment(match, ev)  -> event location vs. passer's tracked position

Run as a script to print a summary, alignment check and one pitch control surface.

Coordinate systems (verified on J03WMX, see validate_event_alignment):
    tracking : metres, origin at pitch centre, x in [-52.5, 52.5], y in [-34, 34]
    events   : metres, origin at a corner, at_x in [0, 105], at_y in [0, 68]
    A pure shift (x - L/2, y - W/2) aligns them in both halves; no axis flips.
Event time -> tracking frame: frame = round(gameclock * framerate), per period.
"""
import warnings

import numpy as np
import pandas as pd

from floodlight.io.datasets import IDSSEDataset

warnings.filterwarnings("ignore", message="The 'gameclock' column")

PERIODS = ("firstHalf", "secondHalf")
DEFAULT_MATCH = "J03WMX"  # 1. FC Koeln (Home) vs FC Bayern Muenchen (Away)
MATCHES = {
    "J03WMX": "1. FC Köln vs. FC Bayern München",
    "J03WN1": "VfL Bochum 1848 vs. Bayer 04 Leverkusen",
    "J03WPY": "Fortuna Düsseldorf vs. 1. FC Nürnberg",
    "J03WOH": "Fortuna Düsseldorf vs. SSV Jahn Regensburg",
    "J03WQQ": "Fortuna Düsseldorf vs. FC St. Pauli",
    "J03WOY": "Fortuna Düsseldorf vs. F.C. Hansa Rostock",
    "J03WR9": "Fortuna Düsseldorf vs. 1. FC Kaiserslautern",
}


def load_match(match_id=DEFAULT_MATCH):
    """Download (first call only) and load one match. Data goes to floodlight's .data folder."""
    dataset = IDSSEDataset(match_id=match_id)
    events, xy, possession, ballstatus, teamsheets, pitch = dataset.get(match_id)
    return dict(match_id=match_id, events=events, xy=xy, possession=possession,
                ballstatus=ballstatus, teamsheets=teamsheets, pitch=pitch)


def event_to_tracking_coords(x, y, pitch):
    """Convert event coords (corner origin) to tracking coords (centre origin)."""
    return np.asarray(x, dtype=float) - pitch.length / 2, np.asarray(y, dtype=float) - pitch.width / 2


def build_event_frame(match, drop_delete=True):
    """Merge all periods/teams into one DataFrame, drop 'Delete' rows, centre coordinates.

    Adds: period, side (Home/Away), xID (column index in the XY object), frame,
    and x/y/to_x_c/to_y_c in tracking coordinates (NaN where the event has no location).
    """
    pitch, ts = match["pitch"], match["teamsheets"]
    fps = match["xy"]["firstHalf"]["Ball"].framerate
    pid_map = {r.pID: (side, int(r.xID)) for side in ("Home", "Away") for _, r in ts[side].teamsheet.iterrows()}
    tid_side = {ts[side].teamsheet.tID.iloc[0]: side for side in ("Home", "Away")}

    parts = []
    for period in PERIODS:
        for side in ("Home", "Away"):
            df = match["events"][period][side].events.copy()
            df["period"] = period
            parts.append(df)
    ev = pd.concat(parts, ignore_index=True)
    if drop_delete:
        ev = ev[ev.eID != "Delete"]
    ev = ev.assign(_p=ev.period.map(PERIODS.index)).sort_values(["_p", "gameclock"]).drop(columns="_p")
    ev["side"] = ev.tID.map(tid_side)
    ev["xID"] = ev.pID.map(lambda p: pid_map.get(p, (None, np.nan))[1])
    ev["frame"] = np.round(ev.gameclock * fps).astype(int)
    ev["x"], ev["y"] = event_to_tracking_coords(ev.at_x, ev.at_y, pitch)
    ev["to_x_c"], ev["to_y_c"] = event_to_tracking_coords(ev.to_x, ev.to_y, pitch)
    return ev.reset_index(drop=True)


def validate_event_alignment(match, ev, event_types=("Play_Pass", "Play_Cross"), n_examples=5):
    """Compare event location with the tracked position of the acting player at the event frame.

    Returns (summary dict, examples DataFrame). DFL event locations are hand-annotated,
    so expect metre-level noise (median ~3 m), not centimetre agreement.
    """
    rows = []
    for _, r in ev[ev.eID.isin(event_types) & ev.x.notna() & ev.xID.notna()].iterrows():
        arr = match["xy"][r.period][r.side].xy
        f = int(np.clip(r.frame, 0, len(arr) - 1))
        px, py = arr[f, 2 * int(r.xID)], arr[f, 2 * int(r.xID) + 1]
        bx, by = match["xy"][r.period]["Ball"].xy[f]
        rows.append(dict(period=r.period, gameclock=r.gameclock, side=r.side, eID=r.eID, ev_x=r.x, ev_y=r.y,
                         player_x=px, player_y=py, ball_x=bx, ball_y=by,
                         dist_player=np.hypot(r.x - px, r.y - py), dist_ball=np.hypot(r.x - bx, r.y - by)))
    res = pd.DataFrame(rows)
    summary = {"n": len(res),
               "median_dist_player_m": res.dist_player.median(),
               "share_within_5m_of_player": (res.dist_player < 5).mean(),
               "median_dist_ball_m": res.dist_ball.median()}
    return summary, res.sample(n_examples, random_state=0).round(2)


def compute_velocities(xy, max_speed=12.0):
    """Velocity (m/s) per player from a floodlight XY object via central differences.

    Returns (vx, vy), each shaped (frames, players). NaN positions give NaN velocities.
    """
    pos = xy.xy
    vx = np.gradient(pos[:, 0::2], axis=0) * xy.framerate
    vy = np.gradient(pos[:, 1::2], axis=0) * xy.framerate
    speed = np.hypot(vx, vy)
    scale = np.where(speed > max_speed, max_speed / np.maximum(speed, 1e-9), 1.0)
    return vx * scale, vy * scale


def default_model_params(time_to_control_veto=3):
    """Spearman (2018) parameters, same defaults as LaurieOnTracking."""
    p = dict(max_player_speed=5.0, reaction_time=0.7, tti_sigma=0.45, kappa_def=1.0, lambda_att=4.3,
             average_ball_speed=15.0, int_dt=0.04, max_int_time=10.0, model_converge_tol=0.01)
    p["lambda_def"] = p["lambda_att"] * p["kappa_def"]
    p["lambda_gk"] = p["lambda_def"] * 3.0
    return p


def _goalkeeper_xids(teamsheet):
    ts = teamsheet.teamsheet
    gk = ts[ts.position == "TW"]
    return set(gk.xID.astype(int))


def pitch_control_at_frame(match, period, frame, attacking_side, ball_start=None, velocities=None,
                           params=None, n_grid_x=50, offsides=True, targets=None):
    """Spearman pitch control for `attacking_side` ('Home'/'Away') at one tracking frame.

    ball_start: (x, y) in tracking coords; defaults to the tracked ball position at `frame`.
    velocities: optional {'Home': (vx, vy), 'Away': (vx, vy)} from compute_velocities() for
                this period (pass it in when evaluating many frames).
    targets:    optional (P, 2) array of tracking-coordinate points. If given, returns just the
                (P,) attacking-team control values at those points instead of the grid.
    Returns (PPCF_attacking, xgrid, ygrid); PPCF has shape (n_grid_y, n_grid_x).
    """
    params = params or default_model_params()
    pitch = match["pitch"]
    L, W = pitch.length, pitch.width
    defending_side = "Away" if attacking_side == "Home" else "Home"
    xy_period = match["xy"][period]
    if velocities is None:
        velocities = {s: compute_velocities(xy_period[s]) for s in ("Home", "Away")}
    if ball_start is None:
        ball_start = xy_period["Ball"].xy[frame]
    ball_start = np.asarray(ball_start, dtype=float)

    def team_state(side):
        pos = xy_period[side].xy[frame].reshape(-1, 2)
        vel = np.stack([velocities[side][0][frame], velocities[side][1][frame]], axis=1)
        vel = np.nan_to_num(vel)
        ok = ~np.isnan(pos).any(axis=1)
        ids = np.flatnonzero(ok)
        return ids, pos[ok], vel[ok]

    att_ids, att_pos, att_vel = team_state(attacking_side)
    def_ids, def_pos, def_vel = team_state(defending_side)

    if offsides and len(def_pos) > 1:
        gk = _goalkeeper_xids(match["teamsheets"][defending_side])
        gk_here = [i for i, xid in enumerate(def_ids) if xid in gk]
        half_sign = np.sign(def_pos[gk_here[0], 0]) if gk_here else np.sign(np.nanmean(def_pos[:, 0]))
        second_deepest = sorted(half_sign * def_pos[:, 0], reverse=True)[1]
        line = max(second_deepest, half_sign * ball_start[0], 0.0) + 0.2
        keep = att_pos[:, 0] * half_sign <= line
        att_ids, att_pos, att_vel = att_ids[keep], att_pos[keep], att_vel[keep]

    # grid
    n_y = int(n_grid_x * W / L)
    dx, dy = L / n_grid_x, W / n_y
    xgrid = np.arange(n_grid_x) * dx - L / 2 + dx / 2
    ygrid = np.arange(n_y) * dy - W / 2 + dy / 2
    if targets is None:
        target = np.stack(np.meshgrid(xgrid, ygrid), axis=-1).reshape(-1, 2)  # (G, 2), row-major in y
    else:
        target = np.asarray(targets, dtype=float).reshape(-1, 2)

    def tti(pos, vel):
        reacted = pos + vel * params["reaction_time"]
        return params["reaction_time"] + np.linalg.norm(target[None] - reacted[:, None], axis=2) / params["max_player_speed"]

    tti_att, tti_def = tti(att_pos, att_vel), tti(def_pos, def_vel)  # (N, G)
    gk_def = _goalkeeper_xids(match["teamsheets"][defending_side])
    lam_def = np.array([params["lambda_gk"] if x in gk_def else params["lambda_def"] for x in def_ids])[:, None]
    gk_att = _goalkeeper_xids(match["teamsheets"][attacking_side])
    lam_att = np.array([params["lambda_gk"] if x in gk_att else params["lambda_att"] for x in att_ids])[:, None]

    ball_time = np.linalg.norm(target - ball_start[None], axis=1) / params["average_ball_speed"]  # (G,)
    dt, coef = params["int_dt"], np.pi / np.sqrt(3.0) / params["tti_sigma"]
    ppcf_att_p, ppcf_def_p = np.zeros_like(tti_att), np.zeros_like(tti_def)  # per-player accumulators
    att_tot, def_tot = np.zeros(len(target)), np.zeros(len(target))
    active = np.ones(len(target), dtype=bool)
    for i in range(1, int(params["max_int_time"] / dt) + 1):
        T = ball_time + (i - 1) * dt
        remaining = 1.0 - att_tot - def_tot
        d_att = remaining * lam_att / (1.0 + np.exp(-coef * (T - tti_att))) * dt * active
        d_def = remaining * lam_def / (1.0 + np.exp(-coef * (T - tti_def))) * dt * active
        ppcf_att_p += d_att
        ppcf_def_p += d_def
        att_tot, def_tot = ppcf_att_p.sum(axis=0), ppcf_def_p.sum(axis=0)
        active = (1.0 - att_tot - def_tot) > params["model_converge_tol"]
        if not active.any():
            break
    if targets is not None:
        return att_tot
    return att_tot.reshape(n_y, n_grid_x), xgrid, ygrid


if __name__ == "__main__":
    match = load_match(DEFAULT_MATCH)
    print(f"Match {match['match_id']}: {MATCHES[match['match_id']]}")
    print("pitch:", match["pitch"].length, "x", match["pitch"].width, match["pitch"].unit)
    for period in PERIODS:
        print(period, "frames:", match["xy"][period]["Ball"].xy.shape[0], "@", match["xy"][period]["Ball"].framerate, "fps")

    ev = build_event_frame(match)
    print(f"\nevents after dropping 'Delete': {len(ev)}  (types: {ev.eID.nunique()})")
    print(ev[["period", "gameclock", "side", "eID", "x", "y", "to_x_c", "to_y_c", "frame"]].head(5).round(2).to_string())

    summary, examples = validate_event_alignment(match, ev)
    print("\nevent vs passer tracking position:", {k: round(v, 3) for k, v in summary.items()})
    print(examples.to_string())

    # one pitch control surface: first Home pass of the second half
    e = ev[(ev.period == "secondHalf") & (ev.side == "Home") & (ev.eID == "Play_Pass")].iloc[0]
    ppcf, xg, yg = pitch_control_at_frame(match, e.period, int(e.frame), "Home", ball_start=(e.x, e.y))
    print(f"\npitch control @ {e.period} frame {int(e.frame)} (Home pass at x={e.x:.1f}, y={e.y:.1f}): "
          f"grid {ppcf.shape}, Home mean control {ppcf.mean():.3f}, range [{ppcf.min():.3f}, {ppcf.max():.3f}]")
