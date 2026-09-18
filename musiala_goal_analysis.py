"""Pitch control and pass options at the key pass before Musiala's 89th-minute winner (Koeln 1-2 Bayern).

Finds the goal event (ShotAtGoal_SuccessfulShot by J. Musiala), takes the assist pass named in its
'Assist' qualifier as the key moment, then:
    * draws the Spearman pitch control surface at that frame with floodlight.vis
    * scores every teammate as a pass option (pitch control x EPV) and writes it to JSON

Outputs: outputs/musiala_goal_pitch_control.png, outputs/musiala_goal_pass_options.json
Run:     .venv-floodlight\\Scripts\\python.exe musiala_goal_analysis.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from floodlight.vis.pitches import plot_football_pitch
from floodlight.vis.positions import plot_positions

import load_idsse as L

ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
EPV_FILE = ROOT / "LaurieOnTracking" / "EPV_grid.csv"

ATT_COLOR, DEF_COLOR, INK, MUTED = "#c8501e", "#2a6fdb", "#1f2328", "#6b7280"


def find_key_moment(match, ev):
    """Return (goal_row, assist_pass_row) for Musiala's goal."""
    names = {r.pID: r.player for s in ("Home", "Away") for _, r in match["teamsheets"][s].teamsheet.iterrows()}
    ev = ev.assign(player=ev.pID.map(names))
    goal = ev[(ev.eID == "ShotAtGoal_SuccessfulShot") & (ev.player == "J. Musiala")].iloc[0]
    assister = goal.qualifier["Assist"]
    prior = ev[(ev.pID == assister) & ev.eID.str.contains("Pass") & (ev.period == goal.period) & (ev.gameclock < goal.gameclock)]
    return goal.to_dict(), prior.iloc[-1].to_dict(), names


def epv_at(pos, epv_grid, attack_dir, pitch):
    """EPV lookup (LaurieOnTracking's grid, same convention as Metrica_EPV.get_EPV_at_location)."""
    x, y = pos
    if abs(x) > pitch.length / 2 or abs(y) > pitch.width / 2:
        return 0.0
    grid = np.fliplr(epv_grid) if attack_dir == -1 else epv_grid
    ny, nx = grid.shape
    ix = int((x + pitch.length / 2 - 1e-4) / (pitch.length / nx))
    iy = int((y + pitch.width / 2 - 1e-4) / (pitch.width / ny))
    return float(grid[iy, ix])


def lane_clearance(p, q, others):
    """Smallest distance from any point in `others` (N,2) to segment p->q."""
    d = q - p
    t = np.clip(((others - p) @ d) / max(d @ d, 1e-9), 0.0, 1.0)
    return float(np.linalg.norm(others - (p + t[:, None] * d), axis=1).min())


def interception_margin(p, q, def_pos, def_vel, params, step_m=1.0):
    """Smallest (defender arrival time - ball arrival time) over points on the lane p->q, in seconds.

    Same movement model as the pitch control (reaction time, then straight run at max speed).
    A negative margin means some defender can reach a point of the lane before the ball does.
    """
    n = max(int(np.linalg.norm(q - p) / step_m), 1)
    pts = p + (q - p) * (np.arange(1, n + 1) / n)[:, None]  # (n, 2), excludes the passer's own position
    ball_t = np.linalg.norm(pts - p, axis=1) / params["average_ball_speed"]
    reacted = def_pos + def_vel * params["reaction_time"]
    def_t = params["reaction_time"] + np.linalg.norm(pts[None] - reacted[:, None], axis=2) / params["max_player_speed"]
    return float((def_t - ball_t[None]).min())


def main():
    OUT.mkdir(exist_ok=True)
    match = L.load_match()
    pitch, ev = match["pitch"], L.build_event_frame(match)
    goal, key, names = find_key_moment(match, ev)
    period, frame, side = key["period"], int(key["frame"]), key["side"]
    opp = "Home" if side == "Away" else "Away"
    xy_p = match["xy"][period]
    vel = {s: L.compute_velocities(xy_p[s]) for s in ("Home", "Away")}
    ts = {s: match["teamsheets"][s].teamsheet for s in ("Home", "Away")}

    def state(s):
        pos = xy_p[s].xy[frame].reshape(-1, 2)
        return pos, ~np.isnan(pos).any(axis=1)

    att_pos, att_ok = state(side)
    opp_pos, opp_ok = state(opp)
    params = L.default_model_params()
    opp_vel = np.nan_to_num(np.stack([vel[opp][0][frame], vel[opp][1][frame]], axis=1))[opp_ok]
    passer_x = int(key["xID"])
    passer_pos = att_pos[passer_x]

    # attack direction from own goalkeeper's side of the pitch
    gk_ids = [i for i in L._goalkeeper_xids(match["teamsheets"][side]) if att_ok[i]]
    attack_dir = -int(np.sign(att_pos[gk_ids[0], 0]))
    # in attack-direction coordinates the opponent's goal is at +L/2; line = 2nd-deepest defender / ball / halfway
    opp_x_sorted = sorted(attack_dir * opp_pos[opp_ok, 0], reverse=True)
    offside_line = max(opp_x_sorted[1], attack_dir * passer_pos[0], 0.0) + 0.2

    epv_grid = np.loadtxt(EPV_FILE, delimiter=",")
    recip_pid = key["qualifier"].get("Recipient")

    # ---- pass options
    ids = [i for i in np.flatnonzero(att_ok) if i != passer_x]
    targets = att_pos[ids]
    pc = L.pitch_control_at_frame(match, period, frame, side, ball_start=passer_pos, velocities=vel, targets=targets)
    pid_of = {int(r.xID): r.pID for _, r in ts[side].iterrows()}
    options = []
    for i, ctrl, tgt in zip(ids, pc, targets):
        epv = epv_at(tgt, epv_grid, attack_dir, pitch)
        clear = lane_clearance(passer_pos, tgt, opp_pos[opp_ok])
        margin = interception_margin(passer_pos, tgt, opp_pos[opp_ok], opp_vel, params)
        row = ts[side][ts[side].xID == i].iloc[0]
        options.append(dict(
            player=row.player, jersey=int(row.jID), position_code=row.position, xID=int(i),
            x=round(float(tgt[0]), 2), y=round(float(tgt[1]), 2),
            distance_m=round(float(np.linalg.norm(tgt - passer_pos)), 1),
            lane_clearance_m=round(clear, 2), interception_margin_s=round(margin, 2), visible=bool(margin > 0),
            offside=bool(attack_dir * tgt[0] > offside_line),
            pitch_control=round(float(ctrl), 3), epv_threat=round(epv, 4),
            expected_value=round(float(ctrl) * epv, 4),
            is_actual_target=bool(pid_of[int(i)] == recip_pid)))
    options.sort(key=lambda o: o["expected_value"], reverse=True)
    for r, o in enumerate(options, 1):
        o["rank_by_expected_value"] = r

    # ---- surface
    ppcf, xg, yg = L.pitch_control_at_frame(match, period, frame, side, ball_start=passer_pos, velocities=vel)

    # ---- plot
    cmap = LinearSegmentedColormap.from_list("pc", [DEF_COLOR, "#f1f1ef", ATT_COLOR])
    fig, ax = plt.subplots(figsize=(13, 7.4), dpi=150)
    fig.patch.set_facecolor("white")
    plot_football_pitch(xlim=pitch.xlim, ylim=pitch.ylim, length=pitch.length, width=pitch.width, unit=pitch.unit,
                        color_scheme="bw", show_axis_ticks=False, ax=ax)
    dx, dy = xg[1] - xg[0], yg[1] - yg[0]
    ax.imshow(ppcf, extent=(xg[0] - dx / 2, xg[-1] + dx / 2, yg[0] - dy / 2, yg[-1] + dy / 2), origin="lower",
              cmap=cmap, vmin=0, vmax=1, alpha=0.78, zorder=2, interpolation="bilinear", aspect="auto")
    plot_positions(xy_p[opp], frame=frame, ball=False, ax=ax, color=DEF_COLOR, edgecolors="white", s=95, zorder=5)
    plot_positions(xy_p[side], frame=frame, ball=False, ax=ax, color=ATT_COLOR, edgecolors="white", s=95, zorder=5)
    plot_positions(xy_p["Ball"], frame=frame, ball=True, ax=ax, color="black", edgecolors="white", s=60, zorder=6)

    for o in options:  # pass lanes: solid = visible, dashed grey = blocked
        ax.plot([passer_pos[0], o["x"]], [passer_pos[1], o["y"]], zorder=4, lw=2.2 if o["is_actual_target"] else 1.1,
                color=INK if o["visible"] else MUTED, ls="-" if o["visible"] else (0, (4, 3)), alpha=0.95 if o["visible"] else 0.7)
    ax.scatter(*passer_pos, s=330, facecolors="none", edgecolors=INK, lw=2, zorder=7)
    top_blocked = [o for o in options if not o["visible"]][:2]  # highest-threat options that were interceptable
    for o in options:
        if o["visible"] or o["is_actual_target"] or o in top_blocked:
            tag = f'{o["player"]}  {o["pitch_control"]:.0%}' + ("  ← assisted" if o["is_actual_target"] else "")
            if not o["visible"] and not o["is_actual_target"]:
                tag += "  (interceptable)"
            left = o in top_blocked and top_blocked.index(o) == 1
            ax.annotate(tag, (o["x"], o["y"]), xytext=(-8 if left else 8, 8), textcoords="offset points",
                        ha="right" if left else "left", fontsize=8.5, color=INK, zorder=8,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    ax.annotate(names[key["pID"]] + " (ball)", passer_pos, xytext=(-10, -18), textcoords="offset points", ha="right",
                fontsize=9, color=INK, weight="bold", zorder=8, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))

    ax.set_xlim(*((-53, 4) if attack_dir == -1 else (-4, 53)))  # show the attacking half
    ax.set_ylim(-35, 35)
    mm, ss = divmod(key["gameclock"], 60)
    match_min = 45 + int(mm) + 1 if period == "secondHalf" else int(mm) + 1
    fig.suptitle("Musiala's winner: pitch control at Gnabry's assist pass", x=0.06, ha="left", fontsize=15, weight="bold", color=INK)
    ax.set_title(f"1. FC Köln 1–2 Bayern · {match_min}th minute (2nd half {int(mm):02d}:{int(ss):02d}, frame {frame}) · "
                 f"Bayern attack ← · colour = Bayern control probability", loc="left", fontsize=10, color=MUTED)
    handles = [Line2D([], [], marker="o", ls="", color=ATT_COLOR, label="Bayern"),
               Line2D([], [], marker="o", ls="", color=DEF_COLOR, label="Köln"),
               Line2D([], [], marker="o", ls="", color="black", label="Ball"),
               Line2D([], [], color=INK, label="Open lane (no defender can intercept)"),
               Line2D([], [], color=MUTED, ls=(0, (4, 3)), label="Interceptable lane")]
    ax.legend(handles=handles, loc="upper left", fontsize=8.5, frameon=True, framealpha=0.92, ncol=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("Bayern pitch control (Spearman 2018)", color=MUTED, fontsize=9)
    cb.ax.tick_params(labelsize=8, colors=MUTED)
    fig.savefig(OUT / "musiala_goal_pitch_control.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ---- json
    payload = dict(
        match=dict(id=match["match_id"], name=L.MATCHES[match["match_id"]], key_pass_team=side),
        moment=dict(period=period, gameclock_s=round(key["gameclock"], 2), frame=frame, framerate=25,
                    match_minute=match_min, event=key["eID"], passer=names[key["pID"]],
                    passer_xy=[round(float(passer_pos[0]), 2), round(float(passer_pos[1]), 2)],
                    goal_event=dict(eID=goal["eID"], gameclock_s=round(goal["gameclock"], 2), player="J. Musiala",
                                    xG=float(goal["qualifier"].get("xG", "nan")))),
        notes=["pitch_control = probability Bayern control the ball at the receiver's location (Spearman 2018, ball travel from passer).",
               "epv_threat = LaurieOnTracking EPV grid value at the receiver's location (generic grid, not fitted to this league).",
               "expected_value = pitch_control x epv_threat.",
               "visible = no defender can reach any point of the passer->receiver lane before the ball (15 m/s ball, 0.7 s reaction, 5 m/s runners); "
               "interception_margin_s is the smallest defender-minus-ball arrival time along the lane (negative = intercept possible). "
               "lane_clearance_m is the plain geometric distance from the nearest defender to the lane.",
               "offside is a snapshot at the pass frame using second-deepest defender / ball / halfway line."],
        attack_direction_x=attack_dir, offside_line_abs_x=round(float(offside_line), 2),
        options=options)
    (OUT / "musiala_goal_pass_options.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"key pass: {names[key['pID']]} at {period} {key['gameclock']:.2f}s (frame {frame}); goal at {goal['gameclock']:.2f}s")
    print(f"{'player':<18}{'PC':>6}{'EPV':>8}{'EV':>8}{'lane m':>8}{'margin s':>10}  vis  offs  target")
    for o in options:
        print(f"{o['player']:<18}{o['pitch_control']:>6.2f}{o['epv_threat']:>8.3f}{o['expected_value']:>8.3f}"
              f"{o['lane_clearance_m']:>8.1f}{o['interception_margin_s']:>10.2f}  {'Y' if o['visible'] else '-'}    "
              f"{'Y' if o['offside'] else '-'}    {'<-' if o['is_actual_target'] else ''}")


if __name__ == "__main__":
    main()
