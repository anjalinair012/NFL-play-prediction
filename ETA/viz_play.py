"""
Visualize a play from the NFL Big Data Bowl 2026 (Prediction) data.

Usage (script):
    python viz_play.py --week 1                       # random play from week 1
    python viz_play.py --week 1 --game 2023090700 --play 101 --gif play.gif

Usage (Jupyter):
    from viz_play import load_week, get_play, plot_play, animate_play
    inp, out = load_week("data/train", 1)
    play = get_play(inp, out)                  # random play
    plot_play(play)
    from IPython.display import HTML
    HTML(animate_play(play).to_jshtml())
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation
from matplotlib.patches import Rectangle

FIELD_LEN, FIELD_WID = 120.0, 53.3

ROLE_COLORS = {
    "Targeted Receiver": "#f5c518",
    "Passer": "#ffffff",
    "Other Route Runner": "#4a90e2",
    "Defensive Coverage": "#e24a4a",
}
SIDE_COLORS = {"Offense": "#4a90e2", "Defense": "#e24a4a"}


# ---------------------------------------------------------------- data
def load_week(data_dir, week):
    data_dir = Path(data_dir)
    inp = pd.read_csv(data_dir / f"input_2023_w{week:02d}.csv")
    out = pd.read_csv(data_dir / f"output_2023_w{week:02d}.csv")
    return inp, out


def normalize(df, has_angles=True):
    """Rotate left-moving plays 180 deg so offense always moves toward +x."""
    df = df.copy()
    for c in ("x", "y", "dir", "o", "ball_land_x", "ball_land_y"):
        if c in df:
            df[c] = df[c].astype(float)
    left = df["play_direction"] == "left"
    if "x" in df:
        df.loc[left, "x"] = FIELD_LEN - df.loc[left, "x"]
        df.loc[left, "y"] = FIELD_WID - df.loc[left, "y"]
    if has_angles:
        for c in ("dir", "o"):
            if c in df:
                df.loc[left, c] = (df.loc[left, c] + 180) % 360
    if "ball_land_x" in df:
        df.loc[left, "ball_land_x"] = FIELD_LEN - df.loc[left, "ball_land_x"]
        df.loc[left, "ball_land_y"] = FIELD_WID - df.loc[left, "ball_land_y"]
    return df


def get_play(inp, out, game_id=None, play_id=None, normalize_dir=True, seed=None):
    """Return one play as a single tracking table with pre- and post-throw frames."""
    if game_id is None or play_id is None:
        keys = inp[["game_id", "play_id"]].drop_duplicates()
        game_id, play_id = keys.sample(1, random_state=seed).iloc[0]

    pi = inp[(inp.game_id == game_id) & (inp.play_id == play_id)].copy()
    po = out[(out.game_id == game_id) & (out.play_id == play_id)].copy()
    if pi.empty:
        raise ValueError(f"No play {game_id}/{play_id} in this week")

    # Player info from the input file, attached to the output rows
    meta_cols = [c for c in ["nfl_id", "player_name", "player_position", "player_side",
                             "player_role", "player_to_predict"] if c in pi]
    meta = pi[meta_cols].drop_duplicates("nfl_id")

    last_in = pi.frame_id.max()
    pi["phase"] = "pre"
    po = po.merge(meta, on="nfl_id", how="left")
    po["frame_id"] = po["frame_id"] + last_in        # continue the timeline
    po["phase"] = "post"
    po["play_direction"] = pi.play_direction.iloc[0]

    play_info = pi[["play_direction", "ball_land_x", "ball_land_y",
                    "num_frames_output"]].iloc[[0]].copy()

    if normalize_dir:
        pi = normalize(pi)
        po = normalize(po, has_angles=False)
        play_info = normalize(play_info, has_angles=False)

    track = pd.concat([pi, po], ignore_index=True).sort_values(["nfl_id", "frame_id"])
    return {
        "game_id": game_id,
        "play_id": play_id,
        "track": track,
        "normalized": normalize_dir,
        "play_direction": play_info.play_direction.iloc[0],
        "throw_frame": last_in,
        "ball_land": (play_info.ball_land_x.iloc[0], play_info.ball_land_y.iloc[0]),
    }


def player_color(row):
    role = row.get("player_role")
    if role in ROLE_COLORS:
        return ROLE_COLORS[role]
    return SIDE_COLORS.get(row.get("player_side"), "#cccccc")


# ---------------------------------------------------------------- field
def draw_field(ax):
    ax.add_patch(Rectangle((0, 0), FIELD_LEN, FIELD_WID, color="#2e7d32", zorder=0))
    for x0 in (0, 110):
        ax.add_patch(Rectangle((x0, 0), 10, FIELD_WID, color="#1b5e20", zorder=0))
    for x in range(10, 111, 5):
        ax.plot([x, x], [0, FIELD_WID], color="white", lw=1.5 if x % 10 == 0 else 0.6,
                alpha=0.8, zorder=1)
    for x in range(20, 110, 10):
        n = x - 10 if x <= 60 else 110 - x
        for y in (5, FIELD_WID - 5):
            ax.text(x, y, str(n), color="white", ha="center", va="center",
                    fontsize=11, alpha=0.7, zorder=1)
    ax.set_xlim(0, FIELD_LEN)
    ax.set_ylim(0, FIELD_WID)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])


def _zoom(ax, track, min_span=50):
    cx = (track.x.min() + track.x.max()) / 2
    half = max(min_span, track.x.max() - track.x.min() + 16) / 2
    lo, hi = max(0, cx - half), min(FIELD_LEN, cx + half)
    ax.set_xlim(lo, hi)


def _title(play):
    t = f"Game {play['game_id']}  Play {play['play_id']}"
    if play.get("normalized", True):
        return t + "  (normalized: offense -> right)"
    return t + f"  (raw, play_direction={play.get('play_direction', '?')})"


# ---------------------------------------------------------------- static plot
def plot_play(play, ax=None, zoom=True):
    """Full trajectories: solid = before throw, dashed = ball in the air."""
    track = play["track"]
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 6.5))
    draw_field(ax)

    for nfl_id, p in track.groupby("nfl_id"):
        row = p.iloc[0]
        c = player_color(row)
        pre, post = p[p.phase == "pre"], p[p.phase == "post"]
        ax.plot(pre.x, pre.y, color=c, lw=1.5, alpha=0.9, zorder=2)
        if len(post):
            # connect throw point to first post-throw frame
            xs = np.r_[pre.x.iloc[-1], post.x]; ys = np.r_[pre.y.iloc[-1], post.y]
            ax.plot(xs, ys, color=c, lw=2, ls="--", zorder=3)
            ax.scatter(post.x.iloc[-1], post.y.iloc[-1], color=c, s=40,
                       edgecolor="black", zorder=4)
        ax.scatter(pre.x.iloc[-1], pre.y.iloc[-1], color=c, s=90,
                   edgecolor="black", zorder=4)
        ax.text(pre.x.iloc[-1], pre.y.iloc[-1] + 1.2, str(row.get("player_position", "")),
                fontsize=7, color="white", ha="center", zorder=5)

    bx, by = play["ball_land"]
    ax.scatter(bx, by, marker="X", s=200, color="#ff9800", edgecolor="black",
               zorder=6, label="Ball landing spot")

    if zoom:
        _zoom(ax, track)

    handles = [plt.Line2D([], [], color=c, lw=3, label=r) for r, c in ROLE_COLORS.items()]
    handles += [plt.Line2D([], [], color="white", ls="--", label="ball in air"),
                plt.Line2D([], [], marker="X", ls="", color="#ff9800", label="landing spot")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1),
              fontsize=8, facecolor="#1b5e20", labelcolor="white", framealpha=0.95)
    ax.set_title(_title(play))
    return ax


# ---------------------------------------------------------------- animation
def animate_play(play, interval=100, show_velocity=True, zoom=True):
    """Animate frame by frame (10 fps = 100 ms per frame)."""
    track = play["track"]
    frames = sorted(track.frame_id.unique())
    throw = play["throw_frame"]

    fig, ax = plt.subplots(figsize=(14, 6.5))
    draw_field(ax)
    if zoom:
        _zoom(ax, track)
    bx, by = play["ball_land"]
    ax.scatter(bx, by, marker="X", s=200, color="#ff9800", edgecolor="black", zorder=6)
    ax.set_title(_title(play))

    # Players who aren't predicted vanish from the data after the throw;
    # we keep them frozen and faded at their last known spot.
    ids = track.nfl_id.unique()
    meta = track.drop_duplicates("nfl_id").set_index("nfl_id")
    colors = [player_color(meta.loc[i]) for i in ids]
    pos = {i: g.set_index("frame_id") for i, g in track.groupby("nfl_id")}

    scat = ax.scatter(np.zeros(len(ids)), np.zeros(len(ids)), s=120, c=colors,
                      edgecolor="black", zorder=5)
    quiv = ax.quiver(np.zeros(len(ids)), np.zeros(len(ids)), np.zeros(len(ids)),
                     np.zeros(len(ids)), color="white", scale=1, scale_units="xy",
                     angles="xy", width=0.002, zorder=4) if show_velocity else None
    label = ax.text(0.99, 0.97, "", transform=ax.transAxes, ha="right", va="top",
                    fontsize=12, color="white",
                    bbox=dict(facecolor="black", alpha=0.6))

    def update(f):
        xy, uv, alpha = [], [], []
        for i in ids:
            g = pos[i]
            row = g.loc[f] if f in g.index else g[g.index <= f].iloc[-1]
            xy.append((row.x, row.y))
            alpha.append(1.0 if f in g.index else 0.3)
            # velocity arrow (input frames only have s/dir); 0.5 s lookahead
            if show_velocity and f in g.index and not pd.isna(row.get("s", np.nan)):
                th = np.deg2rad(row["dir"])
                uv.append((0.5 * row.s * np.sin(th), 0.5 * row.s * np.cos(th)))
            else:
                uv.append((0, 0))
        xy = np.array(xy)
        scat.set_offsets(xy)
        scat.set_alpha(None)
        fc = scat.get_facecolors()
        if len(fc) == len(ids):
            fc[:, 3] = alpha
            scat.set_facecolors(fc)
        if quiv is not None:
            quiv.set_offsets(xy)
            uv = np.array(uv)
            quiv.set_UVC(uv[:, 0], uv[:, 1])
        phase = "before throw" if f <= throw else "BALL IN THE AIR"
        label.set_text(f"frame {f}  |  {phase}")
        return (scat, label) + ((quiv,) if quiv is not None else ())

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=interval,
                                   blit=False)
    plt.close(fig)
    return anim


# ---------------------------------------------------------------- CLI
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="nfl-big-data-bowl-2026-prediction\\train")
    ap.add_argument("--week", type=int, default=1)
    ap.add_argument("--game", type=int)
    ap.add_argument("--play", type=int)
    ap.add_argument("--raw", action="store_true", help="don't normalize direction")
    ap.add_argument("--gif", help="save animation to this .gif path")
    args = ap.parse_args()

    inp, out = load_week(args.data, args.week)
    play = get_play(inp, out, args.game, args.play, normalize_dir=not args.raw)
    plot_play(play)
    plt.show()
    if args.gif:
        animate_play(play).save(args.gif, writer="pillow", fps=10)
        print("saved", args.gif)