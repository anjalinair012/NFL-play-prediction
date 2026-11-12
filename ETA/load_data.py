from pathlib import Path

import pandas as pd

DATA_SUBDIR = Path("nfl-big-data-bowl-2026-prediction") / "train"


def find_train_dir() -> Path:
    """Walk up from the working directory (works in notebooks, where __file__ is undefined)."""
    start = Path.cwd().resolve()
    for p in [start, *start.parents]:
        if (p / DATA_SUBDIR).is_dir():
            return p / DATA_SUBDIR
    raise FileNotFoundError(f"Could not find {DATA_SUBDIR} above {start}")


TRAIN_DIR = find_train_dir()

# Compact dtypes: low-cardinality strings become categories, numerics are downcast.
INPUT_DTYPES = {
    "game_id": "int32",
    "play_id": "int16",
    "player_to_predict": "bool",
    "nfl_id": "int32",
    "frame_id": "int16",
    "play_direction": "category",
    "absolute_yardline_number": "int8",
    "player_name": "category",
    "player_height": "category",
    "player_weight": "int16",
    "player_position": "category",
    "player_side": "category",
    "player_role": "category",
    "x": "float32",
    "y": "float32",
    "s": "float32",
    "a": "float32",
    "dir": "float32",
    "o": "float32",
    "num_frames_output": "int16",
    "ball_land_x": "float32",
    "ball_land_y": "float32",
}
OUTPUT_DTYPES = {
    "game_id": "int32",
    "play_id": "int16",
    "nfl_id": "int32",
    "frame_id": "int16",
    "x": "float32",
    "y": "float32",
}


def load_split(prefix: str, dtypes: dict) -> pd.DataFrame:
    """Load and concatenate all weekly files, e.g. input_2023_w01.csv ... w18."""
    files = sorted(TRAIN_DIR.glob(f"{prefix}_*.csv"))
    if not files:
        raise FileNotFoundError(f"No {prefix}_*.csv files in {TRAIN_DIR}")
    frames = []
    for f in files:
        df = pd.read_csv(f, dtype=dtypes, parse_dates=["player_birth_date"] if prefix == "input" else None)
        df["week"] = int(f.stem.rsplit("_w", 1)[1])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_train():
    inputs = load_split("input", INPUT_DTYPES)
    outputs = load_split("output", OUTPUT_DTYPES)
    return inputs, outputs


if __name__ == "__main__":
    inputs, outputs = load_train()
    print("input :", inputs.shape, f"{inputs.memory_usage(deep=True).sum() / 1e6:.0f} MB")
    print("output:", outputs.shape, f"{outputs.memory_usage(deep=True).sum() / 1e6:.0f} MB")
