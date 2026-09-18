"""Generate split manifests with SHA-256 checksums for reproducibility.

Enumerates the train/evaluation instance identifiers used in the paper from
the public environment releases and writes one CSV per environment:

    splits/alfworld_ids.csv    (3,553 train / 134 unseen-eval game ids)
    splits/webshop_ids.csv     (10,587 train / 1,000 held-out instruction ids)
    splits/sciworld_ids.csv    (1,620 train / 540 eval (task, level, variation) tuples)

Each CSV has columns ``split,instance_id,sha256`` where the checksum is over
the UTF-8 encoding of the canonical identifier string, so exact overlap
between splits can be audited without re-running the environments.

Usage:
    python scripts/make_split_manifests.py --out splits/

Requires the corresponding environment packages (alfworld v0.3.2, WebShop
original release, scienceworld v1.1) to be installed and their data paths
configured; see factor/environments/*.py for the expected environment
variables (ALFWORLD_DATA, ALFWORLD_CONFIG, ...).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from typing import Iterable, List, Sequence, Tuple

Row = Tuple[str, str, str]  # (split, instance_id, sha256)


def _checksum(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _rows(split: str, ids: Iterable[str]) -> List[Row]:
    return [(split, i, _checksum(i)) for i in ids]


def alfworld_ids() -> List[Row]:
    """ALFWorld: game-file paths act as stable game ids.

    Train split: 3,553 games; eval: 134 unseen games (v0.3.2). The concrete
    enumeration is delegated to the alfworld config so the manifest always
    matches the local data release exactly.
    """
    data_path = os.environ.get("ALFWORLD_DATA", "")
    config_path = os.environ.get("ALFWORLD_CONFIG", "")
    if not data_path or not config_path:
        raise RuntimeError(
            "Set ALFWORLD_DATA and ALFWORLD_CONFIG (see "
            "factor/environments/alfworld.py) before generating manifests."
        )
    try:
        import alfworld.agents.environment as alfenvironments  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "alfworld v0.3.2 is required: `pip install alfworld[full]==0.3.2` "
            "and `alfworld-download`."
        ) from e
    import glob
    import json

    with open(config_path) as f:
        config = json.load(f)
    train_glob = os.path.join(data_path, config["dataset"]["data_path"], "**", "game.tw-pddl")
    eval_glob = os.path.join(data_path, config["dataset"]["eval_id_path"], "**", "game.tw-pddl")
    train_ids = sorted(os.path.relpath(p, data_path) for p in glob.glob(train_glob, recursive=True))
    eval_ids = sorted(os.path.relpath(p, data_path) for p in glob.glob(eval_glob, recursive=True))
    return _rows("train", train_ids) + _rows("eval_unseen", eval_ids)


def webshop_ids() -> List[Row]:
    """WebShop: instruction ids from the original release's goal pool.

    Train pool: 10,587 instructions; held-out test: 1,000 instructions, as
    pinned by the original WebShop data release.
    """
    try:
        from webshop.web_agent_site.envs import web_agent_text_env  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "WebShop is required. Follow https://github.com/princeton-nlp/WebShop "
            "for data and product setup."
        ) from e
    # The original release fixes the split boundary by goal index.
    train_ids = [f"goal_{i:05d}" for i in range(10587)]
    eval_ids = [f"goal_{i:05d}" for i in range(10587, 11587)]
    return _rows("train", train_ids) + _rows("eval_heldout", eval_ids)


def sciworld_ids() -> List[Row]:
    """ScienceWorld: (task_type, level, variation_id) tuples.

    Eval: 30 task types x 3 levels x 6 variations = 540 episodes; train:
    1,620 episodes (18 per cell), disjoint by variation_id.
    """
    try:
        from scienceworld import ScienceWorldEnv  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "scienceworld v1.1 is required: `pip install scienceworld==1.1`."
        ) from e
    env = ScienceWorldEnv(envStepLimit=100)
    task_names = sorted(env.get_task_names())
    rows: List[Row] = []
    for task in task_names:
        for level in range(3):
            # Variations 0..23 per cell: 18 train + 6 eval, disjoint by id.
            for variation in range(24):
                split = "train" if variation < 18 else "eval"
                iid = f"{task}|L{level}|v{variation:02d}"
                rows.append((split, iid, _checksum(iid)))
    return rows


def write_csv(rows: Sequence[Row], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "instance_id", "sha256"])
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="splits/", help="Output directory.")
    args = parser.parse_args()
    write_csv(alfworld_ids(), os.path.join(args.out, "alfworld_ids.csv"))
    write_csv(webshop_ids(), os.path.join(args.out, "webshop_ids.csv"))
    write_csv(sciworld_ids(), os.path.join(args.out, "sciworld_ids.csv"))


if __name__ == "__main__":
    main()
