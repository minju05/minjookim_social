"""GridToM data loader — info.json → GridToMRecord"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DATASET_DIR = Path("/home/seohyeon/.00_project/26-01-SAI/GridToM_data")
INFO_JSON    = DATASET_DIR / "info.json"
FRAMES_DIR   = DATASET_DIR / "GridToM_1stBelief"

BeliefType = Literal["TrueBelief", "FalseBelief", "both"]


@dataclass(frozen=True)
class GridToMRecord:
    index:       int
    belief_type: str        # "TrueBelief" | "FalseBelief"
    question:    str
    options:     tuple      # (opt_a, opt_b)  — room/color strings
    answer:      str        # lowercase, matches one of options
    env_desc:    str
    caption:     str
    nodes:       tuple      # (n0, n1, n2, n3) key-frame markers


def _key_frame_ids(nodes: tuple) -> list[int]:
    n0, n1, n2, n3 = nodes
    return [n0, (n0+n1)//2, n1, (n1+n2)//2, n2, (n2+n3)//2, n3]


def load_records(belief_type: BeliefType = "both") -> list[GridToMRecord]:
    records = []
    with INFO_JSON.open() as f:
        for line in f:
            item = json.loads(line)
            for belief in item["first_order_belief"]:
                if belief_type != "both" and belief["type"] != belief_type:
                    continue
                records.append(GridToMRecord(
                    index       = item["index"],
                    belief_type = belief["type"],
                    question    = belief["question"],
                    options     = tuple(belief["options"]),
                    answer      = belief["answer"].strip().lower(),
                    env_desc    = item["env_desc"],
                    caption     = belief["caption"],
                    nodes       = tuple(item["nodes"]),
                ))
    return records


def load_frames_b64(record: GridToMRecord) -> list[str]:
    """Returns list of base64-encoded PNG strings for the 7 key frames."""
    frame_dir = FRAMES_DIR / str(record.index) / record.belief_type / "full"
    out = []
    for fid in _key_frame_ids(record.nodes):
        p = frame_dir / f"{fid}.png"
        if p.exists():
            out.append(base64.b64encode(p.read_bytes()).decode())
    return out
