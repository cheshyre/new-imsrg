# SPDX-License-Identifier: GPL-3.0-only
# Copyright 2026 The imsrg authors
"""Tests of splice_tables.py."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import splice_tables as st  # noqa: E402

REPORT = """# R

#### A, X

intro

##### Flow (results.json, flow.txt)

| q | v |
|---|---|
| keep | 1 |

##### Flow directory files (valence H, TCM)

| old | table |
|---|---|

##### Farm: 1 file pairs compared

old text

#### B, X

#### decomposition

| old |
|---|
| x |

after
## 6. next
"""

GEN = """## A, X

### Flow (results.json, flow.txt)

| q | v |
|---|---|
| drop | 2 |

### Flow directory files (valence H, TCM)

| new | table |
|---|---|
| 1 | 2 |

### Farm: 1 file pairs compared

#### Per operator column

| c | n |
|---|---|
| FM0p | 3 |

(wall time 1.0 s, jobs 1)
"""


def test_full_keeps_flow_part():
    out = "\n".join(st.splice(REPORT.split("\n"), "#### A, X",
                              GEN.split("\n")))
    assert "| keep | 1 |" in out and "| drop | 2 |" not in out
    assert "| new | table |" in out and "| old | table |" not in out
    assert "###### Per operator column" in out
    assert "wall time" not in out
    assert "#### B, X" in out  # next section untouched


def test_columns_mode():
    out = "\n".join(st.splice(REPORT.split("\n"), "#### decomposition",
                              GEN.split("\n"), "columns"))
    assert "| FM0p | 3 |" in out and "| x |" not in out
    assert "after" in out and "## 6. next" in out
