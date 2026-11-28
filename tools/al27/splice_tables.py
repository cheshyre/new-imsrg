#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright 2026 The imsrg authors
"""Replace generated sections of a markdown report by regenerated ones.

  splice_tables.py REPORT.md  HEADING=GENERATED.md[:MODE] ...

For each argument, the section of REPORT that starts at the line
`HEADING` (a full heading line such as "#### v2 vs v1, N2LOsat") is
rewritten from the diff_variants.py output GENERATED.md (a `##` section whose
sub-headings are demoted to the level of HEADING + 1). MODE:

* `full`  (default): everything from the "Flow directory files" sub-section
  of the generated text on replaces the report's text from its "Flow
  directory files" heading on; the report's "Flow (results.json, flow.txt)"
  part is kept (it can be hand-corrected, e.g. when a run's flow file was
  overwritten);
* `columns`: only the "Per operator column" table of the generated text
  replaces the first table after HEADING.

The section ends at the next heading of the level of HEADING or higher.
"""

import re
import sys


def level(line):
    """Heading level of a markdown line (0 if not a heading)."""
    m = re.match(r"(#+) ", line)
    return len(m.group(1)) if m else 0


def section_bounds(lines, heading):
    """(start, end) line indices of the section starting at `heading`."""
    start = lines.index(heading)
    lev = level(heading)
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if 0 < level(lines[i]) <= lev:
            end = i
            break
    return start, end


def demote(lines, delta):
    """Demote the headings of `lines` by `delta` levels."""
    return [("#" * delta + ln) if level(ln) else ln for ln in lines]


def subsection(lines, title):
    """Index of the first heading line containing `title`."""
    for i, ln in enumerate(lines):
        if level(ln) and title in ln:
            return i
    raise ValueError(f"no sub-section {title!r}")


def splice(report_lines, heading, gen_lines, mode="full"):
    """Return report_lines with the section of `heading` regenerated."""
    start, end = section_bounds(report_lines, heading)
    sec = report_lines[start:end]
    # generated text starts with "## label": its level maps to heading's
    delta = level(heading) - 2
    gen = demote(gen_lines, delta)
    if mode == "columns":
        g0 = subsection(gen, "Per operator column")
        g1 = g0 + 1
        while g1 < len(gen) and not level(gen[g1]):
            g1 += 1
        new_tab = [ln for ln in gen[g0 + 1:g1] if ln.startswith("|")]
        t0 = next(i for i, ln in enumerate(sec) if ln.startswith("|"))
        t1 = t0
        while t1 < len(sec) and sec[t1].startswith("|"):
            t1 += 1
        sec = sec[:t0] + new_tab + sec[t1:]
    else:
        g0 = subsection(gen, "Flow directory files")
        s0 = subsection(sec, "Flow directory files")
        tail = [ln for ln in gen[g0:] if not ln.startswith("(wall time")]
        while tail and not tail[-1].strip():
            tail.pop()
        sec = sec[:s0] + tail + [""]
    return report_lines[:start] + sec + report_lines[end:]


def main(argv):
    """Entry point."""
    if len(argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    with open(argv[1]) as f:
        lines = f.read().split("\n")
    for arg in argv[2:]:
        heading, _, rest = arg.partition("=")
        path, _, mode = rest.partition(":")
        with open(path) as f:
            gen = f.read().split("\n")
        lines = splice(lines, heading, gen, mode or "full")
    with open(argv[1], "w") as f:
        f.write("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
