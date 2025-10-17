from pathlib import Path
import sys
import re
import json
from statistics import mean, median, stdev, pstdev

#!/usr/bin/env python3
"""
analyze.py

Reads a file of numbers (default "1.txt"), computes basic statistics for the values:
    - count, min, max, span, mean, median, stdev (sample), pstdev (population)

Also computes the same statistics for successive ratios value[i+1] / value[i] (skips zero denominators).

Usage:
    python analyze.py [path/to/1.txt]

Outputs a short human-readable summary to stdout and writes full JSON to analysis.json.
"""


NUMBER_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")

def parse_numbers(text):
        return [float(m.group(0)) for m in NUMBER_RE.finditer(text)]

def stats_for_list(values):
        if not values:
                return {}
        n = len(values)
        mn = min(values)
        mx = max(values)
        span = mx - mn
        out = {
                "count": n,
                "min": mn,
                "max": mx,
                "span": span,
                "mean": mean(values),
                "median": median(values),
                "pstdev": pstdev(values) if n >= 1 else None,
                "stdev": stdev(values) if n >= 2 else None,
        }
        return out

def compute_ratios(values):
        ratios = []
        for a, b in zip(values, values[1:]):
                if a == 0:
                        continue
                ratios.append(b / a)
        return ratios

def main():
        path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(sys.argv[2])
        if not path.exists():
                print(f"file not found: {path}", file=sys.stderr)
                sys.exit(2)
        text = path.read_text(encoding="utf-8")
        values = parse_numbers(text)
        if not values:
                print("no numeric values found in file", file=sys.stderr)
                sys.exit(3)

        size_stats = stats_for_list(values)
        ratios = compute_ratios(values)
        ratio_stats = stats_for_list(ratios)

        result = {
                "file": str(path),
                "sizes": size_stats,
                "ratios": ratio_stats,
        }

        # human readable summary
        print(f"file: {path}")
        print(f"values: {size_stats.get('count',0)} items, min={size_stats.get('min')}, max={size_stats.get('max')}, span={size_stats.get('span')}")
        print(f"mean={size_stats.get('mean')}, median={size_stats.get('median')}, stdev={size_stats.get('stdev')}, pstdev={size_stats.get('pstdev')}")
        print()
        if ratios:
                print(f"ratios: {ratio_stats.get('count',0)} items, min={ratio_stats.get('min')}, max={ratio_stats.get('max')}, span={ratio_stats.get('span')}")
                print(f"mean={ratio_stats.get('mean')}, median={ratio_stats.get('median')}, stdev={ratio_stats.get('stdev')}, pstdev={ratio_stats.get('pstdev')}")
        else:
                print("no valid ratios (need at least two values and non-zero denominators)")

        # write JSON summary
        Path("analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

if __name__ == "__main__":
        main()