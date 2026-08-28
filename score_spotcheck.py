"""Compute the explanation-bearing rate from the labelled spot-check file.

Reads paper/ARR_Rebuttal_Spotcheck_Sample.md, parses each `Label:` line,
and reports yes/partial/no counts. Use 'partial' for borderline cases
(e.g. an explanation embedded in an off-topic utterance).
"""
import re
import sys

PATH = "paper/ARR_Rebuttal_Spotcheck_Sample.md"

VALID = {"yes", "no", "partial"}


def main():
    with open(PATH) as f:
        text = f.read()
    items = re.findall(r"^## (\d+)\.", text, flags=re.MULTILINE)
    labels = re.findall(r"^Label:\s*(.*?)\s*$", text, flags=re.MULTILINE)
    if len(items) != len(labels):
        print(f"WARN: found {len(items)} items but {len(labels)} Label: lines.")
    counts = {"yes": 0, "partial": 0, "no": 0, "(empty)": 0, "(invalid)": 0}
    for label in labels:
        # strip markdown bold/italic/whitespace, lowercase
        norm = re.sub(r"[*_`\s]", "", label).lower()
        if not norm:
            counts["(empty)"] += 1
        elif norm in VALID:
            counts[norm] += 1
        else:
            counts["(invalid)"] += 1
            print(f"  invalid label: {label!r}")
    n = sum(counts.values())
    print(f"\nLabelled {n - counts['(empty)']} / {n} items.\n")
    for k, v in counts.items():
        pct = 100.0 * v / n if n else 0
        print(f"  {k:>10s}: {v:>3d} ({pct:5.1f}%)")
    rated = counts["yes"] + counts["partial"] + counts["no"]
    if rated:
        strict_rate = 100.0 * counts["yes"] / rated
        loose_rate = 100.0 * (counts["yes"] + counts["partial"]) / rated
        print(f"\nStrict explanation-bearing rate (yes only): {strict_rate:.1f}%")
        print(f"Loose explanation-bearing rate (yes + partial): {loose_rate:.1f}%")
    else:
        print("\nNo labels filled in yet.")
        sys.exit(1)


if __name__ == "__main__":
    main()
