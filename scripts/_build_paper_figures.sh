#!/usr/bin/env bash
# Regenerate every figure the IEEE Access manuscript includes, then place them.
#
# Why this exists. Before this script, NO code wrote to paper_access/figures/. All nine PDFs had
# been hand-copied, they shared a single mtime, and four of them (fig2, fig3, fig5, fig6) no
# longer matched what their own producing script emitted -- so the shipped figures came from an
# invocation nobody could reproduce. A figure in a paper has to be a build product.
#
# Sources, and why they differ:
#   scripts/07_make_figures.py      -> figures/            (six data figures)
#   scripts/59_decomposition_figure -> paper/figures/      (writes there by its own hardcoded path)
#   scripts/55_make_fig_pipeline.py -> paper/figures/      (same)
# The two schematic scripts predate paper_access/ and still target paper/. Rather than edit their
# output paths and risk breaking the older manuscript that also builds from them, this script
# collects from both locations.
#
# Run:  bash scripts/_build_paper_figures.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
DEST=paper_access/figures

echo "=== regenerating ==="
"$PY" scripts/07_make_figures.py
"$PY" scripts/59_decomposition_figure.py
"$PY" scripts/55_make_fig_pipeline.py

echo "=== placing into $DEST ==="
for f in fig1_failure_boundary fig2_portability fig3_h3_confound \
         fig4_evasion_window fig5_adaptive_threshold fig6_nc_masks; do
  cp -v "figures/${f}.pdf" "$DEST/${f}.pdf"
done
for f in fig_decomposition fig_pipeline; do
  cp -v "paper/figures/${f}.pdf" "$DEST/${f}.pdf"
done

echo "=== included by the manuscript but not produced above (should be empty) ==="
grep -ohE 'includegraphics(\[[^]]*\])?\{figures/[^}]+\}' paper_access/sections/*.tex \
  | sed 's/.*{figures\///; s/}//' | sort -u \
  | while read -r want; do
      [ -f "$DEST/$want" ] || echo "  MISSING: $want"
    done

echo "=== shipped but never included (dead weight) ==="
for f in "$DEST"/*.pdf; do
  b=$(basename "$f")
  grep -q "figures/$b" paper_access/sections/*.tex paper_access/main.tex 2>/dev/null \
    || echo "  ORPHAN: $b"
done

echo "done"
