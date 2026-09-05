# Backdoor-Detector Portability to Tabular NIDS -- artifact

Code and result files for the manuscript *One Poisoning Recipe, Two Outcomes: Dataset-Dependent
Backdoor Realization and the Portability of Vision-Built Detectors to Tabular Network Intrusion
Detection*.

The paper's appendix maps every reported quantity to a file in `results/`. That index lives here
rather than in the paper, as `docs/artifact/PROVENANCE.md`; each of its rows names a file in
`results/`. The build refuses to publish if any row names a file the export does not contain.

## Layout

| Path | What |
|---|---|
| `src/` | Library code: data loading, victims, trigger construction, constraint manifests and projectors, detectors |
| `scripts/` | Numbered experiment scripts. Each writes a JSON file into `results/` and checkpoints its progress |
| `results/` | Every committed result file the paper cites, plus the resumable checkpoints |
| `tests/` | Test suite covering the constraint layer, the projectors, the detector gates and the decision logic |
| `docs/artifact/` | `PROVENANCE.md`, the quantity-to-file index the appendix defers to, and supporting notes |

## Reproducing

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/ -q
```

The datasets are not redistributed here. CTU-13 comes from the Stratosphere IPS project and
UNSW-NB15 from UNSW Canberra; both carry their own terms. Place them under `data/raw/` as the
loaders in `src/` expect. Every experiment script is checkpointed and resumes, so a long run can be
interrupted and continued, including on a different machine.

Numbers are read from the committed JSON rather than recomputed for the manuscript. Any script
whose result depends on the feature-constraint manifest keys its checkpoint on a fingerprint of
that manifest, so changing a constraint invalidates stale work instead of silently reusing it.

## License

MIT. See `LICENSE`.

## Scope

"Realizable" throughout means feature-constraint valid against the manifests in
`src/constraints_secondary.py` and the vendored CTU-13 constraint set. It does not mean a poisoned
record has been emitted as packets and re-extracted to the same feature vector, which is a stronger
property the paper states it does not establish.
