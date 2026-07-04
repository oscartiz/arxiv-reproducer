# Reproduction gallery

Each entry is one complete, auditable reproduction run, promoted verbatim from
its run workspace. Nothing here is hand-edited: the report is what the agent
wrote, the manifest is what the run recorded.

## Entry layout

```
examples/<arxiv-id>/
├── REPORT.md      # the agent's reproduction report, incl. the Run metadata footer
├── run.json       # machine-readable manifest (verdict, tokens, cost, timings)
├── *.py           # every script the agent wrote — the report is re-runnable
└── *.png          # regenerated figures referenced by the report
```

## Promoting a run

After a run you consider gallery-worthy:

1. Copy the workspace, dropping only the PDF (papers are arXiv's to
   distribute, not ours):

   ```bash
   ID=2301.12345
   SRC=runs/$ID/latest
   mkdir -p examples/$ID
   cp "$SRC"/REPORT.md "$SRC"/run.json examples/$ID/
   cp "$SRC"/*.py "$SRC"/*.png examples/$ID/ 2>/dev/null || true
   ```

2. Add a row to the gallery table in the top-level README:

   ```markdown
   | [arXiv:<id>](https://arxiv.org/abs/<id>) | <target result, one line> | <VERDICT> | [report](examples/<id>/REPORT.md) |
   ```

3. Sanity-check the entry before committing:
   - the verdict in the README row matches `run.json`'s `verdict`;
   - `REPORT.md` documents every assumption and any compute scale-down;
   - a discrepancy between paper and reproduction is *described*, not smoothed
     over — the delta is the scientific content.

A `NOT REPRODUCED` or `PARTIALLY REPRODUCED` run with a well-argued report is
gallery-worthy; a `REPRODUCED` run with a thin report is not.
