# Paper

*Reproduce — or Say Why Not: An Auditable LLM Agent for Text-Only Reproduction
of Computational Results* — a working draft describing this repository's
methodology and system. It deliberately reports no headline numbers: the
funded evaluation (verdict distribution, delta taxonomy, confidence
calibration, cost per verdict) is pre-stated in §7 and will land in the
gallery first.

## Building

```bash
make paper          # uses tectonic if installed, else pdflatex (two passes)
```

CI builds the PDF on every change to `paper/` and uploads it as the
`paper-pdf` workflow artifact — download it from the Actions run page if you
don't have a local TeX installation.

Build artifacts (`*.aux`, `*.log`, `main.pdf`, …) are gitignored; the source
of truth is `main.tex`.
