# arxiv-reproducer

[![CI](https://github.com/oscartiz/arxiv-reproducer/actions/workflows/ci.yml/badge.svg)](https://github.com/oscartiz/arxiv-reproducer/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**An AI agent that attempts to reproduce computational results from arXiv papers — and honestly reports how close it got.**

Give it an arXiv ID. It downloads the paper, reads it, picks a quantitative result it believes can be regenerated from the method description alone, implements that method from scratch in a sandboxed Python environment, runs it, and writes a reproduction report comparing its numbers against the paper's — ending with one of three verdicts: `REPRODUCED`, `PARTIALLY REPRODUCED`, or `NOT REPRODUCED`.

```bash
arxiv-repro 2301.12345
# → runs/2301.12345/REPORT.md, plus every script it wrote and every figure it generated
```

---

## Why

Reproducibility is one of science's quietest crises. Most published computational results are never independently re-derived: reproduction is unglamorous, time-consuming, and unrewarded. Yet the majority of what a reproduction requires — reading the method section, translating it to code, running it, comparing numbers — is exactly the kind of work modern LLM agents are becoming good at.

This project explores that idea seriously, with two design commitments that distinguish it from a demo:

1. **Failure is a first-class outcome.** The agent is explicitly instructed that "I could not reproduce this, and here is exactly where the paper's description was insufficient" is a *valid and valuable* result. It is never rewarded for optimistic agreement. The interesting scientific output is the **delta** between paper and reproduction, not a success rate.
2. **Everything is auditable.** Every script the agent writes, every command it runs, and every figure it produces is preserved in the run workspace. The report is not a claim you have to trust — it is a claim you can re-run.

## What a run looks like

```
$ arxiv-repro https://arxiv.org/abs/2301.12345

Fetching arXiv:2301.12345 ...
<Paper title>
<Authors>

Starting reproduction agent (this can take a while)

I've read the paper. The most feasibly reproducible result is Figure 3:
the phase transition at coupling strength g ≈ 0.42, computed via Monte
Carlo on a 32×32 lattice...
→ install_packages
→ write_file
→ run_python
The first run disagrees in the tail — the paper specifies periodic
boundary conditions in Section III which I missed. Fixing...
→ write_file
→ run_python
...
→ write_file        # REPORT.md

Done. Report: runs/2301.12345/REPORT.md
```

The resulting `REPORT.md` contains:

| Section | Contents |
|---|---|
| **Target Result** | Which figure/table/number was chosen, and why it was feasible |
| **Method Summary** | The paper's method as the agent understood it |
| **Implementation Notes** | Every assumption made where the paper was ambiguous — this is often the most scientifically interesting section |
| **Results Comparison** | Paper-reported vs. reproduced values, with relative errors |
| **Figures** | Regenerated plots saved as PNGs in the workspace |
| **Verdict** | `REPRODUCED` / `PARTIALLY REPRODUCED` / `NOT REPRODUCED`, with justification |

## Architecture

```
 arXiv ID ──▶ fetch PDF + metadata ──▶ extract text
 (paper.py)        (arXiv API)          (pypdf)
                                           │
                                           ▼
                              ┌─────────────────────────┐
                              │   Claude agent loop     │
                              │  (agent.py, tool use)   │
                              │                         │
                              │  write_file             │
                              │  read_file              │──▶ Docker sandbox
                              │  run_python             │    (sandbox.py)
                              │  install_packages       │
                              └─────────────────────────┘
                                           │
                                           ▼
                          REPORT.md  +  generated figures
                              (runs/<arxiv-id>/)
```

The pipeline has three deliberately decoupled layers:

**1. Paper acquisition** (`src/arxiv_reproducer/paper.py`)
Metadata (title, abstract, authors) comes from the arXiv Atom API; the full text is extracted from the PDF with `pypdf`. The ID parser accepts bare IDs (`2301.12345`), versioned IDs (`2301.12345v2`), old-style IDs (`hep-th/9901001`), and full `arxiv.org/abs/...` or `/pdf/...` URLs. PDFs are cached per run directory so repeated runs don't re-download.

**2. The agent loop** (`src/arxiv_reproducer/agent.py`, `prompts.py`)
A Claude tool-use loop built on the Anthropic SDK's tool runner. The agent gets exactly four tools, all scoped to the run's workspace:

| Tool | Purpose |
|---|---|
| `write_file` | Create/overwrite a script or data file in the workspace |
| `read_file` | Read back any workspace file |
| `run_python` | Execute a workspace script inside the sandbox, return stdout/stderr/exit code |
| `install_packages` | `pip install` into the sandbox (never the host) |

Both file tools resolve paths and reject anything that escapes the workspace root, so a confused (or prompt-injected — papers are untrusted input!) agent cannot read or write outside its run directory.

The system prompt (`prompts.py`) encodes the scientific workflow: *select a feasible target → state a plan → implement with exact hyperparameters from the paper → iterate on errors → quantify the discrepancy → write the report*. It also encodes the honesty rules: never fudge numbers, document every assumption, and explicitly note when compute was scaled down (fewer samples, smaller grids) and what effect that should have.

**3. The sandbox** (`src/arxiv_reproducer/sandbox.py`)
Each run gets a fresh `python:3.12-slim` Docker container with a 4 GB memory limit, 2 CPUs, and a single bind mount: the run's workspace at `/workspace`. The container lives for the whole run, so `pip install`s and intermediate files persist across tool calls — then it's destroyed. Agent-generated code never executes on the host, and per-command timeouts (10 min) plus output truncation (20 KB returned to the model) keep runaway scripts from burning the run.

## Design decisions worth reading

**Why reimplement from scratch instead of running the authors' code?**
Running the authors' repo tests whether their *code* runs; reimplementing from the *paper text* tests whether the paper actually communicates the method — which is the real reproducibility question. It also sidesteps an entire class of dependency-archaeology failures that say nothing about the science.

**Prompt caching on the paper text.**
The agent loop re-sends the conversation on every iteration, and the full paper text (often 50–100K characters) sits at the front of it. That first user message carries a `cache_control` breakpoint, so after the first iteration the paper is served from Anthropic's prompt cache at ~10% of the input price. On a 30-iteration run this is the difference between a tolerable bill and an absurd one.

**Adaptive thinking.**
The model decides per-step how much to reason — deep when interpreting an ambiguous method section, shallow when fixing a `NameError`. No thinking budget to tune.

**Honesty by construction, not just by instruction.**
Beyond the prompt rules, the artifact design enforces auditability: the verdict is only as good as the workspace backing it, and the workspace is always preserved. A reviewer (or a future automated grader) can re-run every script in the report.

**Feasibility triage is part of the agent's job.**
Most papers are *not* reproducible in a sandbox — they need proprietary data, weeks of GPU time, or lab hardware. The agent's first task is explicitly to find the *most feasibly reproducible* result, prefer simulations and analytical/numerical results, and say so if no feasible target exists. A correct "nothing here is reproducible without X" is a useful classification, not a failure of the tool.

## Repository tour

```
arxiv-reproducer/
├── src/arxiv_reproducer/
│   ├── paper.py        # arXiv API + PDF text extraction
│   ├── agent.py        # Claude tool-use loop + tool definitions
│   ├── prompts.py      # system prompt: workflow + honesty rules
│   ├── sandbox.py      # Docker container lifecycle + exec
│   └── cli.py          # arxiv-repro entry point
├── tests/
│   └── test_paper.py   # ID-parsing tests (no network, no API key needed)
├── runs/               # per-paper workspaces (gitignored)
└── pyproject.toml
```

## Installation & usage

Requirements: Python 3.11+, Docker running, and an [Anthropic API key](https://platform.claude.com/) for actual reproduction runs.

```bash
git clone https://github.com/oscartiz/arxiv-reproducer
cd arxiv-reproducer
pip install -e ".[dev]"

# Works without an API key:
pytest                       # test suite
arxiv-repro --help

# Needs an API key + Docker:
export ANTHROPIC_API_KEY=sk-ant-...
arxiv-repro 2301.12345
arxiv-repro https://arxiv.org/abs/2301.12345   # URLs work too
arxiv-repro hep-th/9901001                     # old-style IDs work too
```

Each run creates `runs/<arxiv-id>/` containing the downloaded PDF, every script the agent wrote, generated figures, and `REPORT.md`.

### Cost

A reproduction run is a long agentic loop against a frontier model — expect on the order of **a few dollars per paper**, dominated by input tokens (mitigated heavily by prompt caching) and the number of debug iterations. Simple analytical results converge in a handful of iterations; finicky simulations take more. There is no free-tier mode: the project's compute *is* the API usage.

## What makes a good target paper

Best candidates, roughly in order:

1. **Analytical/numerical results** — closed-form values, convergence rates, special-function evaluations
2. **Small simulations** — Monte Carlo, ODE/PDE toy systems, statistical mechanics on small lattices
3. **Small-scale ML experiments** — results on standard public datasets with modest training budgets
4. **Method papers with synthetic benchmarks** — the data is generated by the method itself

Poor candidates: anything needing proprietary data, large-scale training, or physical experiments. The agent will tell you when a paper falls in this bucket.

## Example reproductions

*Coming soon — the gallery will start with computational physics papers, where reproduction targets are cleanest and verdicts are easiest to audit.*

| Paper | Target result | Verdict | Report |
|---|---|---|---|
| — | — | — | — |

## Roadmap

- [ ] **Reproduction gallery** — reports for 5+ papers, committed under `examples/`
- [ ] **Figure-to-figure comparison** — extract the paper's original figures from the PDF and render them beside the regenerated ones in the report
- [ ] **Cost & token accounting** — per-run usage summary (input/output/cache-read tokens, dollar estimate)
- [ ] **Batch screening mode** — headless runs over a list of IDs, producing a feasibility/verdict spreadsheet
- [ ] **Calibrated verdicts** — have the agent attach a confidence to its verdict, then measure calibration against human review
- [ ] **Network-isolated execution** — pre-bake a scientific-stack image so the container can run with `--network none` after setup

## Status

Early stage. The full pipeline — fetch, agent loop, sandbox, report — is implemented and tested at the unit level; the reproduction gallery is pending funded API runs. Contributions and paper suggestions are welcome via issues.

## License

MIT
