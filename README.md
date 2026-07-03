# arxiv-reproducer

[![CI](https://github.com/oscartiz/arxiv-reproducer/actions/workflows/ci.yml/badge.svg)](https://github.com/oscartiz/arxiv-reproducer/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**An AI agent that attempts to reproduce computational results from arXiv papers — and honestly reports how close it got.**

Give it an arXiv ID. It downloads the paper, reads it, picks a quantitative result it believes can be regenerated from the method description alone, implements that method from scratch in a network-isolated Python sandbox, runs it, and writes a reproduction report comparing its numbers against the paper's — ending with one of three verdicts: `REPRODUCED`, `PARTIALLY REPRODUCED`, or `NOT REPRODUCED`.

```bash
arxiv-repro 2301.12345
# → runs/2301.12345/latest/REPORT.md — plus every script it wrote, every
#   figure it generated, and a machine-readable run.json
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

38402 in / 21630 out / 964512 cache-read tokens · estimated cost $1.94
· 1247s · 23 iterations

Done. Report: runs/2301.12345/20260703-142205/REPORT.md
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
| **Run metadata** | Status, model, iterations, wall clock, token counts, estimated cost — appended automatically |

### What a run leaves behind

Each run gets a fresh timestamped workspace; prior reports are never clobbered, and a rerun of the same paper reuses the cached PDF:

```
runs/2301.12345/
├── paper.pdf                # download cache, shared across runs
├── 20260703-142205/         # one workspace per run
│   ├── paper.pdf            # in-workspace copy, so auditors see what the agent saw
│   ├── simulate.py, …       # every script the agent wrote
│   ├── figure3.png, …       # every figure it generated
│   ├── REPORT.md
│   └── run.json             # machine-readable manifest
└── latest → 20260703-142205
```

`run.json` records the arXiv ID, title, model, status, extracted verdict, target result, iteration count, wall clock, per-category token totals, estimated cost, installed packages, and start/finish timestamps — so batches of runs are queryable in aggregate:

```bash
jq -r '[.arxiv_id, .verdict, .status, .estimated_cost_usd] | @tsv' runs/*/latest/run.json
```

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
                          REPORT.md + run.json + figures
                          (runs/<arxiv-id>/<timestamp>/)
```

The pipeline has three deliberately decoupled layers:

**1. Paper acquisition** (`src/arxiv_reproducer/paper.py`)
Metadata (title, abstract, authors) comes from the arXiv Atom API; the full text is extracted from the PDF with `pypdf`. The ID parser accepts bare IDs (`2301.12345`), versioned IDs (`2301.12345v2`), old-style IDs (`hep-th/9901001`), and full `arxiv.org/abs/...` or `/pdf/...` URLs. Fetches retry transient failures (408/429/5xx, connection errors) with exponential backoff and jitter; anything else surfaces immediately. Extraction tolerates individual broken pages, but a PDF that yields almost no text — a scanned paper — is refused up front with a clear error rather than fed to the agent as garbage.

**2. The agent loop** (`src/arxiv_reproducer/agent.py`, `prompts.py`)
A Claude tool-use loop built on the Anthropic SDK's tool runner. The agent gets exactly four tools, all scoped to the run's workspace:

| Tool | Purpose |
|---|---|
| `write_file` | Create/overwrite a script or data file in the workspace |
| `read_file` | Read back any workspace file |
| `run_python` | Execute a workspace script inside the sandbox, return stdout/stderr/exit code |
| `install_packages` | Wheels-only `pip install` into the sandbox's `/workspace/.deps` (never the host) |

Both file tools resolve paths and reject anything that escapes the workspace root, so a confused (or prompt-injected — papers are untrusted input!) agent cannot read or write outside its run directory. The loop itself is bounded by an iteration cap and a wall-clock cap, and accumulates every message's token usage for the final accounting.

The system prompt (`prompts.py`) encodes the scientific workflow: *select a feasible target → state a plan → implement with exact hyperparameters from the paper → iterate on errors → quantify the discrepancy → write the report*. It also encodes the honesty rules: never fudge numbers, document every assumption, and explicitly note when compute was scaled down (fewer samples, smaller grids) and what effect that should have.

**3. The sandbox** (`src/arxiv_reproducer/sandbox.py`, `docker/sandbox.Dockerfile`)
The trust boundary of the tool: papers are untrusted input, and the code the agent writes under their influence is untrusted output. Execution happens in a long-lived container built from a pre-baked image — the scientific stack (numpy, scipy, matplotlib, pandas, sympy, scikit-learn, networkx, pillow) is installed at *build* time, on a `python:3.12-slim` base pinned by digest — so at *run* time the container needs no network at all:

| Guarantee | Mechanism |
|---|---|
| No network | `--network none` — agent code cannot exfiltrate data or phone home |
| No privileges | Non-root user, `--cap-drop ALL`, `no-new-privileges` |
| No host writes | Read-only root filesystem; writable only at the bind-mounted `/workspace` and a size-capped `noexec` `/tmp` tmpfs |
| No runaways | Memory / CPU / pid limits, per-command timeouts, output truncated to 20 KB before it reaches the model |
| No leaks | Teardown guaranteed by context manager + `atexit` + SIGTERM hooks — a crash or kill never strands a container |

Package installs are **two-phase**: `install_packages` runs in a separate, ephemeral container that *does* have network but only ever executes a validated `pip install --only-binary=:all:` — plain PyPI names with optional version pins; URLs, flags, and source builds are refused before Docker is even invoked — into `/workspace/.deps`, which the offline exec container imports via `PYTHONPATH`. Agent-generated code never runs with network access, and never executes on the host.

The container lives for the whole run, so installs and intermediate files persist across tool calls — then it's destroyed.

## Design decisions worth reading

**Why reimplement from scratch instead of running the authors' code?**
Running the authors' repo tests whether their *code* runs; reimplementing from the *paper text* tests whether the paper actually communicates the method — which is the real reproducibility question. It also sidesteps an entire class of dependency-archaeology failures that say nothing about the science.

**Prompt caching on the paper text.**
The agent loop re-sends the conversation on every iteration, and the full paper text (often 50–100K characters) sits at the front of it. That first user message carries a `cache_control` breakpoint, so after the first iteration the paper is served from Anthropic's prompt cache at ~10% of the input price. On a 30-iteration run this is the difference between a tolerable bill and an absurd one.

**Adaptive thinking.**
The model decides per-step how much to reason — deep when interpreting an ambiguous method section, shallow when fixing a `NameError`. No thinking budget to tune.

**Honesty by construction, not just by instruction.**
Beyond the prompt rules, the artifact design enforces auditability: the verdict is only as good as the workspace backing it, and the workspace is always preserved. A reviewer (or a future automated grader) can re-run every script in the report.

**Every run ends in a coherent state.**
However a run ends — completed, capped, API failure after the SDK's retries, Ctrl-C — the container is torn down, `REPORT.md` exists (the agent's, or a stub saying exactly what happened and when), and `run.json` records the status. A batch over fifty papers cannot be poisoned by one hung run or one mystery directory.

**Feasibility triage is part of the agent's job.**
Most papers are *not* reproducible in a sandbox — they need proprietary data, weeks of GPU time, or lab hardware. The agent's first task is explicitly to find the *most feasibly reproducible* result, prefer simulations and analytical/numerical results, and say so if no feasible target exists. A correct "nothing here is reproducible without X" is a useful classification, not a failure of the tool.

**The supply chain is pinned.**
Host dependencies install from a universal lockfile (`uv pip compile`, runtime + dev); the sandbox base image is pinned by digest, so it cannot drift or be tag-hijacked between builds. What ran yesterday is what runs today.

## Configuration

Every tunable — model, token/iteration/wall-clock caps, container resource limits, timeouts, image tag — lives in `config.py`, with three sources; later wins:

1. Built-in defaults (everything works with no config at all)
2. `./arxiv-repro.toml`, or the file named in `ARXIV_REPRO_CONFIG`
3. `ARXIV_REPRO_*` environment variables

```toml
# arxiv-repro.toml — trade accuracy for cost, loosen limits for heavier sims
model = "claude-sonnet-5"
max_iterations = 40
memory_limit = "8g"
```

```bash
ARXIV_REPRO_MODEL=claude-haiku-4-5 arxiv-repro 2301.12345   # one-off override
```

Every key is documented in [`arxiv-repro.example.toml`](arxiv-repro.example.toml). Configuration errors fail fast and say exactly what's wrong — unknown keys are listed alongside the valid set. Deliberately *not* configurable: the sandbox hardening flags. Network isolation and privilege dropping are invariants, not tunables.

## Repository tour

```
arxiv-reproducer/
├── src/arxiv_reproducer/
│   ├── paper.py             # arXiv API + PDF extraction, retried, scan-refusing
│   ├── agent.py             # Claude tool-use loop, tool definitions, run manifest
│   ├── prompts.py           # system prompt: workflow + honesty rules
│   ├── sandbox.py           # container lifecycle, hardening, two-phase installs
│   ├── config.py            # defaults < TOML < env, validated with clear errors
│   ├── costs.py             # per-run token accounting + dollar estimate
│   ├── manifest.py          # run.json writer + verdict extraction
│   ├── runs.py              # timestamped, non-clobbering run directories
│   ├── retry.py             # exponential backoff for transient HTTP failures
│   ├── logs.py              # structured logging (plain or JSON lines)
│   ├── cli.py               # arxiv-repro entry point
│   └── docker/
│       └── sandbox.Dockerfile   # pre-baked scientific image, digest-pinned base
├── tests/                   # full suite runs with no Docker and no API key
├── arxiv-repro.example.toml # every config key, documented
├── requirements-lock.txt    # universal lockfile (uv) — CI installs from this
├── Makefile                 # install / test / integration / lint / typecheck / check
└── runs/                    # per-paper workspaces (gitignored)
```

## Installation & usage

Requirements: Python 3.11+, Docker running, and an [Anthropic API key](https://platform.claude.com/) for actual reproduction runs.

```bash
git clone https://github.com/oscartiz/arxiv-reproducer
cd arxiv-reproducer
python3 -m venv .venv
make install         # locked dependencies + the package, editable

# Works without an API key or Docker:
make check           # lint + types + full unit suite
.venv/bin/arxiv-repro --help

# A real run needs credentials + Docker:
export ANTHROPIC_API_KEY=sk-ant-...    # or `ant auth login`
.venv/bin/arxiv-repro 2301.12345
.venv/bin/arxiv-repro https://arxiv.org/abs/2301.12345   # URLs work too
.venv/bin/arxiv-repro hep-th/9901001                     # old-style IDs work too
```

The first real run builds the sandbox image, which takes a few minutes; after that, container start is instant. `make build-image` builds it ahead of time.

Useful flags: `--runs-dir` to relocate the workspaces, `-v` for debug logging, `--log-json` for JSON log lines on stderr (the human-facing progress stays on stdout either way).

### Cost

A reproduction run is a long agentic loop against a frontier model — expect on the order of **a few dollars per paper**, dominated by input tokens (mitigated heavily by prompt caching) and the number of debug iterations. Simple analytical results converge in a handful of iterations; finicky simulations take more. There is no free-tier mode: the project's compute *is* the API usage.

You never have to guess what a run cost: the exact token counts and dollar estimate are printed at the end of every run, appended to `REPORT.md`, and recorded in `run.json`.

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
- [x] **Cost & token accounting** — per-run usage summary (input/output/cache-read tokens, dollar estimate)
- [ ] **Batch screening mode** — headless runs over a list of IDs, producing a feasibility/verdict spreadsheet
- [ ] **Calibrated verdicts** — have the agent attach a confidence to its verdict, then measure calibration against human review
- [x] **Network-isolated execution** — pre-bake a scientific-stack image so the container can run with `--network none` after setup

## Development

```bash
make help            # list all targets
make check           # ruff + mypy (disallow_untyped_defs) + pytest, 90% coverage floor
make integration     # real-Docker suite: proves the hardening against a live daemon
make lock            # regenerate requirements-lock.txt after changing dependencies
```

The unit suite needs neither Docker nor an API key. The integration tests are opt-in (`pytest --run-docker -k RealDocker`) and CI runs them on every push in a dedicated job that builds the sandbox image from scratch and re-proves the container guarantees — network isolation, non-root with read-only rootfs, two-phase install, teardown — against a real Docker daemon. The main CI job runs lint, mypy, and the unit suite on Python 3.11 and 3.12, installing from the lockfile.

## Status

Early stage. The full pipeline — fetch, agent loop, sandbox, report — is implemented, with unit tests and a real-Docker integration suite run in CI; the reproduction gallery is pending funded API runs. Contributions and paper suggestions are welcome via issues.

## License

MIT
