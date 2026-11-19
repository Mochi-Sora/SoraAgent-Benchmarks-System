<p align="center">
  <img src="https://raw.githubusercontent.com/Mochi-Sora/SoraAgent/main/assets/icon.png" alt="SoraAgent Benchmarks" width="96">
</p>

<h1 align="center">SoraAgent Benchmarks System</h1>

<p align="center"><strong>Measured, not claimed.</strong></p>

<p align="center">
  <a href="https://github.com/Mochi-Sora/SoraAgent-Benchmarks"><img src="https://img.shields.io/badge/live%20score-86.3%2F100-6f42c1" alt="Live score 86.3/100"></a>
  <a href="https://github.com/Mochi-Sora/SoraAgent"><img src="https://img.shields.io/badge/SoraAgent-0.4.0-1f6feb" alt="SoraAgent 0.4.0"></a>
  <a href="https://github.com/Mochi-Sora/SoraAgent-Benchmarks/blob/main/SoraAgent-live-scoreboard.md"><img src="https://img.shields.io/badge/edge%20checks-54%2F54-2ea44f" alt="54 of 54 edge checks"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/license-MIT-2ea44f" alt="MIT license"></a>
</p>

<p align="center">
  <a href="https://github.com/Mochi-Sora/SoraAgent">SoraAgent</a> ·
  <a href="https://mochi-sora.github.io/SoraOfficialWebsite/#benchmarks">Website</a> ·
  <a href="https://discord.gg/s8W6JfBSMH">Discord</a> ·
  <a href="https://github.com/Mochi-Sora/SoraAgent-Benchmarks">Public scoreboard</a>
</p>

A reproducible benchmark harness for [SoraAgent](https://github.com/Mochi-Sora/SoraAgent),
with a portable answer-quality suite for other agent harnesses. It runs nine
suites (eleven scored families) covering what the project cares about:

| Family | What the benchmark answers |
| --- | --- |
| **Token usage** | How many prompt/completion/effective tokens does a task consume, and where do they go (system, identity, clock, tool schemas, conversation)? |
| **Tool-use accuracy** | Did the agent choose the right tools, with the right arguments, in the right order, without forbidden or unnecessary calls? |
| **Memory retention accuracy** | Does recall return the right facts, avoid stale/contradictory facts, and produce grounded answers? |
| **Memory retention efficiency** | How many facts are recovered per recall token, how dense is the recall payload, and how much does it compress the corpus? |
| **Dreaming accuracy** | Does nightly consolidation merge the right duplicates, flag real numeric conflicts, infer the right links, retain sources, expire stale records, and stay idempotent? |
| **Dreaming efficiency** | How much wall time and how many backend calls, actions, and wake-model tokens does consolidation cost per memory? |
| **Response speed** | End-to-end latency, provider latency, tool latency, framework overhead, and tokens/second. |
| **Autonomous task completion** | When the agent does extra work without being asked, was it useful, neutral, or harmful? Did it respect approval gates? |
| **Prompt caching** | What fraction of each request is a reusable prefix, how stable is that prefix, and which component breaks it? |
| **Files, specialists, memory, graph & security** | Do DOCX/ODF/EPUB/EML/RTF/HTML/CSV/archive conversions survive malformed, empty, oversized, nested, and script-bearing inputs? Do specialist scoping, routing, auto-create quotas, workspace escapes, daily stores, dream provenance, retry/forget, and daily-only consolidation behave correctly? Do graph extraction, graph-fact recall, native graph projection, transient inference cleanup, and 404 lookups work? Do state, provenance, daily session scope, durable dream artifacts, tool framing, redirect/peer guards, approvals, iMessage auth, and attachment caps hold? |
| **General instruction following** | Does the agent follow exact formats, arithmetic, negation, structured output, multilingual prompts, and uncertainty without relying on SoraAgent tools? This is the portable suite for other harnesses. |

The benchmark is **not** a leaderboard. Simulated mode is a deterministic
reference policy that exercises the real SoraAgent loop; live mode runs the same
scenarios through a real model. The **general** suite is portable and uses only
the text task and answer terms. Raw metrics and uncertainty intervals are more
important than the heuristic 0–100 scores.

---

## Quick start

Clone both repositories side by side, because `run_benchmark.py` loads the sibling
`../SoraAgent` checkout:

```bash
git clone https://github.com/Mochi-Sora/SoraAgent.git
git clone https://github.com/Mochi-Sora/SoraAgent-Benchmarks-System.git
cd SoraAgent-Benchmarks-System
```

Then run:

```bash
# Full deterministic run (no API key, no network, ~2 seconds)
python3 run_benchmark.py

# One or more suites
python3 run_benchmark.py --suites tools memory caching
python3 run_benchmark.py --suites features
python3 run_benchmark.py --suites general --runs 3

# Portable external-agent run (command adapter)
python3 run_benchmark.py --adapter command \
  --agent-command "/path/to/your-agent" --suites general --runs 3

# Raw metrics only; no aggregate score
python3 run_benchmark.py --suites all --score-mode raw

# A live model behind SoraAgent
export OPENAI_API_KEY=...
python3 run_benchmark.py --mode live --provider openai --model gpt-5.4-mini \
  --runs 3 --suites tools autonomy caching

# DeepSeek / Qwen use their normal environment variables
python3 run_benchmark.py --mode live --provider deepseek --model deepseek-flash
python3 run_benchmark.py --mode live --provider qwen --model qwen3.7-plus
```

Reports are written under `results/` by default:

```
results/
  20261112T062050Z/
    report.json       # full machine-readable report
    summary.md        # human-readable report
    metrics.csv       # flat metric table
    runs.jsonl        # one line per run
    cache_requests.jsonl  # only with --include-requests
  latest.json
  latest.md
```

`python3 -m sora_bench` is equivalent to `python3 run_benchmark.py`. The built-in
SoraAgent adapter loads the sibling `../SoraAgent` checkout; the command adapter
does not require SoraAgent to be installed.

### Useful flags

```text
--mode {simulated,live}      simulated (default) or live model
--provider KIND              openai, deepseek, qwen, custom
--model ID                   live model id
--runs N                     repetitions per scenario
--suites a b c               tokens tools memory dreaming speed autonomy caching features general
--out DIR                    report directory
--autonomy-policy P          minimal | balanced | overreach
--cache-ablation             run cache-breaker ablations (default on in simulated)
--no-cache-ablation
--include-requests           dump every provider request and cache record
--keep-workdir               keep per-run temporary directories
--adapter sora|command       built-in SoraAgent harness or an external command
--agent-command COMMAND      JSON stdin/stdout contract for command adapter
--score-mode heuristic|raw   normalize to 0-100, or suppress the aggregate score
--tokenizer heuristic|tiktoken  deterministic estimate or optional model tokenizer
--list-suites
--list-scenarios
```

## Live three-task benchmark

`live_tasks.py` is the separate real-model run.  It uses SoraAgent's actual
provider/tool loop, the built-in core tool pack, auto-approval for benchmark
actions, and a hard total token cap.

```bash
export DEEPSEEK_API_KEY=...
python3 live_tasks.py \
  --provider deepseek --model deepseek-flash \
  --budget-tokens 10000000
```

DeepSeek V4.1 Flash is served under the `deepseek-flash` model id;
`deepseek-v4-flash` is the compatibility alias.

The three tasks are:

| Task | What it exercises | Pass condition |
| --- | --- | --- |
| Web scraping | `find_tools` → `web_extract` on `https://example.com` | Final answer names the H1 and first sentence, and `web_extract` executed |
| Codebase repair | `read_file`, `search_files`, `patch`, `terminal` on a disposable Python package | The seeded bug is fixed, tests pass, and the test file is unchanged |
| Read-only filesystem audit | `search_files` / `read_file` over a disposable tree | TODO count, largest file, deprecated files correct, and the tree hash is unchanged |

Reports go to `results/live-three-tasks/<timestamp>/live_report.json` and
`live_summary.md`, with per-task prompt/completion/cached tokens, tool calls,
duration, checks, and the remaining 10M budget.

---

## Portable command adapter

The `general` suite can score an agent that is not SoraAgent. The command receives
one JSON object on stdin:

```json
{"scenario_id": "general/arithmetic", "task": "...", "requires_tools": false}
```

It must print one JSON object on stdout:

```json
{
  "answer": "...",
  "tool_calls": [],
  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0},
  "latency_ms": 0
}
```

Then run:

```bash
python3 run_benchmark.py --adapter command \
  --agent-command "/path/to/your-agent" --suites general --runs 3
```

SoraAgent-specific suites (`tokens`, `tools`, `memory`, `dreaming`, `speed`,
`autonomy`, `caching`, `features`) require the built-in adapter; the portable
suite deliberately makes no assumptions about tools, memory, or orchestration.

## Modes

### Simulated (default)

A deterministic `ScriptedProvider` follows a per-scenario plan through the real
`Agent` → `AgentRuntime` → `Runner` → `ToolExecutor` path.  It performs lazy
tool discovery, the required `memory_recall` first action, approval checks, and
observation feedback exactly as a cooperative model would.

What this measures:

* framework behavior and overhead independent of model quality;
* token and prompt-cache properties (the request bytes are real);
* memory and dreaming mechanics against deterministic fixtures;
* the benchmark's own scoring sensitivity (e.g. `--autonomy-policy overreach`
  must produce harmful-action findings).

It does **not** measure a model's reasoning.  A perfect tool-use score in
simulated mode means the reference policy and the harness agree, not that a
particular LLM is good.

### Live

The same tasks and ground truth are sent through a real OpenAI-compatible
provider.  The model chooses its own tools and answers.  Every metric then
reflects `model + SoraAgent`.

Live mode needs an API key:

| Provider | Key | Default base URL | Default model |
| --- | --- | --- | --- |
| `openai` | `OPENAI_API_KEY` | OpenAI default | `OPENAI_MODEL` or `gpt-5.4-mini` |
| `deepseek` | `DEEPSEEK_API_KEY` | `https://api.deepseek.com` | `DEEPSEEK_MODEL` or `deepseek-flash` |
| `qwen` | `DASHSCOPE_API_KEY` | DashScope compatible mode | `QWEN_MODEL` or `qwen3.7-plus` |
| `custom` | optional | `SORA_BASE_URL` / `--base-url` | `SORA_CUSTOM_MODEL` |

Live runs cost money.  Start with `--suites tools --runs 1` to validate the
provider, then add suites.

---

## What each suite does

### `tokens`

Runs four representative tasks: minimal (no memory/identity/clock), full
(memory + identity + clock), multi-tool, and no-memory.  The recorder attributes
every request's tokens to system, identity, clock, state, tool schemas, user,
assistant, and tool results.

Key metrics:

* prompt/completion/total/effective tokens per run;
* tokens per successful task and successful tasks per 1k tokens;
* static prompt overhead share and tool-schema share;
* prompt tokens reused by prefix cache.

Effective tokens subtract the prefix-cache-reusable portion from the prompt
before adding completion tokens.  In simulated mode all token counts are
**estimates** from the deterministic estimator in `tokens.py`; live mode also
reports provider-reported usage separately.

### `tools`

Ten scenarios cover read-only lookup, multi-step approval flows, financial and
calendar actions, a policy trap (a tempting but forbidden refund), no-tool
restraint, natural-language discovery, and a vague request that needs a read
plus a CRM update.

The scorer matches actual business calls against expected calls
(order-insensitively, subset arguments) and reports:

* precision / recall / F1 and exact-trajectory rate;
* argument accuracy;
* forbidden-call rate and unnecessary-call rate;
* tool error rate, approval compliance, discovery calls per run;
* first-call correctness and (when memory is enabled) recall-first compliance.

`find_tools` and `memory_recall` are treated as orchestration and excluded from
business precision/recall, but their call counts are reported.

### `memory`

Seeds 14 facts across daily verbatim memory (today through six days ago) and a
long-term backend (10–20 days old), including:

* direct facts, preferences, identifiers, dates, owners, and a Chinese fact;
* a stale/contradictory preference distractor;
* a wrong-codename distractor;
* a no-answer query (the CEO's shoe size).

It measures retrieval directly (`MemoryRecallService.collect`) and end-to-end
through the real `Agent` + `MemoryHook` path.  The reference answer policy picks
the recall bullet with the strongest lexical overlap with the query, breaking
ties by recency; it never sees the expected marker.

Key metrics:

* retrieval hit rate, precision, MRR, forbidden-fact rate;
* no-answer false-positive rate;
* end-to-end answer accuracy, grounding, recall-first compliance;
* memory write-through (does `after_run` persist the new turn?);
* facts per 1k recall tokens, recall payload tokens, signal density,
  corpus-to-recall compression, recall latency.

### `dreaming`

Seeds a controlled corpus containing:

* two exact duplicate groups that should merge;
* one numeric conflict that should be flagged;
* a same-number pair that must **not** conflict;
* a triangle that should produce one two-hop inferred link;
* an expired record that should be removed;
* distinct daily messages that should transfer without creating noise.

It runs the real `DreamRunner` twice (the second run checks idempotency) with an
observable in-memory backend, a no-op backup manager, and a counting wake model.

Key metrics:

* merge / conflict / inference precision, recall, and F1;
* source retention, expiry accuracy, artifact cleanup;
* idempotency and error rate;
* duration per 100 memories, backend calls per memory, throughput, actions per
  memory, wake-model tokens per call.

### `features`

The suite is labelled **Files, specialists, memory, graph & security** in reports.

Deterministic edge-case suite for the current 0.3.x surface:

* **Files**: DOCX, ODF, EPUB, EML, RTF, nested archives, HTML script
  stripping/tables, empty files, malformed DOCX, archive total-byte caps, and
  wide-CSV fallback.
* **Specialists**: manifest round-trip, lexical routing, forbidden
  delegation-control tools, `find_delegates` / `delegate_to_agent`, scoped
  controls (no peer delegation), workspace escape rejection, SQLite daily
  stores, and provisional auto-create quotas/unsafe-tool stripping.
* **Memory modes**: specialist daily transfer with provenance, half-done →
  complete rounds, failed-transfer retry/forget, and daily-only consolidation
  with local lossless archives.
* **Graph/memory**: graph extraction, graph-aware recall facts, native graph
  projection into dream communities, transient inferred-link cleanup, and
  missing-memory HTTP 404 handling.

The suite runs the real SoraAgent implementations and is deterministic in both
simulated and live modes.

### `speed`

Runs a single-tool task, a four-tool task, and a memory-recall task.  In
simulated mode each provider call gets a deterministic 25 ms delay by default so
end-to-end timings are realistic while framework overhead remains visible.

Key metrics: end-to-end mean/p95, provider time, tool time, framework overhead
mean/p95/share, per-step and per-provider-call latency, completion tokens/second,
tasks/second.

### `autonomy`

Six scenarios where the user asks for something narrow but a helpful—or
harmful—extra action is possible.  The simulated policy is selectable:

* `minimal`: requested actions only;
* `balanced` (default): one or two low-risk helpful extras, no harmful ones;
* `overreach`: harmful extras such as cancellation, duplicate refunds, external
  emails, or unrelated deletion.

Every extra action is classified against scenario labels, falling back to the
tool's intrinsic risk (`read_only`, `mutating`, `external`, `destructive`).
Requested actions are excluded from the "extra" bucket.

Key metrics:

* net utility (positive = helpful, negative = harmful; executed only);
* useful-extra precision and useful-opportunity recall;
* harmful-extra attempt and execution rates;
* runs with negative impact;
* requested-task success and approval-gate compliance.

The `minimal` and `overreach` policies are useful self-checks: a good benchmark
must score balanced > minimal > overreach.

### `caching`

Snapshots the exact request prefix (tool schemas first, then messages) for every
provider call and computes:

* **hit rate**: reusable prefix tokens / all prompt tokens;
* **stability**: mean prefix retention penalized by variation;
* **message prefix hit rate and stability** (conversation/static prefix only,
  excluding tool schemas), which isolates ordering from tool-catalog churn;
* **stable prefix tokens**, mean/worst prefix retention, cacheable request
  rate, break events;
* tool-schema churn, clock churn, **clock trailing rate**, identity churn;
* the first divergent component per transition;
* cross-run first-prompt identity;
* provider-reported cache counters when available (`cached_tokens`,
  `prompt_cache_hit_tokens`, `cache_read_input_tokens`, ...).

In simulated mode it also runs an ablation:

| Variant | Memory | Identity | Clock |
| --- | --- | --- | --- |
| `full` | on | on | advancing every turn |
| `no_clock` | on | on | off |
| `fixed_clock` | on | on | fixed |
| `no_identity` | on | off | advancing |
| `no_memory` | off | on | advancing |
| `minimal` | off | off | off |

The ablation makes the usual cache breakers visible.  The clock is now injected
as the **final** message, after the conversation history, so the order is:

```
static system prompt -> identity -> conversation -> state -> clock
```

That means clock changes no longer invalidate prior turns: the `full` and
`fixed_clock` variants should report the same cache hit rate.  The benchmark
also reports a message-only cache hit rate to show this separately from tool
schemas.

The remaining structural breaker is lazy discovery: the `find_tools` schema
embeds a tool catalog that shrinks whenever a tool is loaded, so each load
rebuilds the tool-definition prefix.  The `cache_breakers.recommendation` field
names whichever breaker is still active; it explicitly recognizes a trailing
clock as cache-safe rather than recommending the fix again.

Those are findings, not benchmark bugs.

---

## Scoring

Every suite reports raw metrics and a heuristic 0–100 score.  Targets are
explicit in `sora_bench/config.py` (`TARGETS`) and copied into
`report.json → diagnostics.targets`.  Higher-is-better metrics use
`min(1, value/target)`; lower-is-better metrics use `min(1, target/value)`.
The overall score is the equal-weight mean of the nine suite scores. Set `--score-mode raw` to suppress it.

Scores are intentionally conservative and **not comparable across different
target sets**.  Change `TARGETS` if your deployment has different budgets, and
compare raw metrics across SoraAgent versions with the same targets.

---

## Report schema

`report.json`:

```json
{
  "schema_version": "1.1",
  "mode": "simulated",
  "provider": "simulated",
  "model": "reference",
  "environment": {"python": "...", "sora_agent_version": "..."},
  "score_mode": "heuristic",
  "adapter": "sora",
  "summary": {
    "overall_score": 83.3,
    "grade": "B",
    "suite_scores": {"tools": 100.0, "...": 0.0},
    "headline": {"tool_f1": {"value": 1.0, "score": 100.0}},
    "uncertainty": {"tools": {"n": 20, "success_ci95": [0.8, 1.0]}}
  },
  "suites": {
    "tools": {
      "score": 100.0,
      "metrics": [{"key": "tool_f1", "value": 1.0, "score": 100.0}],
      "diagnostics": {}
    }
  },
  "runs": [{"scenario_id": "tools/po_status", "requests": []}]
}
```

Each run records `tool_call_log` (attempted/executed/skipped), `ledger` (actual
tool side effects), evaluation details, token attribution, timings, and request
records.  `--include-requests` adds the full per-request cache records and a
`cache_requests.jsonl` dump.

---

## Reproducibility and safety

* No network access in simulated mode; no API key needed.
* Every run uses a fresh temporary directory.  `SORA_MEMORY`,
  `SORA_IDENTITY`, `SORA_CALENDAR_DB`, `SORA_CONFIG`, and `SORA_SECRETS` are
  forced to temporary/no-op values unless already set.
* The benchmark never touches `~/.sora_agent` or the developer's real identity,
  calendar, or memory files.
* Seeds control simulated jitter; the memory fixture uses relative day offsets and the dream fixture uses fixed dates.
* `--keep-workdir` preserves per-run temp directories for debugging.
* Live mode sends the scenario tasks to the chosen provider and costs money.

---

## Project layout

```
SoraAgent benchmark/
  run_benchmark.py          # convenient entry point
  sora_bench/
    cli.py                  # argparse CLI
    config.py               # suites, targets, weights
    harness.py              # builds a real Agent run, records everything
    providers.py            # scripted, recording, latency, live cache-aware
    protocol.py             # canonical prompt prefix + cache analysis
    tokens.py               # deterministic token estimator
    backend.py              # observable in-memory CortexDB stand-in
    tools.py                # business tool catalog + side-effect ledger
    scenarios.py            # task, tool, token, speed, autonomy, cache scenarios
    support.py              # clocks, identity, approval, no-op backup/wake
    report.py               # JSON/Markdown/CSV/JSONL writers
    suites/
      tools.py tokens.py memory.py dreaming.py speed.py autonomy.py caching.py
  tests/                    # pytest suite (offline)
```

---

## Tests

```bash
python3 -m pytest tests -q
```

The tests cover token estimation, prefix/cache analysis, call matching, memory
retrieval and write-through, dream ground truth and idempotency, autonomy
discrimination, and a full end-to-end report run.

---

## Measurement quality

- **Replicates:** `--runs N` repeats each scenario. The report includes Wilson
  intervals for success rates and bootstrap intervals for continuous metrics;
  `n=1` is marked as insufficient for an interval.
- **Raw mode:** `--score-mode raw` suppresses the aggregate 0–100 score while
  retaining all suite metrics. Use it when comparing harnesses or when targets
  are not appropriate for your deployment.
- **Token counts:** provider-reported usage is preferred. Simulated runs use a
  deterministic heuristic; `--tokenizer tiktoken` enables a model tokenizer when
  the optional package is installed, and falls back visibly to the heuristic.
- **Scope:** the features suite is SoraAgent-specific by design. The general
  suite is the portable comparison surface; it does not reward SoraAgent's tool
  catalog or memory architecture.

## Limitations

* Simulated mode uses a scripted reference policy, not a model.  Use live mode
  for model-dependent tool/autonomy scores.
* Token counts in simulated mode are estimates from a fixed heuristic, not a
  vendor tokenizer.  Live mode reports provider usage separately.
* Memory retrieval is measured against a deterministic in-memory backend and
  SQLite daily store, not the CortexDB sidecar.  Dreaming uses the real pipeline
  with observable stand-ins for the sidecar, backup, and wake model.
* Prompt-cache accounting assumes a prefix cache over serialized tool schemas +
  messages.  Vendors differ in cache block size, TTL, and message serialization;
  `--min-cache-block` lets you model a minimum block, and live mode reports the
  vendor's actual counters when available.
* The harness runs one task per `Agent.run`; it does not model multi-session
  chat history or gateway delivery.  The `Agent` facade does not stream today,
  so response speed is end-to-end/turn latency rather than time-to-first-token.


## License

The benchmark harness is **MIT-licensed**; see [`LICENSE`](LICENSE). SoraAgent itself is licensed **AGPL-3.0-only**.
