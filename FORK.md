# The opencode-llama.cpp fork

This is opencode with a built-in, zero-config local-model stack for this machine
(2× RTX 5060 Ti 16GB). One binary: launch `opencode`, local models are in the
model picker, no config files, env vars, auth entries, or scripts. Everything
below lives on the `opencode-llama.cpp` branch on top of upstream `master`.

## Features

### Zero-config llamastack provider

On startup the provider probes for a local OpenAI-compatible server, in order:

1. llama-stack FastAPI manager on `http://127.0.0.1:7860/v1` (may be down; fine)
2. llama-server router on `http://127.0.0.1:9337/v1`
3. If neither answers and `llama-server` exists on disk, it spawns the router
   itself — detached (survives opencode exit), router mode with `--models-max 1`
   so models swap in VRAM instead of stacking, logging to
   `~/.local/state/llamastack/router.log` — and polls `/v1/models` for up to 10s.

Discovered models appear in `/models` under "Llama Stack (local)" with tool
calling and web search enabled, zero cost, 32k default context (or the context
the server advertises). Detection is memoized per process and never fails the
startup path. (`packages/opencode/src/provider/llamastack.ts`)

### Model discovery from LM Studio's folder

Models are read from `~/.lmstudio/models` in the LM Studio layout
(`<publisher>/<repo>/*.gguf`) — anything you download in LM Studio appears in
opencode automatically. Because llama-server's own `--models-dir` scan is one
level deep, the provider generates a preset INI
(`~/.local/state/llamastack/models.ini`, passed as `--models-preset`) covering
the nested repos, mirroring llama.cpp's heuristics: `mmproj` files are attached
as multimodal projectors, the first shard of multi-shard models is used.

New models get a default section appended (`ctx-size = 32768`, `gpu-layers = 99`,
`flash-attn = on`, `cache-type-k/v = q8_0`, `jinja = true`, `cache-ram = 2048`,
`reasoning-preserve = true`); existing sections are never modified or removed —
the file is yours to edit.

### Preserve thinking (the tool-call-loop fix)

Thinking models (Qwen3.6) strip previous-turn `<think>` blocks from the prompt
by default; mid-task the model forgets *why* it called the last three tools and
starts looping — repeating tool calls, sometimes with empty `{}` arguments.
Two halves make the round-trip work, and both are in place:

1. opencode already sends `reasoning_content` back on assistant messages
   (`@ai-sdk/openai-compatible` emits it; parts are replayed from the session DB
   each step).
2. `reasoning-preserve = true` in `models.ini` makes llama-server's template
   keep that reasoning for **all** history turns (Qwen3.6 calls this
   `preserve_thinking` and recommends it for agents). Support is auto-detected
   per template — llama-server logs "chat template supports preserving
   reasoning" when available — and the key is harmless for non-thinking models.

Side benefit: with thinking preserved, the rendered prompt is append-only across
turns, so the KV prefix cache stays hot instead of re-evaluating the tail of the
conversation every step.

Sampling for llamastack models is **owned by `models.ini`** (`temp`, `top-p`,
`top-k`, `min-p` keys — the Qwen3.6 sections use the official "precise coding"
values 0.6/0.95/20/0). opencode deliberately sends no sampler overrides for this
provider (`provider/transform.ts`), so the `/config` temperature field and any
hand-edited INI values actually take effect.

### `/config` — in-TUI model settings screen

Type `/config` in the TUI (command palette: "Configure local models"). Flow:

1. **Pick a model** — the list is the union of `models.ini` sections and
   models discovered in `~/.lmstudio/models` (marked "(defaults)" if they have
   no section yet). A **Server** entry at the top shows the endpoint URL and
   the loaded model's parameters (see below).
2. **Edit settings** — the LM Studio basics, pre-filled with current values and
   marked "(default)" when they match the defaults:
   - Context window (`ctx-size`) — default 32768
   - GPU layers (`gpu-layers`) — default 99 (all)
   - Tensor split GPU 0 / GPU 1 (`tensor-split`, e.g. `0.5,0.5`) — empty = auto
   - KV cache quant K and V (`cache-type-k/v`: `f16` | `q8_0` | `q4_0`) — default `q8_0`
   - Flash attention (on/off, toggles in place) — default on
   - Temperature (`temp`) — empty = server default
3. **Save** — writes the section back to `models.ini` (comments, unmanaged keys
   like `model`/`mmproj`/`jinja`, and other sections are preserved verbatim),
   then asks the router to reload presets (`GET /models?reload=1`, which also
   unloads a running model whose preset changed so the next request picks up the
   new settings). If the router is down it just saves — settings apply on next
   load.

Esc steps back one screen; numeric/split inputs are validated. If `models.ini`
does not exist, the screen still opens and creates it on save with the same
defaults as the provider.

### Model-load progress

Loading a 30B-class model into VRAM takes a while. While a request is in flight
on a llamastack model, the TUI subscribes to the router's `GET /models/sse`
stream and shows llama-server's real-time load progress next to the busy
spinner — stage and percent when available (e.g. `loading Qwen3.6-27B-GGUF
text_model 42%`), nothing otherwise. The subscription is opened only while a
request is pending, closed when it finishes, and degrades silently if the
router is down or predates the endpoint.

### LAN hosting + server status

The router is **exposed on the network by default** (`--host 0.0.0.0`, port
9337) so other machines on the home LAN can use it as an OpenAI-compatible
endpoint. **Security note: there is no authentication — anyone on the LAN can
use the endpoint while it is exposed.** Toggle it under `/config` → Server →
"Expose on network"; the setting persists in
`~/.local/state/llamastack/server.json` and saving restarts the detached router
(via `router.pid`) so the binding changes immediately. If the running router
predates the pidfile it degrades to "restart the router to apply".

Status display: the session sidebar gets a **Local server** panel (llamastack
models only) showing the endpoint URL with the machine's LAN IP when exposed
(e.g. `http://192.168.1.49:9337/v1`, `127.0.0.1` when localhost-only) and the
loaded model's key parameters (`ctx`, `ngl`, `split`, `kv`), sourced from the
router's `GET /models` listing with `models.ini` as fallback, refreshed every
10s. The same information appears under the Server entry at the top of
`/config`.

### Web search for local models

The built-in `websearch` tool is enabled for the llamastack provider
(`packages/opencode/src/tool/registry.ts`), so local models can search the web
like opencode-gateway models can.

### No self-update

The fork's version (`0.0.0-llamastack-TIMESTAMP`) always compares older than
upstream releases, so stock opencode would prompt to "update" — and overwrite
this binary with the upstream release. Fork builds (channel or version
containing `llamastack`) are exempt: no update prompt at launch, and both
`opencode upgrade` and the server upgrade endpoint refuse with
`custom llamastack build — update by rebuilding the fork`.

## Files & ports

| What                    | Where                                                        |
| ----------------------- | ------------------------------------------------------------ |
| Per-model settings      | `~/.local/state/llamastack/models.ini` (user-owned, editable) |
| Global server settings  | `~/.local/state/llamastack/server.json` (`{"expose": true}`)  |
| Router log              | `~/.local/state/llamastack/router.log`                        |
| Router pidfile          | `~/.local/state/llamastack/router.pid`                        |
| Router port             | `9337` (`/v1` OpenAI-compatible; `/models`, `/models/sse` router API) |
| Models directory        | `$LLAMASTACK_MODELS_DIR`, else `~/.lmstudio/models` (LM Studio layout) |
| llama-server binary     | `$LLAMASTACK_SERVER_BIN`, else first hit on `$PATH`, else conventional build dirs (`~/Projects/llama/llama.cpp/build/bin`, `~/Projects/llama.cpp/build/bin`, `~/llama.cpp/build/bin`, `/usr/local/bin`) |
| Escape hatch            | `OPENCODE_DISABLE_LLAMASTACK=1` skips detection entirely      |

## The engine (llama.cpp)

The router is `llama-server` from `~/Projects/llama/llama.cpp/build/bin/`,
build **b9891 (f36e5c348)**, built 2026-07-07 with **CUDA 13.3.1**
(nvcc V13.3.73), host compiler **g++-15** (`gcc15-c++` package — nvcc supports
at most GCC 15). CMake configuration: `GGML_CUDA=ON`,
`CMAKE_CUDA_ARCHITECTURES=120` (→ 120a, Blackwell), `GGML_CUDA_FA_ALL_QUANTS=ON`,
`GGML_CUDA_GRAPHS=ON`, `GGML_CUDA_COMPRESSION_MODE=speed`, `GGML_LTO=ON`,
`GGML_NATIVE=ON`, Ninja, Release.

> **CRITICAL: never rebuild with CUDA 13.2/13.2.1.** Confirmed NVIDIA compiler
> bug miscompiles the quantization kernels and produces gibberish output
> (llama.cpp issue #21255). Use 13.3+ only. A rebuild needs `rm -rf build`
> first (LTO caches stale objects).

Benchmarks on this build (dual-GPU, `-ngl 99 -fa 1`):

| Model                       | pp512      | tg128     |
| --------------------------- | ---------- | --------- |
| Qwen3.6-35B-A3B Q4_K_M      | 2500 t/s   | 131 t/s   |
| Qwen3.6-27B Q4_K_M          | 906 t/s    | 23.4 t/s  |

## Tuned per-model limits (2026-07-07 experiments)

Binary-searched the max loadable context per model (ngl 99, flash-attn, q8_0 KV,
validated by load + 1-token generation), plus a split-mode/ubatch bench sweep on
the 35B. `/config` → "Reset to recommended" restores these.

| Model | Max ctx | Split | ubatch | Generation | Prompt |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B Q4_K_M | 245,760 | tensor | 2048 | 152 t/s | 2,522 t/s |
| **Qwen3.6-27B-MTP UD-Q4_K_XL** | 180,224 | tensor | 2048 | **70.6 t/s** (draft-mtp n=4, q8 KV) | — |
| Qwen3.6-27B Q4_K_M | 180,224 | tensor | 2048 | 40 t/s (23 on layer) | 724 t/s |
| gemma-4-31B QAT Q4_0 | 147,456 | tensor | 512 | 37 t/s | 745 t/s |

Maxima found by binary search through llama-server itself — vision projector (mmproj)
loaded, exact production flags, success = a real chat completion. Bare-model probes
overestimate by ~60-80k: the ~900MB mmproj and tensor-mode buffers are real. t/s from
llama-bench at the same split/ubatch (tg128/pp2048). Tensor split boosts dense-model
generation dramatically (+72% on the 27B) and the MoE by +14%.

Generation-first by user preference: tensor split = +14% generation for −30%
prompt speed (still 2,500+ t/s).

Sweep findings (35B): ub1024 = best prompt speed (+6% over ub512); ub2048 regresses.
`-sm tensor` = 152 t/s generation (+14%) but −30% prompt speed — documented in the
models.ini header as the chat profile. `-sm row` unsupported (no GPU P2P over the
PCH x4 link). Alternative profiles (chat / speed with ngram-mod speculation) are
documented in the models.ini header comments.

## Maintaining the fork

Branch `opencode-llama.cpp` on top of upstream `master`. To update:

```sh
git fetch upstream
git rebase upstream/master
bun install --ignore-scripts
bun run --cwd packages/core fix-node-pty
cd packages/opencode && bun run script/build.ts --single --skip-embed-web-ui
cp dist/opencode-linux-x64/bin/opencode ~/.opencode/bin/opencode
```

A stock binary backup lives at `~/.opencode/bin/opencode-upstream-backup`.
