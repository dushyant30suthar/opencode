import fs from "fs/promises"
import { existsSync } from "fs"
import os from "os"
import path from "path"
import { spawn } from "child_process"
import { Effect } from "effect"
import { ModelV2 } from "@opencode-ai/core/model"
import { ProviderV2 } from "@opencode-ai/core/provider"
import type { Info, Model } from "./provider"

export const PROVIDER_ID = "llamastack"

/** Existing llama-stack FastAPI manager (may be down). */
export const MANAGER_URL = "http://127.0.0.1:7860/v1"
/** llama-server router owned by this fork. */
export const ROUTER_URL = "http://127.0.0.1:9337/v1"
/** llama-server binary: $LLAMASTACK_SERVER_BIN, then $PATH, then conventional build locations. */
export const LLAMA_SERVER_BIN = resolveServerBin()
/** Model files in the LM Studio layout; override with $LLAMASTACK_MODELS_DIR. */
export const MODELS_DIR = process.env["LLAMASTACK_MODELS_DIR"] || path.join(os.homedir(), ".lmstudio", "models")

function resolveServerBin(): string {
  const override = process.env["LLAMASTACK_SERVER_BIN"]
  if (override) return override
  const home = os.homedir()
  const candidates = [
    ...(process.env["PATH"] || "").split(path.delimiter).filter(Boolean),
    path.join(home, "Projects", "opencode-llama.cpp", "llama.cpp", "build", "bin"),
    path.join(home, "Projects", "llama", "llama.cpp", "build", "bin"),
    path.join(home, "Projects", "llama.cpp", "build", "bin"),
    path.join(home, "llama.cpp", "build", "bin"),
    "/usr/local/bin",
  ].map((dir) => path.join(dir, "llama-server"))
  return candidates.find((bin) => existsSync(bin)) ?? candidates[candidates.length - 1]
}

const STATE_DIR = path.join(os.homedir(), ".local", "state", "llamastack")

const PROBE_TIMEOUT = 1_500
const STARTUP_TIMEOUT = 10_000
const POLL_INTERVAL = 500

const DEFAULT_CONTEXT = 32_768
// thinking models spend reasoning tokens from the output budget; 8k truncates
// mid-plan on hard tickets. capped at half the context for small-ctx presets.
const DEFAULT_OUTPUT = 32_768

type DiscoveredModel = {
  id: string
  context?: number
}

function parseModels(body: unknown): DiscoveredModel[] {
  if (!body || typeof body !== "object") return []
  const record = body as Record<string, any>
  const data = Array.isArray(record.data) ? record.data : Array.isArray(record.models) ? record.models : []
  const models: DiscoveredModel[] = []
  for (const entry of data) {
    if (!entry || typeof entry !== "object") continue
    const id = typeof entry.id === "string" ? entry.id : typeof entry.name === "string" ? entry.name : undefined
    if (!id) continue
    // the router advertises a synthetic "default" preset with no model file
    if (id === "default" && entry.status && !(entry.status.args ?? []).includes("--model")) continue
    const meta = entry.meta && typeof entry.meta === "object" ? entry.meta : {}
    const context = [entry.context_length, entry.max_context_length, meta.n_ctx_train, meta.n_ctx].find(
      (value): value is number => typeof value === "number" && value > 0,
    )
    models.push({ id, context })
  }
  return models
}

/** Resolves to undefined when the server is unreachable, [] when reachable with no models. */
async function listModels(baseURL: string, timeout: number): Promise<DiscoveredModel[] | undefined> {
  try {
    const res = await fetch(`${baseURL}/models`, { signal: AbortSignal.timeout(timeout) })
    if (!res.ok) return undefined
    return parseModels(await res.json())
  } catch {
    return undefined
  }
}

/**
 * llama-server's --models-dir scan is one level deep, but the LM Studio layout
 * is <models>/<publisher>/<repo>/*.gguf. Cover the nested repos with a
 * generated preset INI, mirroring llama.cpp's own per-directory heuristics
 * (skip mmproj files, prefer the first shard of multi-shard models).
 */
const PRESET_HEADER = [
  "version = 1",
  "",
  "# Per-model settings — YOURS TO EDIT. New models get a default section appended;",
  "# existing sections are never modified or removed by opencode.",
  "# Any llama-server flag works as a key (without leading dashes). Common ones:",
  "#   ctx-size = 32768          context window tokens",
  "#   gpu-layers = 99           layers offloaded to GPU (99 = all)",
  "#   tensor-split = 0.67,0.33  VRAM ratio across the two GPUs",
  "#   cache-type-k = q8_0       KV cache quant: f16 | q8_0 | q4_0",
  "#   cache-type-v = q8_0",
  "#   flash-attn = on",
  "#   temp = 0.7                sampling temperature",
  "#   reasoning-preserve = true keep <think> blocks in multi-turn history —",
  "#                             thinking models loop/repeat tool calls without it",
  "# Changes apply on next model load (restart the router or swap models).",
].join("\n")

/** "Qwen3.6-27B-Q6_K.gguf" in repo "Qwen3.6-27B-GGUF" → "publisher/Qwen3.6-27B:Q6_K". */
export function quantSectionName(publisher: string, repo: string, filename: string): string {
  const repoShort = repo.replace(/-GGUF$/i, "")
  const stem = filename.replace(/\.gguf$/i, "").replace(/-\d{5}-of-\d{5}$/, "")
  const quant = stem.match(/-((?:I?Q\d[\w.]*)|F16|F32|BF16|MXFP4[\w]*)$/i)?.[1]
  return `${publisher}/${repoShort}:${quant ?? stem}`
}

async function generatePresets(): Promise<string | undefined> {
  const preset = path.join(STATE_DIR, "models.ini")
  const existing = await fs.readFile(preset, "utf8").catch(() => "")
  const known = new Set([...existing.matchAll(/^\[([^\]]+)\]/gm)].map((m) => m[1]))
  // files already referenced by any section's model= line are user-owned — never re-seed
  const referenced = new Set([...existing.matchAll(/^\s*model\s*=\s*(.+?)\s*$/gm)].map((m) => m[1]))
  const sections: string[] = []
  const publishers = await fs.readdir(MODELS_DIR, { withFileTypes: true }).catch(() => [])
  for (const publisher of publishers) {
    if (!publisher.isDirectory()) continue
    const publisherDir = path.join(MODELS_DIR, publisher.name)
    const repos = await fs.readdir(publisherDir, { withFileTypes: true }).catch(() => [])
    for (const repo of repos) {
      if (!repo.isDirectory()) continue
      const repoDir = path.join(publisherDir, repo.name)
      const files = await fs.readdir(repoDir, { withFileTypes: true }).catch(() => [])
      let mmproj: string | undefined
      const entries: string[] = []
      for (const file of files) {
        if (!file.isFile() || !file.name.endsWith(".gguf")) continue
        if (file.name.includes("mmproj")) mmproj = file.name
        else if (/-\d{5}-of-\d{5}\.gguf$/.test(file.name)) {
          if (file.name.includes("-00001-of-")) entries.push(file.name)
        } else entries.push(file.name)
      }
      // one section per quant file, so every variant shows up as its own model
      for (const entry of entries) {
        const full = path.join(repoDir, entry)
        if (referenced.has(full)) continue
        const name =
          entries.length === 1 && !known.has(`${publisher.name}/${repo.name}`)
            ? `${publisher.name}/${repo.name}`
            : quantSectionName(publisher.name, repo.name, entry)
        if (known.has(name)) continue
        known.add(name)
        sections.push(
          [
            `[${name}]`,
            `model = ${full}`,
            ...(mmproj ? [`mmproj = ${path.join(repoDir, mmproj)}`] : []),
            // agent workloads need real context; llama-server's 4096 default is unusable.
            // 32k fits alongside a ~20GB Q4 model on 2x16GB with q8_0 KV cache.
            `ctx-size = 32768`,
            `gpu-layers = 99`,
            `flash-attn = on`,
            `cache-type-k = q8_0`,
            `cache-type-v = q8_0`,
            `jinja = true`,
            `cache-ram = 2048`,
            // thinking models loop without their reasoning in history; the flag
            // is capability-gated per template, so it is safe for non-thinkers
            `reasoning-preserve = true`,
          ].join("\n"),
        )
      }
    }
  }
  if (!existing && sections.length === 0) return undefined
  const body = existing.trim().length > 0 ? existing.replace(/\s+$/, "") : PRESET_HEADER
  const content = sections.length > 0 ? [body, "", sections.join("\n\n"), ""].join("\n") : body + "\n"
  await fs.writeFile(preset, content)
  return preset
}

/**
 * Global router settings sidecar (~/.local/state/llamastack/server.json),
 * managed by the TUI's /config screen. Default: exposed on the LAN.
 */
async function routerHost(): Promise<string> {
  try {
    const parsed = JSON.parse(await fs.readFile(path.join(STATE_DIR, "server.json"), "utf8"))
    if (parsed?.expose === false) return "127.0.0.1"
  } catch {
    // missing or malformed — use the default
  }
  return "0.0.0.0"
}

async function spawnRouter(): Promise<boolean> {
  const stat = await fs.stat(LLAMA_SERVER_BIN).catch(() => undefined)
  if (!stat?.isFile()) return false
  await fs.mkdir(STATE_DIR, { recursive: true }).catch(() => {})
  const preset = await generatePresets().catch(() => undefined)
  const log = await fs.open(path.join(STATE_DIR, "router.log"), "a").catch(() => undefined)
  try {
    const child = spawn(
      LLAMA_SERVER_BIN,
      [
        "--models-dir",
        MODELS_DIR,
        ...(preset ? ["--models-preset", preset] : []),
        // swap models instead of stacking them in VRAM (2x16GB fits ~one 30B-class model)
        "--models-max",
        "1",
        "--host",
        await routerHost(),
        "--port",
        "9337",
      ],
      {
        detached: true,
        stdio: ["ignore", log?.fd ?? "ignore", log?.fd ?? "ignore"],
      },
    )
    child.on("error", () => {})
    child.unref()
    // pidfile lets the TUI restart the detached router when settings change
    if (child.pid) await fs.writeFile(path.join(STATE_DIR, "router.pid"), `${child.pid}\n`).catch(() => {})
    return true
  } catch {
    return false
  } finally {
    await log?.close().catch(() => {})
  }
}

async function awaitRouter(): Promise<DiscoveredModel[] | undefined> {
  const deadline = Date.now() + STARTUP_TIMEOUT
  while (Date.now() < deadline) {
    const models = await listModels(ROUTER_URL, PROBE_TIMEOUT)
    // reachable with no models resolves immediately; the provider is absent
    if (models) return models.length > 0 ? models : undefined
    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL))
  }
  return undefined
}

function displayName(id: string) {
  const base = id.split("/").filter(Boolean).at(-1) ?? id
  return base.replace(/\.gguf$/i, "")
}

function toModel(providerID: ProviderV2.ID, baseURL: string, discovered: DiscoveredModel): Model {
  const context = discovered.context ?? DEFAULT_CONTEXT
  return {
    id: ModelV2.ID.make(discovered.id),
    providerID,
    name: displayName(discovered.id),
    family: "",
    api: {
      id: discovered.id,
      url: baseURL,
      npm: "@ai-sdk/openai-compatible",
    },
    status: "active",
    headers: {},
    options: {},
    cost: { input: 0, output: 0, cache: { read: 0, write: 0 } },
    limit: { context, output: Math.min(DEFAULT_OUTPUT, Math.floor(context / 2)) },
    capabilities: {
      temperature: true,
      reasoning: false,
      attachment: false,
      toolcall: true,
      input: { text: true, audio: false, image: false, video: false, pdf: false },
      output: { text: true, audio: false, image: false, video: false, pdf: false },
      interleaved: false,
    },
    release_date: "",
    variants: {},
  }
}

/**
 * The router's model listing has no context metadata for unloaded models, so
 * without this the models would advertise the 32k fallback and opencode would
 * compact conversations far below the real window. models.ini is the source of
 * truth for what ctx-size a model actually loads with.
 */
async function presetContexts(): Promise<Record<string, number>> {
  const result: Record<string, number> = {}
  const text = await fs.readFile(path.join(STATE_DIR, "models.ini"), "utf8").catch(() => "")
  let section = ""
  for (const line of text.split("\n")) {
    const header = line.match(/^\s*\[([^\]]+)\]\s*$/)
    if (header) {
      section = header[1]
      continue
    }
    const ctx = line.match(/^\s*ctx-size\s*=\s*(\d+)\s*$/)
    if (ctx && section) result[section] = Number.parseInt(ctx[1], 10)
  }
  return result
}

async function toInfo(baseURL: string, discovered: DiscoveredModel[]): Promise<Info> {
  const providerID = ProviderV2.ID.llamastack
  const contexts = await presetContexts()
  const models: Record<string, Model> = {}
  for (const entry of discovered) {
    models[entry.id] = toModel(providerID, baseURL, { ...entry, context: contexts[entry.id] ?? entry.context })
  }
  return {
    id: providerID,
    name: "Llama Stack (local)",
    source: "custom",
    env: [],
    // The local server takes any key; the SDK just needs one to be present.
    options: { baseURL, apiKey: "llamastack" },
    models,
  }
}

async function run(): Promise<Info | undefined> {
  if (process.env["OPENCODE_DISABLE_LLAMASTACK"]) return undefined
  let routerReachable = false
  for (const baseURL of [MANAGER_URL, ROUTER_URL]) {
    const models = await listModels(baseURL, PROBE_TIMEOUT)
    if (models === undefined) continue
    if (models.length > 0) return await toInfo(baseURL, models)
    if (baseURL === ROUTER_URL) routerReachable = true
  }
  // router already up but empty; nothing to gain from spawning another one
  if (routerReachable) return undefined
  if (!(await spawnRouter())) return undefined
  const models = await awaitRouter()
  if (!models) return undefined
  return await toInfo(ROUTER_URL, models)
}

let detection: Promise<Info | undefined> | undefined

/**
 * Detect a local llama.cpp server, spawning the router when needed.
 * Never fails; resolves to undefined when no server is available.
 * Memoized so the process only probes (and spawns) once.
 */
export const detect = (): Effect.Effect<Info | undefined> =>
  Effect.promise(() => (detection ??= run().catch(() => undefined)))

export * as LlamaStack from "./llamastack"
