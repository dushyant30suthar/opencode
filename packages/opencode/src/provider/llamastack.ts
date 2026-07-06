import fs from "fs/promises"
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
export const LLAMA_SERVER_BIN = "/home/dushyant30suthar/Projects/llama/llama.cpp/build/bin/llama-server"
export const MODELS_DIR = "/home/dushyant30suthar/.lmstudio/models"

const STATE_DIR = path.join(os.homedir(), ".local", "state", "llamastack")

const PROBE_TIMEOUT = 1_500
const STARTUP_TIMEOUT = 10_000
const POLL_INTERVAL = 500

const DEFAULT_CONTEXT = 32_768
const DEFAULT_OUTPUT = 8_192

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
async function generatePresets(): Promise<string | undefined> {
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
      let model: string | undefined
      let shard: string | undefined
      let mmproj: string | undefined
      for (const file of files) {
        if (!file.isFile() || !file.name.endsWith(".gguf")) continue
        if (file.name.includes("mmproj")) mmproj = file.name
        else if (file.name.includes("-00001-of-")) shard = file.name
        else model = file.name
      }
      const entry = shard ?? model
      if (!entry) continue
      sections.push(
        [
          `[${publisher.name}/${repo.name}]`,
          `model = ${path.join(repoDir, entry)}`,
          ...(mmproj ? [`mmproj = ${path.join(repoDir, mmproj)}`] : []),
        ].join("\n"),
      )
    }
  }
  if (sections.length === 0) return undefined
  const preset = path.join(STATE_DIR, "models.ini")
  await fs.writeFile(preset, ["version = 1", "", sections.join("\n\n"), ""].join("\n"))
  return preset
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
        "--host",
        "127.0.0.1",
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
    limit: { context, output: Math.min(DEFAULT_OUTPUT, context) },
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

function toInfo(baseURL: string, discovered: DiscoveredModel[]): Info {
  const providerID = ProviderV2.ID.llamastack
  const models: Record<string, Model> = {}
  for (const entry of discovered) {
    models[entry.id] = toModel(providerID, baseURL, entry)
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
    if (models.length > 0) return toInfo(baseURL, models)
    if (baseURL === ROUTER_URL) routerReachable = true
  }
  // router already up but empty; nothing to gain from spawning another one
  if (routerReachable) return undefined
  if (!(await spawnRouter())) return undefined
  const models = await awaitRouter()
  if (!models) return undefined
  return toInfo(ROUTER_URL, models)
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
