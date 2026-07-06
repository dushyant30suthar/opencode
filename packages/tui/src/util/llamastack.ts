import fs from "fs/promises"
import os from "os"
import path from "path"
import { spawn } from "child_process"
import { createSignal } from "solid-js"

/**
 * Helpers for the llamastack provider's per-model settings file
 * (~/.local/state/llamastack/models.ini, a llama.cpp preset INI).
 * Paths and default sections mirror packages/opencode/src/provider/llamastack.ts —
 * keep the two in sync.
 */

export const MODELS_DIR = "/home/dushyant30suthar/.lmstudio/models"
export const LLAMA_SERVER_BIN = "/home/dushyant30suthar/Projects/llama/llama.cpp/build/bin/llama-server"
export const ROUTER_PORT = 9337
export const ROUTER_BASE = `http://127.0.0.1:${ROUTER_PORT}`

const STATE_DIR = path.join(os.homedir(), ".local", "state", "llamastack")
export const PRESET_PATH = path.join(STATE_DIR, "models.ini")
export const SERVER_SETTINGS_PATH = path.join(STATE_DIR, "server.json")
export const PID_PATH = path.join(STATE_DIR, "router.pid")

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
  "# Changes apply on next model load (restart the router or swap models).",
]

export type IniSection = {
  name: string
  /** Verbatim body lines (comments, blanks, key = value) — order preserved. */
  lines: string[]
}

export type IniDocument = {
  /** Verbatim lines before the first section header. */
  prelude: string[]
  sections: IniSection[]
}

/** Line-preserving INI parse: everything that is not a `[section]` header is kept verbatim. */
export function parseIni(text: string): IniDocument {
  const prelude: string[] = []
  const sections: IniSection[] = []
  let current: IniSection | undefined
  for (const line of text.split(/\r?\n/)) {
    const header = line.match(/^\s*\[([^\]]+)\]\s*$/)
    if (header) {
      current = { name: header[1], lines: [] }
      sections.push(current)
      continue
    }
    if (current) current.lines.push(line)
    else prelude.push(line)
  }
  return { prelude, sections }
}

export function serializeIni(doc: IniDocument): string {
  const lines = [...doc.prelude]
  for (const section of doc.sections) {
    lines.push(`[${section.name}]`, ...section.lines)
  }
  return lines.join("\n")
}

function keyPattern(key: string) {
  const escaped = key.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
  return new RegExp(`^\\s*${escaped}\\s*=\\s*(.*?)\\s*$`)
}

export function getValue(section: IniSection, key: string): string | undefined {
  const pattern = keyPattern(key)
  for (const line of section.lines) {
    const match = line.match(pattern)
    if (match) return match[1]
  }
  return undefined
}

/** Replace, insert, or (with undefined) remove a `key = value` line; other lines stay verbatim. */
export function setValue(section: IniSection, key: string, value: string | undefined): void {
  const pattern = keyPattern(key)
  const index = section.lines.findIndex((line) => pattern.test(line))
  if (value === undefined) {
    if (index >= 0) section.lines.splice(index, 1)
    return
  }
  const line = `${key} = ${value}`
  if (index >= 0) {
    section.lines[index] = line
    return
  }
  // insert after the last non-blank line so trailing blank separators stay at the end
  let insert = section.lines.length
  while (insert > 0 && section.lines[insert - 1].trim() === "") insert--
  section.lines.splice(insert, 0, line)
}

export type LocalModelFile = {
  name: string
  model: string
  mmproj?: string
}

/**
 * Scan the LM Studio layout (<models>/<publisher>/<repo>/*.gguf) the same way
 * generatePresets() in the llamastack provider does: skip mmproj files, prefer
 * the first shard of multi-shard models.
 */
export async function discoverLocalModels(): Promise<LocalModelFile[]> {
  const result: LocalModelFile[] = []
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
      result.push({
        name: `${publisher.name}/${repo.name}`,
        model: path.join(repoDir, entry),
        ...(mmproj ? { mmproj: path.join(repoDir, mmproj) } : {}),
      })
    }
  }
  return result
}

/** Managed-key defaults for models without a models.ini section — same as generatePresets(). */
export const DEFAULT_MODEL_SETTINGS: Record<string, string> = {
  "ctx-size": "32768",
  "gpu-layers": "99",
  "flash-attn": "on",
  "cache-type-k": "q8_0",
  "cache-type-v": "q8_0",
}

/** Default section for a newly configured model — same defaults as generatePresets(). */
export function defaultSection(discovered: LocalModelFile): IniSection {
  return {
    name: discovered.name,
    lines: [
      `model = ${discovered.model}`,
      ...(discovered.mmproj ? [`mmproj = ${discovered.mmproj}`] : []),
      `ctx-size = 32768`,
      `gpu-layers = 99`,
      `flash-attn = on`,
      `cache-type-k = q8_0`,
      `cache-type-v = q8_0`,
      `jinja = true`,
    ],
  }
}

export async function loadPresets(): Promise<IniDocument> {
  const text = await fs.readFile(PRESET_PATH, "utf8").catch(() => "")
  if (text.trim().length === 0) return { prelude: [...PRESET_HEADER], sections: [] }
  return parseIni(text)
}

export async function savePresets(doc: IniDocument): Promise<void> {
  await fs.mkdir(STATE_DIR, { recursive: true })
  let text = serializeIni(doc)
  if (!text.endsWith("\n")) text += "\n"
  await fs.writeFile(PRESET_PATH, text)
}

export type ModelStatusEvent = {
  model: string
  status?: string
  /** Load stage reported by llama-server (e.g. "text_model", "mmproj_model"). */
  stage?: string
  /** Load progress 0-100 within the current stage. */
  percent?: number
}

/**
 * Stream model status events from the router's GET /models/sse endpoint
 * (`data: {"model": "...", "event": "model_status", "data": {"status": "loading",
 * "progress": {"stages": [...], "current": "text_model", "value": 0.5}}}`).
 * Best effort: silently does nothing when the router is down or predates the
 * endpoint. Returns an unsubscribe function that closes the connection.
 */
export function subscribeModelStatus(onEvent: (event: ModelStatusEvent) => void): () => void {
  const controller = new AbortController()
  void (async () => {
    try {
      const res = await fetch(`${ROUTER_BASE}/models/sse`, { signal: controller.signal })
      if (!res.ok || !res.body) return
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ""
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        let boundary
        while ((boundary = buffer.indexOf("\n\n")) >= 0) {
          const frame = buffer.slice(0, boundary)
          buffer = buffer.slice(boundary + 2)
          for (const line of frame.split("\n")) {
            if (!line.startsWith("data:")) continue
            try {
              const payload = JSON.parse(line.slice(5).trim())
              if (payload?.event !== "model_status" || typeof payload.model !== "string") continue
              const data = payload.data && typeof payload.data === "object" ? payload.data : {}
              const progress = data.progress && typeof data.progress === "object" ? data.progress : undefined
              onEvent({
                model: payload.model,
                status: typeof data.status === "string" ? data.status : undefined,
                stage: typeof progress?.current === "string" ? progress.current : undefined,
                percent: typeof progress?.value === "number" ? Math.round(progress.value * 100) : undefined,
              })
            } catch {
              // malformed frame — skip
            }
          }
        }
      }
    } catch {
      // router down, aborted, or endpoint missing — degrade to nothing
    }
  })()
  return () => controller.abort()
}

/**
 * Ask the router to re-read the preset INI. llama-server's reload diffs the new
 * presets against running models and unloads any whose settings changed, so the
 * next request picks up the new configuration. Best effort: false when the
 * router is down or the request fails.
 */
export async function reloadRouter(): Promise<boolean> {
  try {
    const res = await fetch(`${ROUTER_BASE}/models?reload=1`, { signal: AbortSignal.timeout(3_000) })
    return res.ok
  } catch {
    return false
  }
}

export type ServerSettings = {
  /** Bind the router to 0.0.0.0 so other machines on the LAN can use it. */
  expose: boolean
}

/** Global router settings sidecar (~/.local/state/llamastack/server.json). Default: exposed. */
export async function loadServerSettings(): Promise<ServerSettings> {
  try {
    const parsed = JSON.parse(await fs.readFile(SERVER_SETTINGS_PATH, "utf8"))
    return { expose: typeof parsed?.expose === "boolean" ? parsed.expose : true }
  } catch {
    return { expose: true }
  }
}

const [serverSettingsSignal, setServerSettingsSignal] = createSignal<ServerSettings>()
let serverSettingsRequested = false

/** Reactive view of the server settings; lazily loaded from disk on first read. */
export function serverSettings(): ServerSettings | undefined {
  if (!serverSettingsRequested) {
    serverSettingsRequested = true
    void loadServerSettings().then(setServerSettingsSignal)
  }
  return serverSettingsSignal()
}

export async function saveServerSettings(settings: ServerSettings): Promise<void> {
  await fs.mkdir(STATE_DIR, { recursive: true })
  await fs.writeFile(SERVER_SETTINGS_PATH, JSON.stringify(settings, null, 2) + "\n")
  setServerSettingsSignal(settings)
}

/** First non-internal IPv4 address, e.g. the machine's LAN IP. */
export function lanAddress(): string | undefined {
  for (const entries of Object.values(os.networkInterfaces())) {
    for (const entry of entries ?? []) {
      if (entry.family === "IPv4" && !entry.internal) return entry.address
    }
  }
  return undefined
}

export function endpointURL(expose: boolean): string {
  const host = expose ? (lanAddress() ?? "127.0.0.1") : "127.0.0.1"
  return `http://${host}:${ROUTER_PORT}/v1`
}

async function routerPid(): Promise<number | undefined> {
  const raw = await fs.readFile(PID_PATH, "utf8").catch(() => "")
  const pid = Number.parseInt(raw.trim(), 10)
  if (!Number.isInteger(pid) || pid <= 0) return undefined
  try {
    process.kill(pid, 0)
    return pid
  } catch {
    return undefined
  }
}

/** Mirrors spawnRouter in the llamastack provider (which generates the preset INI first). */
async function spawnRouterProcess(expose: boolean): Promise<boolean> {
  const stat = await fs.stat(LLAMA_SERVER_BIN).catch(() => undefined)
  if (!stat?.isFile()) return false
  await fs.mkdir(STATE_DIR, { recursive: true }).catch(() => {})
  const preset = await fs
    .stat(PRESET_PATH)
    .then((item) => item.isFile())
    .catch(() => false)
  const log = await fs.open(path.join(STATE_DIR, "router.log"), "a").catch(() => undefined)
  try {
    const child = spawn(
      LLAMA_SERVER_BIN,
      [
        "--models-dir",
        MODELS_DIR,
        ...(preset ? ["--models-preset", PRESET_PATH] : []),
        "--models-max",
        "1",
        "--host",
        expose ? "0.0.0.0" : "127.0.0.1",
        "--port",
        `${ROUTER_PORT}`,
      ],
      {
        detached: true,
        stdio: ["ignore", log?.fd ?? "ignore", log?.fd ?? "ignore"],
      },
    )
    child.on("error", () => {})
    child.unref()
    if (child.pid) await fs.writeFile(PID_PATH, `${child.pid}\n`).catch(() => {})
    return true
  } catch {
    return false
  } finally {
    await log?.close().catch(() => {})
  }
}

/**
 * Restart the detached router so a host-binding change takes effect. Kills the
 * process recorded in the pidfile, waits for it to exit, and spawns a fresh one.
 * "not-running": nothing reachable on the port — settings apply when the provider
 * next spawns the router. "failed": reachable but not restartable (no/stale pidfile).
 */
export async function restartRouter(expose: boolean): Promise<"restarted" | "not-running" | "failed"> {
  const reachable = await fetch(`${ROUTER_BASE}/models`, { signal: AbortSignal.timeout(1_500) })
    .then((res) => res.ok)
    .catch(() => false)
  if (!reachable) return "not-running"
  const pid = await routerPid()
  if (!pid) return "failed"
  try {
    process.kill(pid, "SIGTERM")
  } catch {
    return "failed"
  }
  const deadline = Date.now() + 5_000
  while (Date.now() < deadline) {
    try {
      process.kill(pid, 0)
    } catch {
      break // exited
    }
    await new Promise((resolve) => setTimeout(resolve, 200))
  }
  await fs.rm(PID_PATH, { force: true }).catch(() => {})
  return (await spawnRouterProcess(expose)) ? "restarted" : "failed"
}

export type LoadedModel = {
  name: string
  status: string
  /** llama-server flags of the child process, keyed without leading dashes. */
  args: Record<string, string>
}

/** The model currently loaded (or loading) per the router's GET /models listing. */
export async function fetchLoadedModel(): Promise<LoadedModel | undefined> {
  try {
    const res = await fetch(`${ROUTER_BASE}/models`, { signal: AbortSignal.timeout(1_500) })
    if (!res.ok) return undefined
    const body = (await res.json()) as any
    const data = Array.isArray(body?.data) ? body.data : Array.isArray(body?.models) ? body.models : []
    for (const entry of data) {
      const status = entry?.status
      if (!status || (status.value !== "loaded" && status.value !== "loading")) continue
      const name = typeof entry.id === "string" ? entry.id : typeof entry.name === "string" ? entry.name : undefined
      if (!name) continue
      const argv: unknown[] = Array.isArray(status.args) ? status.args : []
      const args: Record<string, string> = {}
      for (let i = 0; i < argv.length; i++) {
        const arg = argv[i]
        if (typeof arg !== "string" || !arg.startsWith("--")) continue
        const next = argv[i + 1]
        if (typeof next === "string" && !next.startsWith("--")) {
          args[arg.slice(2)] = next
          i++
        } else {
          args[arg.slice(2)] = "on"
        }
      }
      return { name, status: status.value, args }
    }
  } catch {
    // router down — caller falls back to models.ini
  }
  return undefined
}

/** Compact one-line summary of the load params the /config screen manages. */
export function formatLoadParams(get: (key: string) => string | undefined): string {
  const parts: string[] = []
  const ctx = get("ctx-size")
  if (ctx) parts.push(`ctx ${ctx}`)
  const gpu = get("gpu-layers") ?? get("n-gpu-layers")
  if (gpu) parts.push(`ngl ${gpu}`)
  const split = get("tensor-split")
  if (split) parts.push(`split ${split}`)
  const k = get("cache-type-k")
  const v = get("cache-type-v")
  if (k || v) parts.push(`kv ${k ?? "f16"}/${v ?? "f16"}`)
  return parts.join(" · ")
}
