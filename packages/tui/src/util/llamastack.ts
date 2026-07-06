import fs from "fs/promises"
import os from "os"
import path from "path"

/**
 * Helpers for the llamastack provider's per-model settings file
 * (~/.local/state/llamastack/models.ini, a llama.cpp preset INI).
 * Paths and default sections mirror packages/opencode/src/provider/llamastack.ts —
 * keep the two in sync.
 */

export const MODELS_DIR = "/home/dushyant30suthar/.lmstudio/models"
export const ROUTER_BASE = "http://127.0.0.1:9337"

const STATE_DIR = path.join(os.homedir(), ".local", "state", "llamastack")
export const PRESET_PATH = path.join(STATE_DIR, "models.ini")

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
