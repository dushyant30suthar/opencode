import { describe, expect, test } from "bun:test"
import { parseIni, serializeIni, getValue, setValue } from "../../src/util/llamastack"

const FIXTURE = [
  "version = 1",
  "",
  "# Per-model settings — YOURS TO EDIT.",
  "",
  "[unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF]",
  "model = /home/user/.lmstudio/models/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/model.gguf",
  "mmproj = /home/user/.lmstudio/models/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/mmproj.gguf",
  "ctx-size = 32768",
  "gpu-layers = 99",
  "flash-attn = on",
  "cache-type-k = q8_0",
  "cache-type-v = q8_0",
  "jinja = true",
  "",
  "[ggml-org/gemma-3-4b-it-GGUF]",
  "# hand-written comment the tooling must not eat",
  "model = /home/user/.lmstudio/models/ggml-org/gemma-3-4b-it-GGUF/model.gguf",
  "temp = 0.7",
  "",
].join("\n")

describe("llamastack models.ini", () => {
  test("parse and serialize round-trips verbatim", () => {
    expect(serializeIni(parseIni(FIXTURE))).toBe(FIXTURE)
  })

  test("parses prelude, sections and values", () => {
    const doc = parseIni(FIXTURE)
    expect(doc.prelude[0]).toBe("version = 1")
    expect(doc.sections.map((section) => section.name)).toEqual([
      "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
      "ggml-org/gemma-3-4b-it-GGUF",
    ])
    expect(getValue(doc.sections[0], "ctx-size")).toBe("32768")
    expect(getValue(doc.sections[1], "temp")).toBe("0.7")
    expect(getValue(doc.sections[1], "ctx-size")).toBeUndefined()
  })

  test("edit preserves unmanaged keys, comments and other sections", () => {
    const doc = parseIni(FIXTURE)
    const section = doc.sections[0]
    setValue(section, "ctx-size", "65536")
    setValue(section, "tensor-split", "0.6,0.4")
    setValue(section, "temp", undefined) // absent — no-op
    const output = serializeIni(doc)
    expect(output).toContain("ctx-size = 65536")
    // new key lands inside the section, before the blank separator
    expect(output).toContain("jinja = true\ntensor-split = 0.6,0.4\n\n[ggml-org/gemma-3-4b-it-GGUF]")
    // unmanaged content untouched
    expect(output).toContain("model = /home/user/.lmstudio/models/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/model.gguf")
    expect(output).toContain("mmproj = /home/user/.lmstudio/models/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/mmproj.gguf")
    expect(output).toContain("# hand-written comment the tooling must not eat")
    expect(output).toContain("temp = 0.7")
    expect(output.startsWith("version = 1\n")).toBe(true)
  })

  test("removes a key when value is undefined", () => {
    const doc = parseIni(FIXTURE)
    setValue(doc.sections[1], "temp", undefined)
    const output = serializeIni(doc)
    expect(output).not.toContain("temp = 0.7")
    expect(output).toContain("# hand-written comment the tooling must not eat")
  })

  test("does not confuse similarly prefixed keys", () => {
    const doc = parseIni(FIXTURE)
    const section = doc.sections[0]
    setValue(section, "cache-type-k", "f16")
    expect(getValue(section, "cache-type-k")).toBe("f16")
    expect(getValue(section, "cache-type-v")).toBe("q8_0")
  })
})
