import { createMemo, createResource, createSignal, Match, Switch } from "solid-js"
import { createStore } from "solid-js/store"
import { DialogSelect, type DialogSelectOption } from "../ui/dialog-select"
import { DialogPrompt } from "../ui/dialog-prompt"
import { useDialog } from "../ui/dialog"
import { useToast } from "../ui/toast"
import { useTheme } from "../context/theme"
import { useBindings } from "../keymap"
import * as LlamaStack from "../util/llamastack"

type Field = {
  key: string
  title: string
  kind: "integer" | "number" | "enum" | "toggle" | "split"
  values?: string[]
  /** Value assumed when the key is absent from the section ("" = llama-server default). */
  fallback: string
  hint: string
}

const FIELDS: Field[] = [
  { key: "ctx-size", title: "Context window", kind: "integer", fallback: "32768", hint: "tokens, e.g. 32768" },
  { key: "gpu-layers", title: "GPU layers", kind: "integer", fallback: "99", hint: "layers on GPU, 99 = all" },
  {
    key: "tensor-split",
    title: "Tensor split",
    kind: "split",
    fallback: "",
    hint: "VRAM ratio GPU 0 / GPU 1, e.g. 0.5,0.5 — empty = auto",
  },
  { key: "cache-type-k", title: "KV cache K", kind: "enum", values: ["f16", "q8_0", "q4_0"], fallback: "q8_0", hint: "" },
  { key: "cache-type-v", title: "KV cache V", kind: "enum", values: ["f16", "q8_0", "q4_0"], fallback: "q8_0", hint: "" },
  { key: "flash-attn", title: "Flash attention", kind: "toggle", values: ["on", "off"], fallback: "on", hint: "" },
  { key: "temp", title: "Temperature", kind: "number", fallback: "", hint: "e.g. 0.7 — empty = server default" },
]

function validate(field: Field, value: string): string | undefined {
  if (value === "") return undefined
  switch (field.kind) {
    case "integer":
      return /^\d+$/.test(value) ? undefined : "must be a whole number"
    case "number":
      return /^\d+(\.\d+)?$/.test(value) ? undefined : "must be a number"
    case "split":
      return /^\d*\.?\d+(\s*,\s*\d*\.?\d+)+$/.test(value) ? undefined : 'must be comma-separated ratios like "0.5,0.5"'
    default:
      return undefined
  }
}

type Step = { type: "models" } | { type: "settings" } | { type: "edit"; field: Field }

export function DialogLlamaStackConfig() {
  const dialog = useDialog()
  const toast = useToast()
  const { theme } = useTheme()

  const [data] = createResource(async () => {
    const [doc, discovered] = await Promise.all([LlamaStack.loadPresets(), LlamaStack.discoverLocalModels()])
    return { doc, discovered }
  })

  const [step, setStep] = createSignal<Step>({ type: "models" })
  const [model, setModel] = createSignal<string>()
  const [values, setValues] = createStore<Record<string, string>>({})

  const models = createMemo(() => {
    const loaded = data()
    if (!loaded) return []
    const names = loaded.discovered.map((entry) => entry.name)
    for (const section of loaded.doc.sections) {
      if (section.name === "*") continue // llama.cpp global preset section
      if (!names.includes(section.name)) names.push(section.name)
    }
    return names
  })

  function openModel(name: string) {
    const section = data()?.doc.sections.find((item) => item.name === name)
    for (const field of FIELDS) {
      setValues(field.key, (section ? LlamaStack.getValue(section, field.key) : undefined) ?? field.fallback)
    }
    setModel(name)
    setStep({ type: "settings" })
  }

  async function save() {
    const loaded = data()
    const name = model()
    if (!loaded || !name) return
    const doc = loaded.doc
    let section = doc.sections.find((item) => item.name === name)
    if (!section) {
      const discovered = loaded.discovered.find((entry) => entry.name === name)
      section = discovered ? LlamaStack.defaultSection(discovered) : { name, lines: [] }
      // blank line between the previous content and the new section
      const previous = doc.sections.at(-1)?.lines ?? doc.prelude
      if (previous.at(-1)?.trim() !== "") previous.push("")
      section.lines.push("")
      doc.sections.push(section)
    }
    for (const field of FIELDS) {
      const value = values[field.key]?.trim() ?? ""
      LlamaStack.setValue(section, field.key, value === "" ? undefined : value)
    }
    try {
      await LlamaStack.savePresets(doc)
    } catch (error) {
      toast.error(error as Error)
      return
    }
    const reloaded = await LlamaStack.reloadRouter()
    toast.show({
      message: reloaded ? `Saved settings for ${name}` : `Saved settings for ${name} — applies on next model load`,
      variant: "success",
      duration: 5000,
    })
    dialog.clear()
  }

  // esc steps back through the flow instead of closing the dialog outright
  useBindings(() => ({
    enabled: step().type !== "models",
    // must win over the dialog layer's own escape (priority 0)
    priority: 1,
    bindings: [
      {
        key: "escape",
        desc: "Back",
        group: "Dialog",
        cmd: () => setStep((prev) => (prev.type === "edit" ? { type: "settings" } : { type: "models" })),
      },
    ],
  }))

  const modelOptions = createMemo<DialogSelectOption<string>[]>(() =>
    models().map((name) => ({
      title: name,
      value: name,
      description: data()?.doc.sections.some((section) => section.name === name) ? undefined : "(defaults)",
      onSelect: () => openModel(name),
    })),
  )

  const settingOptions = createMemo<DialogSelectOption<string>[]>(() => [
    ...FIELDS.map((field) => {
      const value = values[field.key] ?? ""
      const display = value === "" ? "auto" : value
      return {
        title: field.title,
        value: field.key,
        description: value === field.fallback ? `${display} (default)` : display,
        onSelect: () => {
          if (field.kind === "toggle") {
            setValues(field.key, value === "on" ? "off" : "on")
            return
          }
          setStep({ type: "edit", field })
        },
      }
    }),
    {
      title: "Save",
      value: "save",
      description: "write models.ini and reload the router",
      onSelect: () => void save(),
    },
  ])

  const editing = createMemo(() => {
    const current = step()
    return current.type === "edit" ? current.field : undefined
  })

  function confirmEdit(field: Field, raw: string) {
    const value = raw.replace(/\s+/g, " ").trim()
    const error = validate(field, value)
    if (error) {
      toast.show({ message: `${field.title} ${error}`, variant: "error", duration: 3000 })
      return
    }
    setValues(field.key, value)
    setStep({ type: "settings" })
  }

  return (
    <Switch>
      <Match when={step().type === "models"}>
        <DialogSelect
          title="Configure local model"
          placeholder="Search models"
          options={modelOptions()}
          emptyView={
            <box paddingLeft={4} paddingRight={4} paddingTop={1}>
              <text fg={theme.textMuted}>
                {data.loading ? "Scanning local models..." : `No local models found in ${LlamaStack.MODELS_DIR}`}
              </text>
            </box>
          }
        />
      </Match>
      <Match when={step().type === "settings"}>
        <DialogSelect title={model() ?? ""} options={settingOptions()} renderFilter={false} />
      </Match>
      <Match when={editing()?.kind === "enum"}>
        <DialogSelect
          title={editing()!.title}
          options={(editing()!.values ?? []).map((value) => ({
            title: value,
            value,
            onSelect: () => {
              setValues(editing()!.key, value)
              setStep({ type: "settings" })
            },
          }))}
          renderFilter={false}
          current={values[editing()!.key]}
        />
      </Match>
      <Match when={editing()}>
        <DialogPrompt
          title={editing()!.title}
          description={() => <text fg={theme.textMuted}>{editing()!.hint}</text>}
          placeholder={editing()!.fallback === "" ? "auto" : editing()!.fallback}
          value={values[editing()!.key]}
          onConfirm={(value) => confirmEdit(editing()!, value)}
        />
      </Match>
    </Switch>
  )
}
