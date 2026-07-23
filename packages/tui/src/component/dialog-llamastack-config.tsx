import { createMemo, createResource } from "solid-js"
import { createStore } from "solid-js/store"
import { DialogSelect, type DialogSelectOption } from "../ui/dialog-select"
import { DialogPrompt } from "../ui/dialog-prompt"
import { useDialog } from "../ui/dialog"
import { useToast } from "../ui/toast"
import { useTheme } from "../context/theme"
import { useLocal } from "../context/local"
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

// The dialog layer re-invokes the top-of-stack element factory when its store
// changes, remounting the dialog component and resetting any local step state
// (that was the "/config blinks back to the model list" bug). So this flow is
// structured the way the rest of the codebase does multi-step dialogs: one
// component per screen, chained with dialog.replace(), with the edit draft in
// a module-level map so every screen re-seeds identically after any remount.
const drafts = new Map<string, Record<string, string>>()

async function seedDraft(model: string): Promise<Record<string, string>> {
  const existing = drafts.get(model)
  if (existing) return existing
  const doc = await LlamaStack.loadPresets()
  const section = doc.sections.find((item) => item.name === model)
  const values: Record<string, string> = {}
  for (const field of FIELDS) {
    values[field.key] = (section ? LlamaStack.getValue(section, field.key) : undefined) ?? field.fallback
  }
  drafts.set(model, values)
  return values
}

/** Entry point: model list + Server entry. Registered as /config. */
export function DialogLlamaStackConfig() {
  const dialog = useDialog()
  const local = useLocal()
  const { theme } = useTheme()

  const [data] = createResource(async () => {
    const [doc, discovered] = await Promise.all([LlamaStack.loadPresets(), LlamaStack.discoverLocalModels()])
    return { doc, discovered }
  })
  const [loaded] = createResource(LlamaStack.fetchLoadedModel)

  const models = createMemo(() => {
    const value = data()
    if (!value) return []
    // files already claimed by a (possibly renamed/tuned) section belong to that
    // section — don't list them again under their generated quant name
    const claimed = new Set(
      value.doc.sections.map((section) => LlamaStack.getValue(section, "model")).filter(Boolean),
    )
    const names = value.discovered.filter((entry) => !claimed.has(entry.model)).map((entry) => entry.name)
    for (const section of value.doc.sections) {
      if (section.name === "*") continue // llama.cpp global preset section
      if (!names.includes(section.name)) names.push(section.name)
    }
    return names
  })

  // endpoint + loaded-model summary shown under the Server entry
  const serverDetails = createMemo(() => {
    const settings = LlamaStack.serverSettings()
    const details = [LlamaStack.endpointURL(settings?.expose ?? true)]
    const entry = loaded()
    if (entry) {
      const params = LlamaStack.formatLoadParams((key) => entry.args[key])
      details.push(`${entry.status}: ${entry.name}${params ? " · " + params : ""}`)
      return details
    }
    const current = local.model.current()
    if (current?.providerID !== "llamastack") {
      details.push("no model loaded")
      return details
    }
    const section = data()?.doc.sections.find((item) => item.name === current.modelID)
    const params = LlamaStack.formatLoadParams((key) => {
      const value = section ? LlamaStack.getValue(section, key) : FIELDS.find((field) => field.key === key)?.fallback
      return value || undefined
    })
    details.push(`not loaded: ${current.modelID}${params ? " · " + params : ""}`)
    return details
  })

  const options = createMemo<DialogSelectOption<string>[]>(() => [
    {
      title: "Server",
      value: "__server__",
      category: "Server",
      details: serverDetails(),
      onSelect: () => dialog.replace(() => <DialogLlamaStackServer />),
    },
    ...models().map((name) => {
      const active = loaded()?.name === name ? loaded()!.status : undefined
      const custom = data()?.doc.sections.some((section) => section.name === name)
      return {
        title: name,
        value: name,
        category: "Models",
        description: [active ? `● ${active}` : undefined, custom ? undefined : "(defaults)"]
          .filter(Boolean)
          .join(" "),
        onSelect: () => {
          void seedDraft(name).then(() => dialog.replace(() => <DialogLlamaStackSettings model={name} />))
        },
      }
    }),
  ])

  return (
    <DialogSelect
      title="Configure local model"
      placeholder="Search models"
      options={options()}
      emptyView={
        <box paddingLeft={4} paddingRight={4} paddingTop={1}>
          <text fg={theme.textMuted}>
            {data.loading ? "Scanning local models..." : `No local models found in ${LlamaStack.MODELS_DIR}`}
          </text>
        </box>
      }
    />
  )
}

function DialogLlamaStackSettings(props: { model: string }) {
  const dialog = useDialog()
  const toast = useToast()
  const local = useLocal()
  const [loaded, { refetch: refetchLoaded }] = createResource(LlamaStack.fetchLoadedModel)
  // seedDraft ran before this dialog opened; a remount re-reads the same draft.
  // CLONE the draft: wrapping the shared object directly would cache a store
  // proxy on it, and the edit dialog's raw writes would then read back stale.
  const [values, setValues] = createStore<Record<string, string>>({ ...(drafts.get(props.model) ?? {}) })

  function set(key: string, value: string) {
    setValues(key, value)
    const draft = drafts.get(props.model)
    if (draft) draft[key] = value
  }

  // esc returns to the model list instead of closing the dialog outright
  useBindings(() => ({
    enabled: true,
    // must win over the dialog layer's own escape (priority 0)
    priority: 1,
    bindings: [
      {
        key: "escape",
        desc: "Back",
        group: "Dialog",
        cmd: () => {
          drafts.delete(props.model)
          dialog.replace(() => <DialogLlamaStackConfig />)
        },
      },
    ],
  }))

  async function save() {
    const doc = await LlamaStack.loadPresets()
    let section = doc.sections.find((item) => item.name === props.model)
    if (!section) {
      const discovered = (await LlamaStack.discoverLocalModels()).find((entry) => entry.name === props.model)
      section = discovered ? LlamaStack.defaultSection(discovered) : { name: props.model, lines: [] }
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
    drafts.delete(props.model)
    const reloaded = await LlamaStack.reloadRouter()
    toast.show({
      message: reloaded
        ? `Saved settings for ${props.model}`
        : `Saved settings for ${props.model} — applies on next model load`,
      variant: "success",
      duration: 5000,
    })
    dialog.clear()
  }

  const options = createMemo<DialogSelectOption<string>[]>(() => [
    ...FIELDS.map((field) => {
      const value = values[field.key] ?? ""
      const display = value === "" ? "auto" : value
      return {
        title: field.title,
        value: field.key,
        description: value === field.fallback ? `${display} (default)` : display,
        onSelect: () => {
          if (field.kind === "toggle") {
            set(field.key, value === "on" ? "off" : "on")
            return
          }
          dialog.replace(() => <DialogLlamaStackEdit model={props.model} fieldKey={field.key} />)
        },
      }
    }),
    ...(loaded()?.name === props.model
      ? [
          {
            title: "Unload model",
            value: "__unload__",
            description: `${loaded()!.status} — unload to free VRAM`,
            onSelect: () =>
              void LlamaStack.unloadModel(props.model).then((ok) => {
                toast.show({
                  message: ok ? `Unloaded ${props.model} — VRAM freed` : `Could not unload ${props.model}`,
                  variant: ok ? "success" : "warning",
                  duration: 5000,
                })
                void refetchLoaded()
              }),
          },
        ]
      : []),
    // /config's model list looks like a picker, so make switching possible here —
    // selecting a model in this dialog otherwise only configures it (real switch
    // lives in "Switch model", ctrl+x m), which has confused actual users
    ...(local.model.current()?.providerID === "llamastack" && local.model.current()?.modelID === props.model
      ? []
      : [
          {
            title: "Use this model",
            value: "__use__",
            description: "switch the session to this model",
            onSelect: () => {
              local.model.set({ providerID: "llamastack", modelID: props.model })
              toast.show({ message: `Switched to ${props.model}`, variant: "success", duration: 4000 })
              dialog.clear()
            },
          },
        ]),
    {
      title: "Reset to recommended",
      value: "__reset__",
      description: "max-context values tuned for this machine",
      onSelect: () => {
        const recommended = LlamaStack.recommendedFor(props.model)
        for (const field of FIELDS) set(field.key, recommended[field.key] ?? field.fallback)
      },
    },
    {
      title: "Save",
      value: "__save__",
      description: "write models.ini and reload the router",
      onSelect: () => void save(),
    },
  ])

  return <DialogSelect title={props.model} options={options()} renderFilter={false} />
}

function DialogLlamaStackEdit(props: { model: string; fieldKey: string }) {
  const dialog = useDialog()
  const toast = useToast()
  const { theme } = useTheme()
  const field = FIELDS.find((item) => item.key === props.fieldKey)!
  const draft = drafts.get(props.model)

  function back() {
    dialog.replace(() => <DialogLlamaStackSettings model={props.model} />)
  }

  function confirm(raw: string) {
    const value = raw.replace(/\s+/g, " ").trim()
    const error = validate(field, value)
    if (error) {
      toast.show({ message: `${field.title} ${error}`, variant: "error", duration: 3000 })
      return
    }
    if (draft) draft[field.key] = value
    back()
  }

  useBindings(() => ({
    enabled: true,
    priority: 1,
    bindings: [{ key: "escape", desc: "Back", group: "Dialog", cmd: back }],
  }))

  if (field.kind === "enum") {
    return (
      <DialogSelect
        title={field.title}
        options={(field.values ?? []).map((value) => ({
          title: value,
          value,
          onSelect: () => {
            if (draft) draft[field.key] = value
            back()
          },
        }))}
        renderFilter={false}
        current={draft?.[field.key]}
      />
    )
  }
  return (
    <DialogPrompt
      title={field.title}
      description={() => <text fg={theme.textMuted}>{field.hint}</text>}
      placeholder={field.fallback === "" ? "auto" : field.fallback}
      value={draft?.[field.key]}
      onConfirm={confirm}
    />
  )
}

function DialogLlamaStackServer() {
  const dialog = useDialog()
  const toast = useToast()

  const [settings] = createResource(LlamaStack.loadServerSettings)
  const [loaded, { refetch: refetchLoaded }] = createResource(LlamaStack.fetchLoadedModel)
  const [values, setValues] = createStore<{ expose: boolean | undefined }>({ expose: undefined })
  const expose = createMemo(() => values.expose ?? settings()?.expose ?? true)

  async function unload() {
    const entry = loaded()
    if (!entry) return
    const ok = await LlamaStack.unloadModel(entry.name)
    toast.show({
      message: ok ? `Unloaded ${entry.name} — VRAM freed` : `Could not unload ${entry.name}`,
      variant: ok ? "success" : "warning",
      duration: 5000,
    })
    void refetchLoaded()
  }

  useBindings(() => ({
    enabled: true,
    priority: 1,
    bindings: [
      {
        key: "escape",
        desc: "Back",
        group: "Dialog",
        cmd: () => dialog.replace(() => <DialogLlamaStackConfig />),
      },
    ],
  }))

  async function save() {
    // `web` and `apiKey` are owned elsewhere (sidebar toggle / generated once) —
    // carry them through so saving the LAN setting doesn't tear down a live
    // tunnel or rotate the key out from under connected clients.
    const current = LlamaStack.serverSettings()
    const next = { expose: expose(), web: current?.web ?? false, apiKey: current?.apiKey ?? "" }
    try {
      await LlamaStack.saveServerSettings(next)
    } catch (error) {
      toast.error(error as Error)
      return
    }
    const result = await LlamaStack.restartRouter(next.expose)
    toast.show({
      message:
        result === "restarted"
          ? `Router restarted on ${LlamaStack.endpointURL(next.expose)}`
          : result === "not-running"
            ? "Saved — applies when the router starts"
            : "Saved — restart the router to apply",
      variant: result === "failed" ? "warning" : "success",
      duration: 5000,
    })
    dialog.clear()
  }

  const options = createMemo<DialogSelectOption<string>[]>(() => [
    {
      title: "Expose on network",
      value: "expose",
      description: expose()
        ? `on — anyone on your LAN can use ${LlamaStack.endpointURL(true)}`
        : "off — localhost only",
      onSelect: () => setValues("expose", !expose()),
    },
    ...(loaded()
      ? [
          {
            title: "Unload model",
            value: "__unload__",
            description: `${loaded()!.name} is ${loaded()!.status} — unload to free VRAM`,
            onSelect: () => void unload(),
          },
        ]
      : []),
    {
      title: "Save",
      value: "__save__",
      description: "write server.json and restart the router",
      onSelect: () => void save(),
    },
  ])

  return <DialogSelect title="Server" options={options()} renderFilter={false} />
}
