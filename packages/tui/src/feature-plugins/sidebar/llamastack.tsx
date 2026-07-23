import type { TuiPlugin, TuiPluginApi } from "@opencode-ai/plugin/tui"
import type { BuiltinTuiPlugin } from "../builtins"
import { createEffect, createMemo, createResource, createSignal, For, onCleanup, Show } from "solid-js"
import { useLocal } from "../../context/local"
import * as LlamaStack from "../../util/llamastack"

const id = "internal:sidebar-llamastack"

/**
 * "Local server" sidebar panel for the llamastack provider: the router endpoint
 * (LAN IP when exposed) and the loaded model's key load params, sourced from the
 * router's GET /models listing with models.ini as fallback. Renders nothing for
 * other providers.
 */
function View(props: { api: TuiPluginApi }) {
  const theme = () => props.api.theme.current
  const local = useLocal()
  const active = createMemo(() => local.model.current()?.providerID === "llamastack")

  const [info, { refetch }] = createResource(
    () => (active() ? (local.model.current()?.modelID ?? true) : undefined),
    async () => {
      const loaded = await LlamaStack.fetchLoadedModel()
      if (loaded) {
        return {
          label: `${loaded.name}${loaded.status === "loading" ? " (loading)" : ""}`,
          params: LlamaStack.formatLoadParams((key) => loaded.args[key]),
        }
      }
      const current = local.model.current()
      if (current?.providerID !== "llamastack") return undefined
      const doc = await LlamaStack.loadPresets()
      const section = doc.sections.find((item) => item.name === current.modelID)
      return {
        label: `${current.modelID} (not loaded)`,
        params: LlamaStack.formatLoadParams((key) => {
          const value = section ? LlamaStack.getValue(section, key) : LlamaStack.DEFAULT_MODEL_SETTINGS[key]
          return value || undefined
        }),
      }
    },
  )

  const [gpus, { refetch: refetchGpus }] = createResource(() => (active() ? true : undefined), LlamaStack.fetchGpuStats)

  createEffect(() => {
    if (!active()) return
    const interval = setInterval(() => {
      void refetch()
      void refetchGpus()
    }, 10_000)
    onCleanup(() => clearInterval(interval))
  })

  const gpuLines = createMemo(() =>
    (gpus() ?? []).map(
      (gpu) =>
        `GPU${gpu.index} ${(gpu.usedMiB / 1024).toFixed(1)}/${(gpu.totalMiB / 1024).toFixed(0)}G · ${gpu.utilization}%`,
    ),
  )

  const endpoint = createMemo(() => LlamaStack.endpointURL(LlamaStack.serverSettings()?.expose ?? true))

  // Web access: the sidebar line doubles as the toggle. `busy` covers the ~10s
  // cloudflared takes to be assigned a hostname, so the row isn't just blank.
  const webOn = createMemo(() => LlamaStack.serverSettings()?.web ?? false)
  const webUrl = createMemo(() => LlamaStack.tunnelURL())
  const [busy, setBusy] = createSignal(false)

  // reconcile on mount: settings say web is on but no live tunnel (e.g. after a
  // reboot killed cloudflared) -> bring it back up.
  createEffect(() => {
    if (!active() || !webOn() || busy() || webUrl()) return
    void toggleWeb(true)
  })

  async function toggleWeb(on: boolean) {
    if (busy()) return
    setBusy(true)
    try {
      await LlamaStack.setWebAccess(on)
    } finally {
      setBusy(false)
    }
  }

  const webLine = createMemo(() => {
    if (busy()) return webOn() ? "web: starting…" : "web: stopping…"
    if (!webOn()) return "web: off (click to enable)"
    return webUrl() ?? "web: unavailable (cloudflared missing?)"
  })

  // Shown so the endpoint is usable from ANY OpenAI-compatible client — Claude
  // Code, Cursor, curl — not just this one. Bearer auth, so: base URL + key.
  const apiKey = createMemo(() => LlamaStack.serverSettings()?.apiKey)

  return (
    <Show when={active()}>
      <box>
        <text fg={theme().text}>
          <b>Local server</b>
        </text>
        <text fg={theme().textMuted} wrapMode="none">
          {endpoint()}
        </text>
        <text
          fg={webOn() && webUrl() ? theme().text : theme().textMuted}
          wrapMode="none"
          onMouseUp={() => void toggleWeb(!webOn())}
        >
          {webLine()}
        </text>
        <Show when={apiKey()}>
          {(key) => (
            <text fg={theme().textMuted} wrapMode="none">
              {`key: ${key()}`}
            </text>
          )}
        </Show>
        <Show when={info()}>
          {(item) => (
            <>
              <text fg={theme().textMuted} wrapMode="none">
                {item().label}
              </text>
              <Show when={item().params}>
                <text fg={theme().textMuted} wrapMode="none">
                  {item().params}
                </text>
              </Show>
            </>
          )}
        </Show>
        <For each={gpuLines()}>
          {(line) => (
            <text fg={theme().textMuted} wrapMode="none">
              {line}
            </text>
          )}
        </For>
      </box>
    </Show>
  )
}

const tui: TuiPlugin = async (api) => {
  api.slots.register({
    order: 350,
    slots: {
      sidebar_content() {
        return <View api={api} />
      },
    },
  })
}

const plugin: BuiltinTuiPlugin = {
  id,
  tui,
}

export default plugin
