import { createEffect, createMemo, createSignal, onCleanup, Show } from "solid-js"
import { useLocal } from "../context/local"
import { useTheme } from "../context/theme"
import * as LlamaStack from "../util/llamastack"

/**
 * Live "loading model" indicator for the llamastack provider. Local models are
 * loaded into VRAM on demand by the llama-server router, which can take a
 * while; this streams the router's load progress next to the busy spinner.
 * Mount it only while a request is in flight — it subscribes to the router's
 * SSE endpoint on mount (llamastack models only) and closes the connection on
 * unmount. Renders nothing for other providers or when nothing is loading.
 */
export function LlamaStackLoadStatus() {
  const local = useLocal()
  const { theme } = useTheme()
  const [loading, setLoading] = createSignal<LlamaStack.ModelStatusEvent>()

  const enabled = createMemo(() => local.model.current()?.providerID === "llamastack")

  createEffect(() => {
    if (!enabled()) return
    const unsubscribe = LlamaStack.subscribeModelStatus((event) => {
      if (event.status === "loading") setLoading(event)
      else setLoading((prev) => (prev?.model === event.model ? undefined : prev))
    })
    onCleanup(() => {
      unsubscribe()
      setLoading(undefined)
    })
  })

  const label = createMemo(() => {
    const event = loading()
    if (!event) return ""
    const name = event.model.split("/").filter(Boolean).at(-1) ?? event.model
    const stage = event.stage ? ` ${event.stage}` : ""
    const percent = event.percent !== undefined ? ` ${event.percent}%` : "…"
    return `loading ${name}${stage}${percent}`
  })

  return (
    <Show when={loading()}>
      <text fg={theme.accent} wrapMode="none">
        {label()}
      </text>
    </Show>
  )
}
