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
  const [failed, setFailed] = createSignal<LlamaStack.ModelStatusEvent>()

  const enabled = createMemo(() => local.model.current()?.providerID === "llamastack")

  createEffect(() => {
    if (!enabled()) return
    const unsubscribe = LlamaStack.subscribeModelStatus((event) => {
      if (event.status === "loading") {
        setLoading(event)
        setFailed(undefined)
        return
      }
      // a watched load that ends in error/nonzero exit died mid-load — say so
      // in place instead of silently vanishing
      setLoading((prev) => {
        if (prev?.model !== event.model) return prev
        if (event.status === "error" || (event.exitCode !== undefined && event.exitCode !== 0)) {
          setFailed(event)
          setTimeout(() => setFailed((current) => (current === event ? undefined : current)), 30_000)
        }
        return undefined
      })
    })
    onCleanup(() => {
      unsubscribe()
      setLoading(undefined)
      setFailed(undefined)
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

  const failedLabel = createMemo(() => {
    const event = failed()
    if (!event) return ""
    const name = event.model.split("/").filter(Boolean).at(-1) ?? event.model
    const code = event.exitCode !== undefined ? ` (exit ${event.exitCode})` : ""
    return `${name} failed to load${code} — likely out of VRAM; lower ctx-size in /config`
  })

  return (
    <>
      <Show when={loading()}>
        <text fg={theme.accent} wrapMode="none">
          {label()}
        </text>
      </Show>
      <Show when={failed()}>
        <text fg={theme.error} wrapMode="none">
          {failedLabel()}
        </text>
      </Show>
    </>
  )
}
