import os from "os"
import path from "path"
import fs from "fs"
import { spawn } from "child_process"
import * as prompts from "@clack/prompts"
import { InstallationVersion } from "@opencode-ai/core/installation/version"

/**
 * Stack upgrader for llamastack fork builds: `opencode upgrade` checks both
 * components (the opencode fork and the llama.cpp submodule) against their
 * upstreams, tests that syncing is conflict-free, and then pulls + compiles +
 * installs — so upgrading never means hand-running build scripts.
 *
 * The manual equivalent of everything here is documented in the project
 * repo's docs/upgrading.md.
 */

const UPSTREAM_OPENCODE = "https://github.com/sst/opencode.git"
const UPSTREAM_OPENCODE_BRANCH = "dev"
const UPSTREAM_LLAMA = "https://github.com/ggml-org/llama.cpp"
const UPSTREAM_LLAMA_BRANCH = "master"

const LOG_PATH = path.join(os.homedir(), ".local", "state", "llamastack", "upgrade.log")

type Component = "opencode" | "llama.cpp"

type Status = {
  component: Component
  dir: string
  behind: number
  ahead: number
  clean: boolean
  conflicts?: string
  dirty?: boolean
}

/** The opencode-llama.cpp project repo: $LLAMASTACK_REPO, else the conventional location. */
export function stackRepoRoot(): string | undefined {
  const override = process.env["LLAMASTACK_REPO"]
  const candidates = [override, path.join(os.homedir(), "Projects", "opencode-llama.cpp")].filter(
    (dir): dir is string => !!dir,
  )
  for (const dir of candidates) {
    if (fs.existsSync(path.join(dir, ".gitmodules")) && fs.existsSync(path.join(dir, "scripts", "build-llama.sh")))
      return dir
  }
  return undefined
}

function sh(cmd: string, args: string[], cwd: string, logFd?: number): Promise<{ code: number; out: string }> {
  return new Promise((resolve) => {
    const child = spawn(cmd, args, {
      cwd,
      stdio: ["ignore", logFd ?? "pipe", logFd ?? "pipe"],
    })
    let out = ""
    child.stdout?.on("data", (d) => (out += d))
    child.stderr?.on("data", (d) => (out += d))
    child.on("close", (code) => resolve({ code: code ?? 1, out: out.trim() }))
    child.on("error", (err) => resolve({ code: 127, out: String(err) }))
  })
}

async function git(dir: string, ...args: string[]) {
  return sh("git", args, dir)
}

async function componentStatus(component: Component, dir: string): Promise<Status> {
  const upstream = component === "opencode" ? UPSTREAM_OPENCODE : UPSTREAM_LLAMA
  const branch = component === "opencode" ? UPSTREAM_OPENCODE_BRANCH : UPSTREAM_LLAMA_BRANCH
  const dirty = (await git(dir, "status", "--porcelain")).out.length > 0
  const fetch = await git(dir, "fetch", "--quiet", upstream, branch)
  if (fetch.code !== 0) return { component, dir, behind: -1, ahead: -1, clean: false, conflicts: fetch.out, dirty }
  const behind = parseInt((await git(dir, "rev-list", "--count", "HEAD..FETCH_HEAD")).out) || 0
  const ahead = parseInt((await git(dir, "rev-list", "--count", "FETCH_HEAD..HEAD")).out) || 0
  // ahead == 0 means fast-forward: conflict-free by construction. Otherwise a
  // 3-way merge-tree dry run approximates the rebase outcome without touching
  // the working tree (git >= 2.38; on older git we optimistically proceed and
  // rely on rebase --abort).
  let clean = true
  let conflicts: string | undefined
  if (behind > 0 && ahead > 0) {
    const probe = await git(dir, "merge-tree", "--write-tree", "--name-only", "FETCH_HEAD", "HEAD")
    if (probe.code === 1) {
      clean = false
      conflicts = probe.out.split("\n").slice(1).join(", ")
    }
  }
  return { component, dir, behind, ahead, clean, conflicts, dirty }
}

async function upgradeLlama(status: Status, root: string, logFd: number): Promise<string | undefined> {
  const ff = await git(status.dir, "merge", "--ff-only", "FETCH_HEAD")
  if (ff.code !== 0) return `llama.cpp: not fast-forwardable (local patches?) — follow docs/upgrading.md (${ff.out})`
  const build = await sh("bash", [path.join(root, "scripts", "build-llama.sh"), status.dir], root, logFd)
  if (build.code !== 0) return `llama.cpp build failed — see ${LOG_PATH}`
  const server = path.join(status.dir, "build", "bin", "llama-server")
  const version = await sh(server, ["--version"], root)
  if (version.code !== 0) return `built llama-server does not run — see ${LOG_PATH}`
  return undefined
}

async function upgradeOpencode(status: Status, logFd: number): Promise<string | undefined> {
  const rebase = await git(status.dir, "rebase", "FETCH_HEAD")
  if (rebase.code !== 0) {
    await git(status.dir, "rebase", "--abort")
    return "opencode: rebase hit conflicts after all — aborted cleanly; resolve manually per docs/upgrading.md"
  }
  const install = await sh("bun", ["install", "--ignore-scripts"], status.dir, logFd)
  if (install.code !== 0) return `bun install failed — see ${LOG_PATH}`
  await sh("bun", ["run", "--cwd", "packages/core", "fix-node-pty"], status.dir, logFd)
  const pkg = path.join(status.dir, "packages", "opencode")
  const build = await sh("bun", ["run", "script/build.ts", "--single", "--skip-embed-web-ui"], pkg, logFd)
  if (build.code !== 0) return `opencode build failed — see ${LOG_PATH}`
  const built = path.join(pkg, "dist", "opencode-linux-x64", "bin", "opencode")
  if (!fs.existsSync(built)) return `build produced no binary at ${built}`
  // Replace the running binary: write beside it, then rename over (avoids ETXTBSY).
  const target = process.execPath
  fs.copyFileSync(target, target + ".backup")
  fs.copyFileSync(built, target + ".new")
  fs.chmodSync(target + ".new", 0o755)
  fs.renameSync(target + ".new", target)
  return undefined
}

export async function runStackUpgrade(args: { target?: string; check?: boolean; yes?: boolean }) {
  const want = (args.target ?? "all").toLowerCase()
  if (!["all", "opencode", "llama", "llama.cpp"].includes(want)) {
    prompts.log.error(`unknown component "${args.target}" — use: all | opencode | llama`)
    return
  }
  const root = stackRepoRoot()
  if (!root) {
    prompts.log.error(
      "cannot find the opencode-llama.cpp repo (looked at $LLAMASTACK_REPO and ~/Projects/opencode-llama.cpp)",
    )
    return
  }
  prompts.log.info(`stack repo: ${root}`)
  prompts.log.info(`installed:  ${InstallationVersion}`)

  const wanted: Component[] = []
  if (want === "all" || want === "opencode") wanted.push("opencode")
  if (want === "all" || want.startsWith("llama")) wanted.push("llama.cpp")

  const spinner = prompts.spinner()
  spinner.start("Checking upstreams...")
  const statuses: Status[] = []
  for (const component of wanted) {
    statuses.push(await componentStatus(component, path.join(root, component === "opencode" ? "opencode" : "llama.cpp")))
  }
  spinner.stop("Checked upstreams")

  for (const s of statuses) {
    if (s.behind < 0) prompts.log.error(`${s.component}: upstream fetch failed (${s.conflicts})`)
    else {
      const state = s.behind === 0 ? "up to date" : `${s.behind} commits behind upstream`
      const sync = s.behind === 0 ? "" : s.clean ? " — sync is conflict-free" : ` — CONFLICTS in: ${s.conflicts}`
      const dirty = s.dirty ? " (working tree DIRTY — commit or stash first)" : ""
      prompts.log.info(`${s.component}: ${state} (${s.ahead} local commits on top)${sync}${dirty}`)
    }
  }

  const upgradable = statuses.filter((s) => s.behind > 0 && s.clean && !s.dirty)
  if (args.check) return
  if (upgradable.length === 0) {
    prompts.log.info("nothing to upgrade")
    return
  }

  if (!args.yes) {
    const go = await prompts.select({
      message: `Upgrade ${upgradable.map((s) => s.component).join(" + ")}? (compiles from source — llama.cpp takes a while)`,
      options: [
        { label: "Yes", value: true },
        { label: "No", value: false },
      ],
      initialValue: true,
    })
    if (go !== true) return
  }

  fs.mkdirSync(path.dirname(LOG_PATH), { recursive: true })
  const logFd = fs.openSync(LOG_PATH, "w")
  try {
    for (const s of upgradable) {
      const work = prompts.spinner()
      work.start(`Upgrading ${s.component} (+${s.behind} commits, log: ${LOG_PATH})...`)
      const err = s.component === "llama.cpp" ? await upgradeLlama(s, root, logFd) : await upgradeOpencode(s, logFd)
      if (err) {
        work.stop(`${s.component}: FAILED`, 1)
        prompts.log.error(err)
        return
      }
      work.stop(`${s.component}: upgraded and rebuilt`)
    }
  } finally {
    fs.closeSync(logFd)
  }

  // Record the new validated combination as submodule pins (committed, not pushed).
  await git(root, "add", "opencode", "llama.cpp")
  const pin = await git(root, "commit", "-m", "chore: bump pins after `opencode upgrade` stack upgrade")
  if (pin.code === 0) prompts.log.info("pin bump committed in the stack repo (push when happy)")
  prompts.log.info("remember: git push --force <fork remote> in opencode/ after validating the rebase")
}
