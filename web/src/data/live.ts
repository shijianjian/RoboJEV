/**
 * The live console's half of `EpisodeSource`: one WebSocket, one episode, one reducer.
 *
 * The server is `robojev console` (`robojev/console/`), the messages are `PROTOCOL.md`, and the
 * one sentence the whole design rests on is that **a live `decision` message is byte-for-byte one
 * entry of a bundle's `decisions` array**. So there is no live `Decision` type: the panel, the
 * state text, the details row and the stage timeline are handed the same objects a replay hands
 * them, and {@link liveEpisode} assembles the header and the decisions so far into the same
 * {@link Episode} shape the replay components already take.
 *
 * Everything that decides *what the page shows* is {@link applyMessage}, a pure function from
 * (state, message) to state. The socket is a thin thing around it, which is what lets the
 * protocol be tested without a browser or a server -- and what keeps "the bars lit an override
 * that the server refused" from being possible: the armed set comes from the server's `status`
 * and from nowhere else.
 */
import type { Decision, Episode, EpisodeIndexEntry } from "./types";

/** Where the socket is. */
export type Connection = "connecting" | "open" | "closed" | "refused";

/** Where the run is, as `status.state` reports it. */
export type RunState = "idle" | "starting" | "waiting" | "paused" | "running" | "done" | "error";

/** The id the episode strip and the hash give the console. */
export const LIVE_ID = "live";

/** How the console publishes its camera renders. `poll` is one HTTP request per picture, which
 *  needs nothing in the page but an `<img>` and nothing in the server but the standard library. */
export interface LiveVideo {
  kind: string;
  cameras: Partial<Record<"agentview" | "wrist", string>>;
  rate: number;
}

/** What a `start` asks for. */
export interface StartSpec {
  suite: string;
  task: number;
  init: number;
  policy: string;
  selection: string;
  temperature?: number | null;
  checkpoint?: string | null;
}

/** One set of weights the page offers: choosing it starts with its `policy` and `checkpoint`. */
export interface LiveWeights {
  id: string;
  label: string;
  revision: string | null;
  policy: string;
  checkpoint: string | null;
}

/** What this console can be asked for, sent once on connect. */
export interface LiveConfig {
  /** Absent from a console older than the weights picker: the page then offers its engines. */
  weights?: LiveWeights[];
  protocol: number;
  policies: string[];
  suites: string[];
  default: StartSpec;
  video: LiveVideo | null;
  replays: string | null;
}

/** One task of a suite, once the console has read the simulator's task definitions. */
export interface LiveTask {
  index: number;
  instruction: string;
  init_states: number;
}

/** The episode header: the bundle's top-level fields, minus the ones only an ended episode has. */
export interface LiveHeader {
  schema_version: number;
  id: string;
  suite: string;
  task_index: number;
  init_state_index: number;
  instruction: string;
  title: string;
  note: string;
  policy: string;
  checkpoint_revision: string | null;
  checkpoint_repo: string | null;
  questions_version: string | null;
  selection: string;
  control_rate: number;
  wait_steps: number;
  execute_steps: number;
  max_steps: number;
  max_decisions: number;
}

export interface LiveDone {
  success: boolean;
  terminated_by: string;
  steps: number;
  decisions: number;
  error: string | null;
}

export interface LiveSaved {
  id: string;
  path: string;
  url: string | null;
  decisions: number;
  bytes: number;
  success: boolean;
}

/** Where the live episode's 3D scene loads from (`hello.scene`). */
export interface LiveSceneRef {
  hash: string;
  xml: string;
  assets: string;
  nq: number | null;
}

/** The simulator's joint positions at one control step (`pose`). */
export interface LivePose {
  step: number;
  qpos: number[];
}

export interface LiveState {
  connection: Connection;
  run: RunState;
  /** An episode is open: there is a header, and the controls are the driving ones. */
  episode: boolean;
  header: LiveHeader | null;
  video: LiveVideo | null;
  config: LiveConfig | null;
  decisions: Decision[];
  /** The control step the episode has reached, from `status`. */
  step: number | null;
  /** The armed overrides, **as the server reports them** — never as the page remembers clicking. */
  overrides: Record<string, string>;
  message: string | null;
  error: string | null;
  /** The console will not take this connection back: another window has it. */
  fatal: boolean;
  done: LiveDone | null;
  saved: LiveSaved | null;
  tasks: LiveTask[] | null;
  /** The 3D scene of the episode in hand, or null (none exported, or a console without scenes). */
  scene: LiveSceneRef | null;
  /** The newest pose the console sent, for the 3D scene. */
  pose: LivePose | null;
}

export const EMPTY_LIVE: LiveState = {
  connection: "connecting", run: "idle", episode: false, header: null, video: null, config: null,
  decisions: [], step: null, overrides: {}, message: null, error: null, fatal: false, done: null,
  saved: null, tasks: null, scene: null, pose: null,
};

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

const RUN_STATES: RunState[] = ["idle", "starting", "waiting", "paused", "running", "done", "error"];

function runState(raw: unknown, fallback: RunState): RunState {
  return RUN_STATES.includes(raw as RunState) ? (raw as RunState) : fallback;
}

/**
 * One message, folded into the page's state.
 *
 * A message this page does not know changes nothing rather than throwing: a console newer than
 * the build in front of it should degrade to the parts they share, not to a blank screen.
 */
export function applyMessage(state: LiveState, raw: unknown): LiveState {
  if (!isRecord(raw)) return state;
  switch (raw.type) {
    case "config": {
      const config = raw as unknown as LiveConfig;
      return {
        ...state,
        config,
        video: config.video ?? state.video,
        run: runState(raw.state, state.run),
        connection: "open",
      };
    }
    case "hello": {
      // A new episode replaces everything about the previous one in one go: a decision of the old
      // one under the new one's header would be a paragraph about a scene that no longer exists.
      return {
        ...state,
        header: (raw.episode ?? null) as LiveHeader | null,
        video: (raw.video ?? state.video) as LiveVideo | null,
        episode: true,
        decisions: [],
        step: null,
        done: null,
        saved: null,
        error: null,
        message: null,
        overrides: {},
        scene: sceneRef(raw.scene),
        pose: null,
      };
    }
    case "pose": {
      if (typeof raw.step !== "number" || !Array.isArray(raw.qpos)
          || !raw.qpos.every((v) => typeof v === "number" && Number.isFinite(v))) return state;
      // A pose older than the one on screen is a late message, not a step back in time.
      if (state.pose !== null && raw.step < state.pose.step) return state;
      return { ...state, pose: { step: raw.step, qpos: raw.qpos as number[] } };
    }
    case "decision": {
      const decision = raw.decision as Decision | undefined;
      if (decision === undefined || typeof decision.index !== "number") return state;
      const decisions = state.decisions.slice();
      // Indexed rather than appended: a reconnect that replays a decision must not draw it twice.
      decisions[decision.index] = decision;
      return { ...state, decisions, error: null };
    }
    case "status": {
      return {
        ...state,
        run: runState(raw.state, state.run),
        step: typeof raw.step === "number" ? raw.step : state.step,
        overrides: isRecord(raw.overrides) ? (raw.overrides as Record<string, string>) : {},
        message: typeof raw.message === "string" ? raw.message : null,
        episode: raw.episode === true,
      };
    }
    case "done":
      return { ...state, done: raw as unknown as LiveDone };
    case "error":
      return {
        ...state,
        error: typeof raw.message === "string" ? raw.message : "the console refused that",
        fatal: state.fatal || raw.fatal === true,
      };
    case "saved":
      return { ...state, saved: raw as unknown as LiveSaved, error: null };
    case "tasks":
      return { ...state, tasks: Array.isArray(raw.tasks) ? (raw.tasks as LiveTask[]) : null };
    default:
      return state;
  }
}

function sceneRef(raw: unknown): LiveSceneRef | null {
  if (!isRecord(raw) || typeof raw.hash !== "string" || typeof raw.xml !== "string"
      || typeof raw.assets !== "string") return null;
  return { hash: raw.hash, xml: raw.xml, assets: raw.assets,
           nq: typeof raw.nq === "number" ? raw.nq : null };
}

/**
 * The episode so far, in the shape the replay components already read.
 *
 * Not a conversion: every field below is either the header's own or a count of what has arrived.
 * `media` is empty because a live episode has no mp4 — it is still happening — and the page shows
 * the console's picture stream in its place.
 */
export function liveEpisode(state: LiveState): Episode | null {
  const header = state.header;
  if (header === null) return null;
  const last = state.decisions[state.decisions.length - 1];
  const reached = state.step ?? (last === undefined ? 0 : last.control_step + header.execute_steps);
  const frames = Math.max(reached, 1);
  return {
    schema_version: header.schema_version,
    id: header.id,
    suite: header.suite,
    task_index: header.task_index,
    init_state_index: header.init_state_index,
    instruction: header.instruction,
    title: header.title,
    note: header.note,
    policy: header.policy,
    checkpoint_revision: header.checkpoint_revision,
    checkpoint_repo: header.checkpoint_repo,
    questions_version: header.questions_version,
    success: state.done?.success ?? false,
    terminated_by: state.done?.terminated_by ?? "live",
    error: state.done?.error ?? null,
    steps: state.done?.steps ?? reached,
    control_rate: header.control_rate,
    wait_steps: header.wait_steps,
    execute_steps: header.execute_steps,
    max_steps: header.max_steps,
    max_decisions: header.max_decisions,
    total_frames: frames,
    duration_s: frames / header.control_rate,
    recorded_at: "",
    wall_seconds: 0,
    media: {},
    poster: null,
    decisions: state.decisions,
  };
}

/** The strip item the console occupies: one row, in front of the recorded ones. */
export function liveEntry(state: LiveState): EpisodeIndexEntry {
  const header = state.header;
  return {
    id: LIVE_ID,
    title: header?.title ?? "live console",
    // Short, because the strip cuts a long one to a line: before an episode there is no task
    // sentence to show and "live console" is what the item *is*.
    instruction: header?.instruction ?? "live console",
    note: "",
    success: state.done?.success ?? false,
    decisions: state.decisions.length,
    max_decisions: header?.max_decisions ?? 0,
    task_index: header?.task_index ?? 0,
    init_state_index: header?.init_state_index ?? 0,
    suite: header?.suite ?? "",
    poster: null,
  };
}

/** Which commands the console would accept right now — the same rules the server enforces, so a
 *  control that is offered is a control that works. */
export function commandsEnabled(state: LiveState): {
  start: boolean; step: boolean; run: boolean; pause: boolean; reset: boolean; save: boolean;
  override: boolean;
} {
  const open = state.connection === "open";
  const driving = open && (state.run === "paused" || state.run === "waiting");
  return {
    start: open && state.run === "idle",
    step: driving,
    run: driving,
    pause: open && state.run === "running",
    reset: open && state.episode,
    save: open && state.episode && state.run !== "starting",
    override: driving,
  };
}

/** The line under the controls: where the session is, in one sentence. */
export function sessionLine(state: LiveState): string {
  if (state.connection === "refused") return "another window has this console";
  if (state.connection !== "open") return "connecting to the console…";
  switch (state.run) {
    case "idle":
      return "no episode · pick a task and weights, then Start";
    case "starting":
      return state.message ?? "starting · building the simulator and the policy";
    case "waiting":
      return "settling · the scene is dropping into place";
    case "running":
      return `running · step ${state.step ?? 0} of ${state.header?.max_steps ?? "?"}`;
    case "done":
      return state.done === null
        ? "the episode has ended"
        : `${state.done.terminated_by} · ${state.done.decisions} decisions, ${state.done.steps} steps`;
    case "error":
      return "the episode stopped on an error";
    default:
      return `ready · step ${state.step ?? 0} of ${state.header?.max_steps ?? "?"} — Step takes one decision`;
  }
}

// ------------------------------------------------------------------------------- the socket

/** The two methods and four handlers this uses of a `WebSocket`, so a test can be one. */
export interface LiveSocket {
  send(data: string): void;
  close(): void;
  onopen: ((event?: unknown) => void) | null;
  onmessage: ((event: { data: unknown }) => void) | null;
  onclose: ((event?: unknown) => void) | null;
  onerror: ((event?: unknown) => void) | null;
}

export type SocketFactory = (url: string) => LiveSocket;

/** How long a dropped connection waits before trying again. Long enough not to hammer a console
 *  that has been stopped, short enough that restarting one does not need a reload. */
export const RECONNECT_MS = 1500;

/**
 * The console, as an {@link EpisodeSource} plus the commands a viewer does not have.
 *
 * `EpisodeSource` is satisfied honestly: `list()` is the one episode in progress, `load()` waits
 * for its header, and `subscribe()` is the decisions as they are made.
 */
export class LiveSource {
  readonly kind = "live" as const;
  readonly url: string;

  private readonly make: SocketFactory;
  private socket: LiveSocket | null = null;
  private state: LiveState = EMPTY_LIVE;
  private readonly listeners = new Set<(state: LiveState) => void>();
  private readonly decisionListeners = new Set<(d: Decision) => void>();
  private timer: ReturnType<typeof setTimeout> | null = null;
  private stopped = false;
  private seen = 0;

  constructor(url: string, make?: SocketFactory) {
    this.url = url;
    this.make = make ?? ((target: string) => new WebSocket(target) as unknown as LiveSocket);
  }

  snapshot(): LiveState {
    return this.state;
  }

  subscribeState(fn: (state: LiveState) => void): () => void {
    this.listeners.add(fn);
    fn(this.state);
    return () => { this.listeners.delete(fn); };
  }

  connect(): void {
    if (this.socket !== null || this.stopped) return;
    this.set({ ...this.state, connection: "connecting" });
    let socket: LiveSocket;
    try {
      socket = this.make(this.url);
    } catch {
      // A malformed `?live=` never becomes a thrown error in a render: it becomes a page that says
      // it cannot reach a console.
      this.set({ ...this.state, connection: "closed", error: `${this.url} is not a WebSocket address` });
      return;
    }
    this.socket = socket;
    socket.onopen = () => { this.set({ ...this.state, connection: "open", error: null }); };
    socket.onmessage = (event) => { this.receive(event.data); };
    socket.onerror = () => { /* `onclose` follows and carries the retry; two paths would retry twice. */ };
    socket.onclose = () => {
      this.socket = null;
      const refused = this.state.fatal;
      this.set({ ...this.state, connection: refused ? "refused" : "closed", episode: false });
      if (!refused && !this.stopped) {
        this.timer = setTimeout(() => { this.timer = null; this.connect(); }, RECONNECT_MS);
      }
    };
  }

  close(): void {
    this.stopped = true;
    if (this.timer !== null) { clearTimeout(this.timer); this.timer = null; }
    const socket = this.socket;
    this.socket = null;
    if (socket !== null) {
      socket.onclose = null;
      socket.close();
    }
  }

  /** One message from the console. Exposed so a test can drive the reducer through the same door
   *  a socket does. */
  receive(data: unknown): void {
    let parsed: unknown;
    try {
      parsed = typeof data === "string" ? JSON.parse(data) : data;
    } catch {
      this.set({ ...this.state, error: "the console sent something that is not JSON" });
      return;
    }
    const next = applyMessage(this.state, parsed);
    this.set(next);
    for (let i = this.seen; i < next.decisions.length; i += 1) {
      const decision = next.decisions[i];
      if (decision !== undefined) this.decisionListeners.forEach((fn) => fn(decision));
    }
    this.seen = next.decisions.length;
  }

  send(message: Record<string, unknown>): boolean {
    const socket = this.socket;
    if (socket === null || this.state.connection !== "open") return false;
    socket.send(JSON.stringify(message));
    return true;
  }

  start(spec: StartSpec): boolean { return this.send({ type: "start", ...spec }); }
  step(): boolean { return this.send({ type: "step" }); }
  run(): boolean { return this.send({ type: "run" }); }
  pause(): boolean { return this.send({ type: "pause" }); }
  reset(): boolean { return this.send({ type: "reset" }); }
  ping(): boolean { return this.send({ type: "ping" }); }
  save(name?: string | null): boolean {
    return this.send(name === undefined || name === null ? { type: "save" } : { type: "save", name });
  }
  askTasks(suite: string): boolean { return this.send({ type: "tasks", suite }); }

  /**
   * Arm (or disarm) one question's answer for the next decision.
   *
   * Optimistic about nothing: the click is sent and the bars wait for the `status` that comes
   * back. An override that lit a bar the server then refused would be the worst failure this page
   * has — an operator watching the model's own answer execute while the screen says otherwise.
   */
  override(qid: string, candidate: string | null): boolean {
    return this.send({ type: "override", qid, candidate });
  }

  /** The console's picture for one camera, or null when it has not named one. */
  frameUrl(camera: "agentview" | "wrist"): string | null {
    return this.state.video?.cameras?.[camera] ?? null;
  }

  // -- EpisodeSource ---------------------------------------------------------------------

  list(): Promise<EpisodeIndexEntry[]> {
    return Promise.resolve([liveEntry(this.state)]);
  }

  /** The episode in progress, once it has one. Resolves on the first `hello` and thereafter
   *  grows — which is what `subscribe` is for. */
  load(): Promise<Episode> {
    const now = liveEpisode(this.state);
    if (now !== null) return Promise.resolve(now);
    return new Promise((resolve, reject) => {
      const off = this.subscribeState((state) => {
        const episode = liveEpisode(state);
        if (episode !== null) { off(); resolve(episode); }
        else if (state.connection === "refused") { off(); reject(new Error(state.error ?? "refused")); }
      });
    });
  }

  subscribe(_id: string, onDecision: (d: Decision) => void): () => void {
    this.decisionListeners.add(onDecision);
    return () => { this.decisionListeners.delete(onDecision); };
  }

  private set(state: LiveState): void {
    this.state = state;
    this.listeners.forEach((fn) => fn(state));
  }
}
