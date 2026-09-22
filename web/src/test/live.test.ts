/**
 * The live source: every message the console can send, folded into what the page shows.
 *
 * `applyMessage` is a pure function, which is the whole reason the live half can be checked
 * without a browser, a socket or a simulator. What is pinned here is the handful of rules that
 * would each be an invisible lie on screen if they broke: a new `hello` clears the previous
 * episode, the armed overrides are the *server's* and never this page's memory of a click,
 * `liveEpisode` builds the exact `Episode` shape the replay components already read, and a fatal
 * refusal does not reconnect in a loop.
 */
import { describe, expect, it } from "vitest";
import {
  applyMessage, commandsEnabled, EMPTY_LIVE, liveEntry, liveEpisode, LIVE_ID, LiveSource,
  sessionLine, type LiveSocket, type LiveState,
} from "../data/live";
import { liveUrl } from "../data/source";

const HEADER = {
  schema_version: 1,
  id: "live-2026-09-22T09-14-03",
  suite: "libero_spatial",
  task_index: 4,
  init_state_index: 1,
  instruction: "pick up the black bowl in the top drawer and place it on the plate",
  title: "pick up the black bowl in the top drawer and place it on the plate",
  note: "",
  policy: "expert",
  checkpoint_revision: "scripted-v2",
  checkpoint_repo: null,
  questions_version: "v2",
  selection: "argmax",
  control_rate: 20,
  wait_steps: 0,
  execute_steps: 5,
  max_steps: 220,
  max_decisions: 44,
};

const VIDEO = {
  kind: "poll",
  cameras: { agentview: "http://127.0.0.1:8765/frame/agentview.png",
             wrist: "http://127.0.0.1:8765/frame/wrist.png" },
  rate: 20,
};

function decision(index: number, extra: Record<string, unknown> = {}) {
  return {
    index,
    step: index,
    control_step: index * 5,
    t: index * 0.25,
    state: `x: +1.0 cm (decision ${index})`,
    questions: {
      move_x: { candidates: [{ id: "-", p: 0.1 }, { id: "hold", p: 0.2 }, { id: "+", p: 0.7 }],
                choice: "+", overridden: false },
    },
    grip_latch: { asked: false, closed: false, refused: false },
    grounding: null,
    subgoal: "reach",
    substage: null,
    target: "bowl_1",
    destination: "plate_1",
    waypoint: null,
    waypoint_cm: null,
    rim_candidates: [],
    rim_chosen: null,
    action: [0, 0, 0, 0, 0, 0, -1],
    forward_passes: null,
    candidate_paths: null,
    step_sizes_cm: null,
    ...extra,
  };
}

function fold(messages: unknown[], from: LiveState = EMPTY_LIVE): LiveState {
  return messages.reduce<LiveState>((state, message) => applyMessage(state, message), from);
}

const CONFIG = {
  type: "config", protocol: 1, policies: ["expert"], suites: ["libero_spatial"],
  default: { suite: "libero_spatial", task: 0, init: 0, policy: "expert", selection: "argmax" },
  video: VIDEO, state: "idle", replays: "replays/",
};

describe("applyMessage", () => {
  it("takes the console's own defaults and picture stream off config", () => {
    const state = fold([CONFIG]);
    expect(state.connection).toBe("open");
    expect(state.config?.policies).toEqual(["expert"]);
    expect(state.video?.cameras.agentview).toContain("/frame/agentview.png");
    expect(state.run).toBe("idle");
  });

  it("opens an episode on hello and clears everything the previous one left", () => {
    const first = fold([CONFIG, { type: "hello", episode: HEADER, video: VIDEO },
                        { type: "decision", decision: decision(0) },
                        { type: "done", success: true, terminated_by: "success", steps: 5,
                          decisions: 1, error: null }]);
    expect(first.decisions).toHaveLength(1);
    const second = applyMessage(first, { type: "hello", episode: { ...HEADER, id: "live-2" }, video: VIDEO });
    expect(second.decisions).toEqual([]);
    expect(second.done).toBeNull();
    expect(second.saved).toBeNull();
    expect(second.header?.id).toBe("live-2");
    expect(second.episode).toBe(true);
  });

  it("places a decision by its index, so a replayed one is not drawn twice", () => {
    const state = fold([{ type: "decision", decision: decision(0) },
                        { type: "decision", decision: decision(1) },
                        { type: "decision", decision: decision(1) }]);
    expect(state.decisions.map((d) => d.index)).toEqual([0, 1]);
  });

  it("ignores a decision with no index rather than putting a hole in the list", () => {
    const state = applyMessage(EMPTY_LIVE, { type: "decision", decision: { state: "x" } });
    expect(state.decisions).toEqual([]);
  });

  it("takes the armed overrides from status and from nowhere else", () => {
    const armed = fold([{ type: "status", state: "paused", step: 5, decisions: 1,
                          overrides: { move_x: "+" }, episode: true }]);
    expect(armed.overrides).toEqual({ move_x: "+" });
    // A status that carries none clears them: the console has spent or dropped the override, and a
    // bar still lit would be claiming an answer is held that is not.
    const spent = applyMessage(armed, { type: "status", state: "paused", step: 10, episode: true });
    expect(spent.overrides).toEqual({});
  });

  it("keeps an error until something works, and remembers a fatal one", () => {
    const refused = fold([{ type: "error", message: "another window has it", fatal: true }]);
    expect(refused.error).toBe("another window has it");
    expect(refused.fatal).toBe(true);
    const still = applyMessage(refused, { type: "status", state: "idle" });
    expect(still.error).toBe("another window has it");
    const cleared = applyMessage(still, { type: "decision", decision: decision(0) });
    expect(cleared.error).toBeNull();
    expect(cleared.fatal).toBe(true);
  });

  it("records what was saved and what the tasks are called", () => {
    const state = fold([
      { type: "saved", id: "live-1", path: "/r/live-1", url: "replays/live-1/", decisions: 3,
        bytes: 2048, success: true },
      { type: "tasks", suite: "libero_spatial", tasks: [{ index: 0, instruction: "pick up", init_states: 50 }] },
    ]);
    expect(state.saved?.id).toBe("live-1");
    expect(state.tasks?.[0].instruction).toBe("pick up");
  });

  it("changes nothing for a message it does not know, or for something that is not one", () => {
    const state = fold([CONFIG]);
    expect(applyMessage(state, { type: "telemetry", fps: 20 })).toBe(state);
    expect(applyMessage(state, "hello")).toBe(state);
    expect(applyMessage(state, null)).toBe(state);
  });
});

describe("liveEpisode", () => {
  it("is nothing until there is a header", () => {
    expect(liveEpisode(EMPTY_LIVE)).toBeNull();
  });

  it("is the Episode shape the replay components already read", () => {
    const state = fold([CONFIG, { type: "hello", episode: HEADER, video: VIDEO },
                        { type: "decision", decision: decision(0) },
                        { type: "decision", decision: decision(1) },
                        { type: "status", state: "paused", step: 10, episode: true }]);
    const episode = liveEpisode(state)!;
    expect(episode.id).toBe(HEADER.id);
    expect(episode.instruction).toBe(HEADER.instruction);
    expect(episode.control_rate).toBe(20);
    expect(episode.max_decisions).toBe(44);
    expect(episode.decisions).toHaveLength(2);
    // A live episode has no mp4 -- it is still happening -- so the page shows the console's own
    // picture stream in its place.
    expect(episode.media).toEqual({});
    expect(episode.total_frames).toBe(10);
    expect(episode.duration_s).toBe(0.5);
    expect(episode.terminated_by).toBe("live");
  });

  it("takes the outcome from done once the episode has one", () => {
    const state = fold([{ type: "hello", episode: HEADER, video: VIDEO },
                        { type: "decision", decision: decision(0) },
                        { type: "done", success: true, terminated_by: "success", steps: 144,
                          decisions: 29, error: null }]);
    const episode = liveEpisode(state)!;
    expect(episode.success).toBe(true);
    expect(episode.terminated_by).toBe("success");
    expect(episode.steps).toBe(144);
  });
});

describe("the strip item and the controls", () => {
  it("names the console before it has an episode and the task once it has", () => {
    expect(liveEntry(EMPTY_LIVE).id).toBe(LIVE_ID);
    expect(liveEntry(EMPTY_LIVE).max_decisions).toBe(0);
    const state = fold([{ type: "hello", episode: HEADER, video: VIDEO }]);
    expect(liveEntry(state).instruction).toBe(HEADER.instruction);
    expect(liveEntry(state).task_index).toBe(4);
  });

  it("offers exactly the commands the console would accept", () => {
    const offline = commandsEnabled(EMPTY_LIVE);
    expect(offline.start).toBe(false);
    expect(offline.step).toBe(false);

    const idle = commandsEnabled(fold([CONFIG]));
    expect(idle.start).toBe(true);
    expect(idle.step).toBe(false);
    expect(idle.reset).toBe(false);

    const paused = commandsEnabled(fold([CONFIG, { type: "hello", episode: HEADER, video: VIDEO },
                                         { type: "status", state: "paused", episode: true }]));
    expect(paused).toMatchObject({ start: false, step: true, run: true, pause: false,
                                   reset: true, save: true, override: true });

    const running = commandsEnabled(fold([CONFIG, { type: "hello", episode: HEADER, video: VIDEO },
                                          { type: "status", state: "running", episode: true }]));
    expect(running).toMatchObject({ step: false, run: false, pause: true, override: false });
  });

  it("says where the session is in one line", () => {
    expect(sessionLine(EMPTY_LIVE)).toContain("connecting");
    expect(sessionLine(fold([CONFIG]))).toContain("pick a scene");
    expect(sessionLine(fold([CONFIG, { type: "status", state: "running", step: 85, episode: true },
                             { type: "hello", episode: HEADER, video: VIDEO },
                             { type: "status", state: "running", step: 85, episode: true }])))
      .toBe("running · step 85 of 220");
    expect(sessionLine(fold([CONFIG, { type: "status", state: "done" },
                             { type: "done", success: true, terminated_by: "success", steps: 144,
                               decisions: 29, error: null }]))).toContain("29 decisions");
  });
});

// ------------------------------------------------------------------------------ the socket

class FakeSocket implements LiveSocket {
  static last: FakeSocket | null = null;
  static built = 0;
  sent: string[] = [];
  closed = false;
  onopen: ((event?: unknown) => void) | null = null;
  onmessage: ((event: { data: unknown }) => void) | null = null;
  onclose: ((event?: unknown) => void) | null = null;
  onerror: ((event?: unknown) => void) | null = null;

  constructor(readonly url: string) {
    FakeSocket.last = this;
    FakeSocket.built += 1;
  }

  send(data: string): void { this.sent.push(data); }
  close(): void { this.closed = true; }
  open(): void { this.onopen?.(); }
  deliver(message: unknown): void { this.onmessage?.({ data: JSON.stringify(message) }); }
  drop(): void { this.onclose?.(); }
  parsed(): Record<string, unknown>[] { return this.sent.map((s) => JSON.parse(s)); }
}

function connected(): { source: LiveSource; socket: FakeSocket } {
  FakeSocket.built = 0;
  const source = new LiveSource("ws://127.0.0.1:8765/ws", (url) => new FakeSocket(url));
  source.connect();
  const socket = FakeSocket.last!;
  socket.open();
  socket.deliver(CONFIG);
  return { source, socket };
}

describe("LiveSource", () => {
  it("connects, reports every state it goes through, and sends nothing before it is open", () => {
    const seen: string[] = [];
    const source = new LiveSource("ws://x/ws", (url) => new FakeSocket(url));
    source.subscribeState((s) => seen.push(s.connection));
    expect(source.step()).toBe(false);                       // no socket yet: refused, not thrown
    source.connect();
    FakeSocket.last!.open();
    expect(seen).toEqual(["connecting", "connecting", "open"]);
    source.close();
  });

  it("speaks the commands PROTOCOL.md lists", () => {
    const { source, socket } = connected();
    source.start({ suite: "libero_spatial", task: 4, init: 1, policy: "expert", selection: "argmax" });
    source.step();
    source.run();
    source.pause();
    source.override("grip", "true");
    source.override("grip", null);
    source.save("my-episode");
    source.save();
    source.reset();
    source.ping();
    source.askTasks("libero_spatial");
    expect(socket.parsed().map((m) => m.type)).toEqual([
      "start", "step", "run", "pause", "override", "override", "save", "save", "reset", "ping",
      "tasks",
    ]);
    expect(socket.parsed()[0]).toMatchObject({ task: 4, init: 1, policy: "expert" });
    expect(socket.parsed()[4]).toEqual({ type: "override", qid: "grip", candidate: "true" });
    expect(socket.parsed()[5]).toEqual({ type: "override", qid: "grip", candidate: null });
    expect(socket.parsed()[6]).toEqual({ type: "save", name: "my-episode" });
    expect(socket.parsed()[7]).toEqual({ type: "save" });
    source.close();
  });

  it("does not light a bar until the console says the override is held", () => {
    const { source, socket } = connected();
    socket.deliver({ type: "hello", episode: HEADER, video: VIDEO });
    socket.deliver({ type: "status", state: "paused", step: 0, episode: true });
    source.override("move_x", "+");
    expect(source.snapshot().overrides).toEqual({});          // sent, not assumed
    socket.deliver({ type: "status", state: "paused", step: 0, overrides: { move_x: "+" }, episode: true });
    expect(source.snapshot().overrides).toEqual({ move_x: "+" });
    // And the console's refusal leaves nothing held.
    socket.deliver({ type: "error", message: "override 'elbow': no such question" });
    socket.deliver({ type: "status", state: "paused", step: 0, episode: true });
    expect(source.snapshot().overrides).toEqual({});
    expect(source.snapshot().error).toContain("no such question");
    source.close();
  });

  it("hands each new decision to its subscribers exactly once", () => {
    const { source, socket } = connected();
    const got: number[] = [];
    source.subscribe("live", (d) => got.push(d.index));
    socket.deliver({ type: "hello", episode: HEADER, video: VIDEO });
    socket.deliver({ type: "decision", decision: decision(0) });
    socket.deliver({ type: "decision", decision: decision(1) });
    socket.deliver({ type: "status", state: "paused", step: 10, episode: true });
    expect(got).toEqual([0, 1]);
    source.close();
  });

  it("names the console's pictures once it has said where they are", () => {
    const { source, socket } = connected();
    expect(source.frameUrl("agentview")).toBe(VIDEO.cameras.agentview);
    socket.deliver({ type: "hello", episode: HEADER, video: { ...VIDEO, cameras: {} } });
    expect(source.frameUrl("wrist")).toBeNull();
    source.close();
  });

  it("survives something that is not JSON rather than throwing into a render", () => {
    const { source, socket } = connected();
    socket.onmessage?.({ data: "<html>404</html>" });
    expect(source.snapshot().error).toContain("not JSON");
    source.close();
  });

  it("reconnects when the console goes away, and does not when it has refused", async () => {
    const { source, socket } = connected();
    socket.drop();
    expect(source.snapshot().connection).toBe("closed");
    await new Promise((r) => setTimeout(r, 1700));
    expect(FakeSocket.built).toBe(2);
    source.close();

    FakeSocket.built = 0;
    const refused = new LiveSource("ws://x/ws", (url) => new FakeSocket(url));
    refused.connect();
    FakeSocket.last!.open();
    FakeSocket.last!.deliver({ type: "error", message: "another window", fatal: true });
    FakeSocket.last!.drop();
    expect(refused.snapshot().connection).toBe("refused");
    await new Promise((r) => setTimeout(r, 1700));
    expect(FakeSocket.built).toBe(1);
    refused.close();
  }, 10000);

  it("resolves load() with the episode the console has begun", async () => {
    const { source, socket } = connected();
    const waiting = source.load();
    socket.deliver({ type: "hello", episode: HEADER, video: VIDEO });
    const episode = await waiting;
    expect(episode.id).toBe(HEADER.id);
    expect((await source.list())[0].id).toBe(LIVE_ID);
    source.close();
  });
});

describe("liveUrl", () => {
  it("is null for the static site, which therefore probes nothing", () => {
    expect(liveUrl("", null)).toBeNull();
    expect(liveUrl("?t=3", null)).toBeNull();
  });

  it("takes the console's own address off the page it serves", () => {
    expect(liveUrl("", { ws: "ws://127.0.0.1:8765/ws" })).toBe("ws://127.0.0.1:8765/ws");
  });

  it("lets ?live= win, so one deployed build can be pointed anywhere without rebuilding", () => {
    expect(liveUrl("?live=ws://other:9000/ws", { ws: "ws://127.0.0.1:8765/ws" }))
      .toBe("ws://other:9000/ws");
  });

  it("ignores an empty one rather than connecting to nothing", () => {
    expect(liveUrl("?live=", null)).toBeNull();
    expect(liveUrl("", { ws: "" })).toBeNull();
    expect(liveUrl("", { ws: 7 })).toBeNull();
  });
});
