/**
 * The console's controls: what to start an episode with, and — once it is open — how to drive it.
 *
 * Before an episode this is the scene and the policy: a suite, a task, a start state, which of the
 * three engines answers, and how it picks among its own candidates. During one it is five buttons
 * and a line of state. Every button is enabled by the same rule the server enforces
 * (`data/live.ts#commandsEnabled`) rather than by a guess, so a control that is offered is a
 * control that works.
 *
 * The pickers are numeric until the console has read the simulator's own task definitions and sent
 * them (`tasks`), and then they name the tasks. That order is deliberate: loading LIBERO costs
 * seconds, and a console that could not be looked at until it had would be a console that takes
 * five seconds to say hello.
 */
import { useEffect, useState } from "react";
import type { LiveState, StartSpec } from "../data/live";
import { commandsEnabled, sessionLine } from "../data/live";
import { Chip } from "./ui";

/** The selection modes a console offers. `argmax` is the model's own best answer; `sample` draws
 *  from its distribution at a temperature, which is what makes the bars worth watching. */
const MODES = ["argmax", "sample"] as const;

const POLICY_LABEL: Record<string, string> = {
  expert: "scripted expert",
  model: "local checkpoint",
  jev: "hosted Jev",
};

export function LiveControls({ state, onStart, onCommand, onSave }: {
  state: LiveState;
  onStart: (spec: StartSpec) => void;
  onCommand: (op: "step" | "run" | "pause" | "reset") => void;
  onSave: () => void;
}) {
  const config = state.config;
  const [suite, setSuite] = useState("libero_spatial");
  const [task, setTask] = useState(0);
  const [init, setInit] = useState(0);
  const [policy, setPolicy] = useState("expert");
  const [mode, setMode] = useState<string>("argmax");
  const [temperature, setTemperature] = useState("1");

  // The console's own defaults, adopted once — `robojev console --task 4 --policy expert` should
  // open a page that is already set up the way the command line asked for.
  const [seeded, setSeeded] = useState(false);
  useEffect(() => {
    if (seeded || config === null) return;
    setSeeded(true);
    setSuite(config.default.suite);
    setTask(config.default.task);
    setInit(config.default.init);
    setPolicy(config.default.policy);
    setMode(config.default.selection.startsWith("sample") ? "sample" : "argmax");
  }, [config, seeded]);

  const can = commandsEnabled(state);
  const tasks = state.tasks;
  const chosen = tasks?.find((t) => t.index === task) ?? null;
  const running = state.run === "running";
  const before = !state.episode;
  const tone = state.connection !== "open" ? "failure"
    : state.run === "running" ? "accent"
    : state.run === "done" || state.run === "error" ? "neutral" : "success";

  return (
    <div className="card livebar" data-testid="live-controls">
      <div className="card__body card__body--tight">
        <div className="livebar__row">
          {before ? (
            <>
              <label className="field">
                <span className="field__label">suite</span>
                <select className="input" value={suite} data-testid="live-suite"
                        onChange={(e) => { setSuite(e.target.value); setTask(0); setInit(0); }}>
                  {(config?.suites ?? [suite]).map((s) => <option key={s} value={s}>{s}</option>)}
                </select>
              </label>
              <label className="field field--wide">
                <span className="field__label">task</span>
                {tasks === null ? (
                  <input className="input" type="number" min={0} value={task} data-testid="live-task"
                         onChange={(e) => setTask(Math.max(0, Number(e.target.value) || 0))} />
                ) : (
                  <select className="input" value={String(task)} data-testid="live-task"
                          onChange={(e) => { setTask(Number(e.target.value)); setInit(0); }}>
                    {tasks.map((t) => (
                      <option key={t.index} value={String(t.index)}>{t.index} · {t.instruction}</option>
                    ))}
                  </select>
                )}
              </label>
              <label className="field">
                <span className="field__label">init</span>
                <input className="input" type="number" min={0}
                       max={chosen === null ? undefined : Math.max(chosen.init_states - 1, 0)}
                       value={init} data-testid="live-init"
                       onChange={(e) => setInit(Math.max(0, Number(e.target.value) || 0))} />
              </label>
              <label className="field">
                <span className="field__label">policy</span>
                <select className="input" value={policy} data-testid="live-policy"
                        onChange={(e) => setPolicy(e.target.value)}>
                  {(config?.policies ?? [policy]).map((p) => (
                    <option key={p} value={p}>{POLICY_LABEL[p] ?? p}</option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span className="field__label">answers</span>
                <select className="input" value={mode} data-testid="live-selection"
                        onChange={(e) => setMode(e.target.value)}>
                  {MODES.map((m) => <option key={m} value={m}>{m}</option>)}
                </select>
              </label>
              {mode === "sample" && (
                <label className="field">
                  <span className="field__label">temperature</span>
                  <input className="input" type="number" min={0.01} max={5} step={0.1}
                         value={temperature} data-testid="live-temperature"
                         onChange={(e) => setTemperature(e.target.value)} />
                </label>
              )}
              <button
                type="button"
                className="btn btn--primary"
                disabled={!can.start}
                data-testid="live-start"
                onClick={() => onStart({
                  suite, task, init, policy,
                  selection: mode === "argmax" ? "argmax" : "sample",
                  temperature: mode === "argmax" ? null : Number(temperature) || 1,
                })}
              >
                {state.run === "starting" ? "Starting…" : "Start"}
              </button>
            </>
          ) : (
            <>
              <button type="button" className="btn" disabled={!can.step} data-testid="live-step"
                      onClick={() => onCommand("step")} title="one decision (→ five control steps)">
                Step
              </button>
              <button type="button" className="btn btn--primary"
                      disabled={running ? !can.pause : !can.run}
                      data-testid={running ? "live-pause" : "live-run"}
                      onClick={() => onCommand(running ? "pause" : "run")}>
                {running ? "❚❚ Pause" : "▶ Run"}
              </button>
              <button type="button" className="btn" disabled={!can.reset} data-testid="live-reset"
                      onClick={() => onCommand("reset")}
                      title="end this episode and let go of the simulator">
                Reset
              </button>
              <button type="button" className="btn" disabled={!can.save} data-testid="live-save"
                      onClick={onSave}
                      title="write this episode into the replays directory as a bundle">
                Save
              </button>
              <span className="livebar__scene num" data-testid="live-scene">
                {state.header?.suite} · task {state.header?.task_index} · init{" "}
                {state.header?.init_state_index} · {state.header?.policy} · {state.header?.selection}
              </span>
            </>
          )}
          <span className="livebar__right">
            <Chip tone={tone} testId="live-connection">
              {state.connection === "open" ? state.run : state.connection}
            </Chip>
          </span>
        </div>

        <p className="footnote" data-testid="live-line">
          {sessionLine(state)}
          {can.override && (
            <span className="dim"> · click a candidate to hold it for the next decision</span>
          )}
        </p>

        {state.saved !== null && (
          <p className="footnote" data-testid="live-saved">
            saved <span className="num">{state.saved.id}</span> —{" "}
            <span className="num">{state.saved.decisions}</span> decisions,{" "}
            <span className="num">{(state.saved.bytes / 1e6).toFixed(2)}</span> MB, in the replays
            directory. It is in the strip after a refresh.
          </p>
        )}

        {state.error !== null && (
          <p className="notice notice--bad livebar__error" data-testid="live-error">{state.error}</p>
        )}
      </div>
    </div>
  );
}
