import { useMemo, useState } from "react";
import type { SceneProgress } from "./mujoco";

/**
 * What a scene view is doing, in the words the status chip shows and the progress line reads.
 *
 * `text` is the viewer's permanent status line below the canvas (and what the e2e specs read);
 * `note` is the short form that sits beside the progress line over the canvas, which only
 * appears while `busy` - because that is exactly when what you are looking at is *not* what the
 * page says it is showing: the previous scene is kept up while the next one compiles.
 */
export interface SceneStatusState {
  text: string;
  note: string;
  tone: "success" | "failure" | "queued";
  busy: boolean;
}

/** Stable across renders, so a load effect can depend on it without restarting itself. */
export interface SceneStatusControls {
  startLoad: () => void;
  progress: (p: SceneProgress) => void;
  ready: () => void;
  failed: (e: unknown) => void;
}

const LOADING: SceneStatusState = { text: "loading MuJoCo", note: "loading MuJoCo…", tone: "queued", busy: true };

/**
 * The same wait, said honestly, for a scene the registry marks heavy.
 *
 * A LIBERO bundle compiles in about a second and the sweep line is enough on its own; a RoboCasa
 * kitchen is ~70 MB and 13-14 s in the browser (measured on two loads: ~4 s fetching the assets,
 * ~9 s inside `mj_loadXML`, which reports nothing from inside itself), which is long enough that
 * a line reading only
 * "loading MuJoCo…" looks like a page that has stopped. `text` is left exactly as it was - it is
 * the permanent status chip the e2e specs read - and the admission goes in `note`, which is the
 * line actually over the canvas while the wait is happening.
 */
const LOADING_HEAVY: SceneStatusState = {
  text: "loading MuJoCo", note: "loading MuJoCo — kitchens take 10-15 seconds…", tone: "queued", busy: true };

export function useSceneStatus(heavy = false): [SceneStatusState, SceneStatusControls] {
  const [state, setState] = useState<SceneStatusState>(heavy ? LOADING_HEAVY : LOADING);
  const controls = useMemo<SceneStatusControls>(() => ({
    startLoad: () => setState(heavy ? LOADING_HEAVY : LOADING),
    progress: (p) => setState(
      // Fetching and compiling are one wait as far as the reader is concerned, so the count is
      // shown as progress *through* the compile rather than as a separate download step. It
      // stops updating the moment the compile starts, which is the honest thing for it to do:
      // `mj_loadXML` reports nothing from inside itself, so there is nothing to report until it
      // returns. (The page is no longer frozen while it runs — the compile is in a worker — but
      // the worker has no more to say about it than the main thread did.)
      p.phase === "fetching" && p.total > 0
        ? { text: `compiling scene… ${p.loaded}/${p.total} assets`, note: `${p.loaded}/${p.total} assets`, tone: "queued", busy: true }
        // The compile itself is the part with nothing to report, and on a kitchen it is the part
        // that lasts: that is where the admission earns its place.
        : { text: "compiling…", note: heavy ? "compiling… kitchens take 10-15 seconds" : "compiling…", tone: "queued", busy: true },
    ),
    ready: () => setState({ text: "ready", note: "", tone: "success", busy: false }),
    failed: (e) => {
      const text = "error: " + (e instanceof Error ? e.message : String(e));
      setState({ text, note: "", tone: "failure", busy: false });
    },
    // `heavy` is a property of the run being replayed, so this is as stable as it ever was: the
    // load effect that depends on these controls does not restart while one run is on screen.
  }), [heavy]);
  return [state, controls];
}

/**
 * The progress line over the canvas: three pixels on the stage's top edge, sweeping.
 *
 * It is deliberately not a React animation, a spinner driven by rAF, or anything else that needs
 * the main thread: compiling a LIBERO scene in MuJoCo WASM blocks it outright for seconds, and
 * every JS-driven indicator freezes mid-stride exactly when the user most needs to be told that
 * something is still happening. A CSS keyframe animation on `transform` runs on the compositor
 * and keeps sweeping straight through the block. See `scene-loading.css`.
 */
export function SceneProgressLine({ state }: { state: SceneStatusState }) {
  if (!state.busy) return null;
  return (
    <div className="scene-progress" data-testid="scene-loading">
      <div className="scene-progress__track">
        <span className="scene-progress__sweep" aria-hidden />
      </div>
      {state.note ? <span className="scene-progress__note" role="status" aria-live="polite">{state.note}</span> : null}
    </div>
  );
}
