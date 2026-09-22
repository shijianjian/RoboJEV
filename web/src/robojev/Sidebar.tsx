import { RunCell, type RunCellRun } from "../ui/RunCell";
import { SceneSidebar, type SceneFamily } from "./SceneSidebar";
import type { SidebarTask } from "../ui/TaskRow";

/** One run in the Runs tab, plus the scene it came from. */
export type RailRun = RunCellRun & { suite: string; suiteLabel: string; taskIndex: number };

/**
 * The app's one navigation: robopp's scene sidebar, with two tabs over it.
 *
 * **Runs** is the runs robopp lists in its rail, as robopp's `RunCell`s in one column - every
 * recorded bundle, each labelled by its task (a live episode is the Dataset tab's until it is
 * saved) - and it leads to the replay page. **Dataset** is robopp's sidebar as the Run and RoboJEV tabs have it -
 * the datasets, and under the chosen one its tasks as `TaskRow`s - and it leads to the inference
 * page. The tab strip is
 * robopp's `stage-tabs` markup; each tab is a link, so the address says which page is open.
 */
export function Sidebar({
  tab, tabs: shown, datasetHref, runsHref, families, suite, suiteLabel, tasks, taskIndex, onSuite, onTask,
  runs, selectedRun, onSelectRun, onGoToScene,
}: {
  tab: "dataset" | "runs";
  /** The tabs this build has: both locally, Runs alone on GitHub Pages. */
  tabs: readonly ("runs" | "dataset")[];
  datasetHref: string;
  runsHref: string;
  families: SceneFamily[];
  suite: string;
  suiteLabel: string;
  tasks: SidebarTask[];
  taskIndex: number;
  onSuite: (suite: string) => void;
  onTask: (task: number) => void;
  runs: RailRun[];
  selectedRun: string | null;
  onSelectRun: (id: string) => void;
  onGoToScene: (suite: string, task: number, state: number) => void;
}) {
  const tabs = [
    { id: "runs" as const, label: "Runs", href: runsHref },
    { id: "dataset" as const, label: "Dataset", href: datasetHref },
  ].filter((t) => shown.includes(t.id));
  return (
    <div className="app-side" data-testid="app-side">
      <div className="stage-tabs app-side__tabs">
        <span className="stage-tabs__strip" role="tablist" aria-label="pages">
          {tabs.map((t) => (
            <span key={t.id} role="presentation" className={`stage-tab${tab === t.id ? " is-active" : ""}`}>
              <a
                role="tab"
                aria-selected={tab === t.id}
                className="stage-tab__label"
                href={t.href}
                data-testid={`side-tab-${t.id}`}
              >
                {t.label}
              </a>
            </span>
          ))}
        </span>
      </div>
      {tab === "dataset" ? (
        <SceneSidebar
          families={families}
          suite={suite}
          suiteLabel={suiteLabel}
          tasks={tasks}
          taskIndex={taskIndex}
          onSuite={onSuite}
          onTask={onTask}
        />
      ) : (
        <ul className="run-cells app-side__runs" data-testid="recent-runs">
          {runs.map((run) => (
            <RunCell
              key={run.id}
              run={run}
              where={
                <button
                  type="button"
                  className="run-cell__scene"
                  onClick={() => onGoToScene(run.suite, run.taskIndex, run.initStateIndex)}
                  data-testid={`scene-of-${run.id}`}
                >
                  {run.suiteLabel} <span className="u-dim">/</span> task <span className="u-num">{run.taskIndex}</span>
                </button>
              }
              whereLabel={`${run.suiteLabel} task ${run.taskIndex}`}
              selected={run.id === selectedRun}
              onSelect={onSelectRun}
            />
          ))}
        </ul>
      )}
    </div>
  );
}
