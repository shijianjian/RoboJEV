import { useState } from "react";
import { TaskRow, type SidebarTask } from "../ui/TaskRow";

/** One simulator family and the suites it runs, as the sidebar groups them. */
export interface SceneFamily {
  simulator: string;
  heading: string;
  suites: { id: string; displayName: string; nTasks: number }[];
}

/**
 * robopp's scene sidebar (`app/run/SceneSidebar.tsx`), unchanged: which dataset, then which task.
 *
 * It is the Replay sidebar's Datasets group with one difference, and the difference is the whole
 * point: these are not filters. A run is recorded on exactly one task of exactly one suite, so
 * every row here is a radio rather than a checkbox - which is not only the right semantics for a
 * screen reader but the reason the arrow keys work at all, since a browser moves the selection
 * within a named radio group by itself.
 *
 * Everything else is deliberately the same as the other tab: the same disclosure per simulator
 * family, the same row markup, the same 32px thumbnails (see `../ui/TaskRow`), so that the two
 * sidebars read as one control the site uses twice rather than as two lists that happen to hold
 * suites.
 */
export function SceneSidebar({ families, suite, suiteLabel, tasks, taskIndex, onSuite, onTask }: {
  families: SceneFamily[];
  suite: string;
  suiteLabel: string;
  /** Every task of the selected suite. A task with no catalogue row is still listed: it has no
   *  picture and no instruction, but it is a run the server would accept. */
  tasks: SidebarTask[];
  taskIndex: number;
  onSuite: (suite: string) => void;
  onTask: (taskIndex: number) => void;
}) {
  // Only what the reader has folded away. Every family starts open: there are two of them, and a
  // sidebar that opens closed is a sidebar whose first click tells you nothing.
  const [folded, setFolded] = useState<ReadonlySet<string>>(() => new Set<string>());
  // The family holding the selected suite is never folded away - what is selected has to be
  // visible, or the reader cannot tell where they are.
  const selectedFamily = families.find((f) => f.suites.some((s) => s.id === suite))?.simulator ?? null;
  const isOpen = (id: string) => id === selectedFamily || !folded.has(id);

  /**
   * The disclosure, controlled: `open` alone is not enough to hold one open.
   *
   * Clicking a `<summary>` closes the `<details>` in the DOM before React hears about it, and a
   * re-render that computes the same `open={true}` writes nothing back - so a forced-open family
   * could be folded away by a click, hiding the selection. Put back in the toggle event, which is
   * where the DOM says what it did. (The same rule, and the same fix, as the Replay sidebar.)
   */
  const onToggle = (id: string, el: HTMLDetailsElement) => {
    if (id === selectedFamily) {
      if (!el.open) el.open = true;
      return;
    }
    setFolded((prev) => {
      const next = new Set(prev);
      if (el.open) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  return (
    <div className="replay-side" data-testid="scene-sidebar">
      <section className="replay-group" aria-labelledby="run-datasets">
        <h2 className="sidenav__heading" id="run-datasets">Datasets</h2>
        {families.map((family) => (
          <details
            key={family.simulator}
            className="replay-fold"
            open={isOpen(family.simulator)}
            onToggle={(e) => onToggle(family.simulator, e.currentTarget)}
            data-testid={`family-${family.simulator}`}
          >
            <summary className="replay-row__summary">
              <span className="replay-row__name">{family.heading}</span>
              <span className="replay-row__count u-dim">
                <span className="u-num">{family.suites.length}</span> {family.suites.length === 1 ? "suite" : "suites"}
              </span>
            </summary>
            {/* One choice across every family, so one radio group across all of them: the arrow
                keys walk the whole list of suites, which is what a reader expects of a list of
                things exactly one of which is true. */}
            <div className="replay-list" role="radiogroup" aria-label={`datasets under ${family.heading}`}>
              {family.suites.map((s) => (
                <div key={s.id}>
                  <label className="replay-check">
                    <input
                      type="radio"
                      name="run-suite"
                      className="checkbox"
                      checked={s.id === suite}
                      onChange={() => onSuite(s.id)}
                      data-testid={`suite-pick-${s.id}`}
                    />
                    <span className="replay-check__body">
                      <span className="replay-check__name">{s.displayName}</span>
                      <span className="replay-check__sub u-dim u-num">{s.nTasks} tasks</span>
                    </span>
                  </label>
                  {/* The tasks of the selected suite, under the suite they belong to - the one
                      place they mean anything, since a task index is only a scene under a suite. */}
                  {s.id === suite && tasks.length > 0 && (
                    <div
                      className="replay-tasks"
                      role="radiogroup"
                      aria-label={`tasks in ${suiteLabel}`}
                      data-testid="scene-tasks"
                    >
                      {tasks.map((task) => (
                        <TaskRow
                          key={task.taskIndex}
                          task={task}
                          control="radio"
                          name="run-task"
                          checked={task.taskIndex === taskIndex}
                          onSelect={() => onTask(task.taskIndex)}
                          testId={`task-pick-${task.taskIndex}`}
                        />
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </details>
        ))}
      </section>
    </div>
  );
}
