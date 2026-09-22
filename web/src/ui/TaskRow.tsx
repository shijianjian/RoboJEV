
/** One task of a suite, as a sidebar lists it. The thumbnail is already a URL: the storage lookup
 *  is async and belongs on the server. A task nobody has exported has neither. */
export interface SidebarTask {
  taskIndex: number;
  instruction: string | null;
  thumbnail: string | null;
}

/**
 * A task in a sidebar: its catalogue thumbnail, its index, and what it asks for.
 *
 * The two tabs pick tasks for different reasons and so with different controls - Replay ticks any
 * number of them as a filter, the Run tab chooses exactly one to record a run on - but they are
 * looking at the same thing, and a reader who has learned to read one of these rows should not
 * have to learn the other. So the row is one component with the control as a parameter: a
 * checkbox, or a radio in a named group (which is also what gives the Run tab's list its
 * arrow-key navigation, for free and correctly, rather than a hand-rolled roving tabindex).
 *
 * The 32px thumbnail is the point of the row. `task 7` distinguishes nothing on a suite of ten
 * kitchen scenes, and the instruction only says which one once you are reading closely.
 */
export function TaskRow({ task, control, name, checked, onSelect, testId }: {
  task: SidebarTask;
  control: "checkbox" | "radio";
  /** The radio group this row belongs to. Required for `radio`, ignored for `checkbox`. */
  name?: string;
  checked: boolean;
  onSelect: () => void;
  testId: string;
}) {
  return (
    <label className="replay-check replay-check--task">
      <input
        type={control}
        name={name}
        className="checkbox"
        checked={checked}
        onChange={() => onSelect()}
        data-testid={testId}
      />
      {task.thumbnail !== null
        ? <img className="replay-task__thumb" src={task.thumbnail} alt="" loading="lazy" decoding="async" />
        : <span className="replay-task__thumb skeleton" aria-hidden />}
      <span className="replay-check__body">
        <span className="replay-check__name u-num">task {task.taskIndex}</span>
        {task.instruction !== null && (
          <span className="replay-check__sub u-dim">{task.instruction}</span>
        )}
      </span>
    </label>
  );
}
