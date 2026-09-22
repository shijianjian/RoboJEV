/** What a suite and an engine are called on screen. */

export const SUITE_LABEL: Record<string, string> = {
  libero_spatial: "LIBERO-Spatial",
  libero_object: "LIBERO-Object",
  libero_goal: "LIBERO-Goal",
  libero_10: "LIBERO-Long",
  libero_90: "LIBERO-90",
};

export const suiteLabel = (id: string) => SUITE_LABEL[id] ?? id;

/** A LIBERO suite's saved start states, until the console has said how many a task has. */
export const DEFAULT_INIT_STATES = 50;

/** The engine that answered a recorded run, as its run cell names it. */
export function policyName(id: string | null | undefined): string {
  const names: Record<string, string> = { robojev: "RoboJEV", model: "RoboJEV", jev: "Jev", expert: "scripted expert" };
  return id != null && Object.hasOwn(names, id) ? names[id] : id ?? "";
}
