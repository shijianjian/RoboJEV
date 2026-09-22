import type { QposTable } from "../data/types";
import type { SceneFiles } from "./SceneView";

/** Where one bundle's files are, as absolute URLs (`ReplaySource.files`): the MuJoCo worker
 *  fetches the scene, and a worker resolves a relative URL against its own script. */
export interface BundleFiles {
  media: Record<string, string>;
  poster: string | null;
  qpos: { url: string; spec: QposTable } | null;
  scene: SceneFiles | null;
  episodeJson: string;
}
