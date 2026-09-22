import { useEffect, useRef } from "react";

/** The shortest gap between two picture requests: the simulator makes twenty a second. */
const FRAME_GAP_MS = 45;
/** How long to wait after a picture request that failed (a console between episodes has none). */
const FRAME_RETRY_MS = 600;

/**
 * One camera, refreshed a picture at a time: the next request goes out when the previous
 * picture has painted, so a slow machine shows fewer frames rather than queueing requests.
 */
export function LiveFrame({ url, alt, testId, className }: {
  url: string | null;
  alt: string;
  testId: string;
  className: string;
}) {
  const img = useRef<HTMLImageElement>(null);
  useEffect(() => {
    const element = img.current;
    if (element === null || url === null) return;
    let alive = true;
    let seq = 0;
    let timer = 0;
    const next = () => {
      if (!alive) return;
      seq += 1;
      element.src = `${url}${url.includes("?") ? "&" : "?"}n=${seq}`;
    };
    const again = (ms: number) => {
      if (!alive) return;
      window.clearTimeout(timer);
      timer = window.setTimeout(next, ms);
    };
    const onLoad = () => again(FRAME_GAP_MS);
    const onError = () => again(FRAME_RETRY_MS);
    element.addEventListener("load", onLoad);
    element.addEventListener("error", onError);
    next();
    return () => {
      alive = false;
      window.clearTimeout(timer);
      element.removeEventListener("load", onLoad);
      element.removeEventListener("error", onError);
    };
  }, [url]);
  return <img ref={img} className={className} alt={alt} data-testid={testId} width={256} height={256} />;
}
