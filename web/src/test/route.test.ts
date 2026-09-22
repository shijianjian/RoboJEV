/** Which episode a URL means — including the link shape the first version of the site handed
 *  out, which has to keep resolving. */
import { describe, expect, it } from "vitest";
import { DEFAULT_EPISODE, episodeFor, frameOf, hashFor, routeOf } from "../data/route";

const KNOWN = ["drawer", "bowl-plate", "cookie-box", "failure"];

describe("routeOf", () => {
  it("reads today's shape", () => {
    expect(routeOf("#/drawer")).toBe("drawer");
    expect(routeOf("#/bowl-plate")).toBe("bowl-plate");
    expect(routeOf("#/drawer/")).toBe("drawer");
  });

  it("still reads the old #/replay/<id> links", () => {
    expect(routeOf("#/replay/failure")).toBe("failure");
  });

  it("reads a bare #<id>", () => {
    expect(routeOf("#cookie-box")).toBe("cookie-box");
  });

  it("names nothing for an empty or unparseable hash", () => {
    expect(routeOf("")).toBeNull();
    expect(routeOf("#")).toBeNull();
    expect(routeOf("#/")).toBeNull();
    expect(routeOf("#/a/b/c")).toBeNull();
  });
});

describe("episodeFor", () => {
  it("opens the drawer when the URL names nothing", () => {
    expect(episodeFor("", KNOWN)).toBe(DEFAULT_EPISODE);
    expect(episodeFor("#/", KNOWN)).toBe("drawer");
  });

  it("opens the episode a deep link names", () => {
    expect(episodeFor("#/failure", KNOWN)).toBe("failure");
    expect(episodeFor("#/replay/cookie-box", KNOWN)).toBe("cookie-box");
  });

  it("falls back to the default for an id nothing answers to", () => {
    expect(episodeFor("#/nope", KNOWN)).toBe(DEFAULT_EPISODE);
  });

  it("takes a named id at its word before the index has arrived", () => {
    // Otherwise the first paint is always the default and then jumps.
    expect(episodeFor("#/failure", [])).toBe("failure");
  });

  it("round-trips through hashFor", () => {
    for (const id of KNOWN) expect(episodeFor(hashFor(id), KNOWN)).toBe(id);
  });
});

describe("frameOf", () => {
  it("reads the playhead robopp keeps as ?t=, inside the hash", () => {
    expect(routeOf("#/drawer?t=72")).toBe("drawer");
    expect(frameOf("#/drawer?t=72")).toBe(72);
    expect(frameOf("#/drawer")).toBeNull();
    expect(frameOf("#/drawer?t=-3")).toBeNull();
    expect(frameOf("#/drawer?t=abc")).toBeNull();
  });

  it("writes frame 0 as no ?t= at all", () => {
    expect(hashFor("drawer", 0)).toBe("#/drawer");
    expect(hashFor("drawer", 72)).toBe("#/drawer?t=72");
    expect(frameOf(hashFor("drawer", 72))).toBe(72);
  });
});
