import {describe, expect, it} from "vitest";
import {align, apiUrl, display, type History} from "./api";

describe("viewer data boundary", () => {
  it("encodes run and metric names without changing query semantics", () => {
    const url = apiUrl("/api/history", {run: "a&b", metrics: "loss/train + eval", max_points: 1000});
    const query = new URL(url, "http://localhost").searchParams;
    expect(query.get("run")).toBe("a&b");
    expect(query.get("metrics")).toBe("loss/train + eval");
    expect(query.get("max_points")).toBe("1000");
  });

  it("preserves missing points instead of filling them with previous values", () => {
    const first: History = {x: [0, 2], series: [{metric: "loss", values: [1, null], nonfinite: 1}], total: 2, warnings: []};
    const second: History = {x: [1, 2], series: [{metric: "loss", values: [2, 3], nonfinite: 0}], total: 2, warnings: []};
    expect(align([first, second], "loss")).toEqual({x: [0, 1, 2], ys: [[1, null, null], [null, 2, 3]]});
  });

  it("displays nonfinite values and exact large integers explicitly", () => {
    expect(display({"$et.nonfinite": "NaN"})).toBe("NaN");
    expect(display({"$et.integer": "9007199254740993"})).toBe("9007199254740993");
    expect(display({loss: 0})).toContain('"loss": 0');
  });
});
