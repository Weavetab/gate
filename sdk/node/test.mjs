import assert from "node:assert/strict";
import test from "node:test";
import { Gate } from "./index.js";

test("rejects single choice", () => {
  const g = new Gate();
  assert.throws(() => g.decide("hello world", ["only_one"]));
});
