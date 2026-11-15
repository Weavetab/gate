import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const dir = path.dirname(fileURLToPath(import.meta.url));
const { platform, arch } = process;

const map = {
  "win32-x64": "weavetab-gate.win32-x64-msvc.node",
  "darwin-arm64": "weavetab-gate.darwin-arm64.node",
  "darwin-x64": "weavetab-gate.darwin-x64.node",
  "linux-x64": "weavetab-gate.linux-x64-gnu.node",
  "linux-arm64": "weavetab-gate.linux-arm64-gnu.node",
};

const file = map[`${platform}-${arch}`];
if (!file) {
  throw new Error(`@weavetab/gate: unsupported platform ${platform}-${arch}`);
}

let binding;
try {
  binding = require(path.join(dir, file));
} catch {
  binding = require(`@weavetab/gate-${platform}-${arch}`);
}

export const Gate = binding.Gate;
export default binding;
