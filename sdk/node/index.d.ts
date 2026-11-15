export interface GateConfig {
  cacheDir?: string;
  cacheEntries?: number;
  threads?: number;
}

export interface OptionScore {
  label: string;
  score: number;
}

export interface Decision {
  choice: string;
  confidence: number;
  scores: OptionScore[];
  cached: boolean;
  latencyMs: number;
}

export interface YesNo {
  answer: boolean;
  probability: number;
  cached: boolean;
  latencyMs: number;
}

export interface Score {
  level: string;
  expected: number;
  scores: OptionScore[];
  cached: boolean;
  latencyMs: number;
}

export type GateInput = string | Buffer;

export class Gate {
  constructor(config?: GateConfig | null);
  cachePath(): string;
  decide(input: GateInput, choices: string[]): Decision;
  yesNo(input: GateInput, question: string): YesNo;
  score(input: GateInput, levels: string[]): Score;
}
