// Two interchangeable planners, in Pigey's shape: the planner picks tools, it
// does not command joints, and it is not allowed to declare success. Success
// comes from the tool that measured it.
//
//   local  - a deterministic sequence, used whenever no LLM key is configured.
//            Still a real planner: it chains on measured results, and it
//            retries once from a clean scene before giving up.
//   llm    - an OpenAI-compatible model over /api/plan, given the same tool
//            catalogue as function definitions. Falls back to `local` if the
//            proxy reports it is not configured.

const LOCAL_SEQUENCE = [
  { tool: "reset", args: {}, stage: 0 },
  { tool: "observe", args: {} },
  { tool: "perceive", args: {}, stage: 1 },
  { tool: "reach", args: { side: "right", target: "#1" }, stage: 2 },
  { tool: "grasp", args: { side: "right", target: "#1" }, stage: 3 },
  { tool: "verify_grasp", args: { side: "right" } },
  { tool: "handover", args: {}, stage: 4 },
  { tool: "place", args: { side: "left" }, stage: 5 },
  { tool: "verify_goal", args: {}, stage: 6 },
];

/** Function-calling schema handed to an OpenAI-compatible model. */
export function toolSchema(catalogue) {
  const params = (properties, required) => ({
    type: "object",
    properties,
    required: required ?? [],
    additionalProperties: false,
  });
  const side = { type: "string", enum: ["left", "right"] };
  const shapes = {
    reset: params({}),
    observe: params({}),
    perceive: params({}),
    reach: params({ side, target: {}, clearance: { type: "number" }, band: { type: "number" } }, ["side", "target"]),
    grasp: params({ side, target: {}, band: { type: "number" } }, ["side", "target"]),
    verify_grasp: params({ side }, ["side"]),
    handover: params({}),
    place: params({ side }, ["side"]),
    verify_goal: params({}),
    home: params({}),
  };
  return catalogue.map(({ name, describe }) => ({
    type: "function",
    function: { name, description: describe, parameters: shapes[name] ?? params({}) },
  }));
}

export class Runner {
  constructor({ tools, catalogue, ui, episode, robot = null }) {
    this.tools = tools;
    this.catalogue = catalogue;
    this.ui = ui;
    this.episode = episode;
    this.robot = robot;
  }

  async call(name, args = {}) {
    const tool = this.tools[name];
    if (!tool) throw new Error(`没有名为 ${name} 的工具。`);
    this.ui?.onToolStart?.(name, args, tool.describe);
    const started = performance.now();
    try {
      const result = await tool.run(args);
      const milliseconds = Math.round(performance.now() - started);
      this.episode?.record({ tool: name, args, ok: true, stage: result.stage, summary: result.summary, data: result.data, milliseconds }, this.robot);
      this.ui?.onToolEnd?.(name, result, { ok: true });
      return result;
    } catch (error) {
      const milliseconds = Math.round(performance.now() - started);
      const message = error instanceof Error ? error.message : String(error);
      const failedStage = tool.stage ?? this.episode?.currentStage ?? null;
      this.episode?.record({ tool: name, args, ok: false, stage: failedStage, summary: message, milliseconds }, this.robot);
      this.ui?.onToolEnd?.(name, { summary: message, data: {} }, { ok: false, stage: failedStage });
      throw error;
    }
  }
}

/** Deterministic planner: run the sequence, verify each step, retry once. */
export async function runLocal(runner, { taskText, maxAttempts = 2 } = {}) {
  const attempts = [];
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      for (const step of LOCAL_SEQUENCE) {
        const result = await runner.call(step.tool, step.args);
        if (step.stage !== undefined && result.stage !== undefined) {
          runner.ui?.onStageVerified?.(result.stage);
        }
      }
      runner.ui?.onMessage?.(
        "任务完成：瓶子已由右手交给左手并放进白盒。",
        { note: "每个阶段都由对应工具实测通过，规划器没有参与判定。" },
      );
      return { ok: true, attempts: attempts.concat([{ attempt, ok: true }]) };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      attempts.push({ attempt, ok: false, message });
      runner.ui?.onMessage?.(
        `第 ${attempt} 次尝试在「${message}」处停下。`,
        { note: attempt < maxAttempts ? "复位后重试一次。" : "不再重试，停在失败阶段。" },
      );
      if (attempt < maxAttempts) {
        try {
          await runner.call("home", {});
        } catch {
          // Recovering the arms is best-effort; the reset below is what matters.
        }
      }
    }
  }
  return { ok: false, attempts };
}

/** LLM planner: the model picks the next tool from the catalogue each step. */
export async function runLlm(runner, { taskText, maxSteps = 24, onStep } = {}) {
  const system = [
    "你是一个机器人任务编排器，运行在本地 OpenArm 仿真工作台上。",
    "你只能从给定工具里选一个执行，不能输出关节指令，也不能自己宣布成功。",
    "每个工具会返回实测结果；阶段是否通过由工具决定，不由你决定。",
    "建议顺序：reset → observe → perceive → reach → grasp → verify_grasp → handover → place → verify_goal。",
    "失败时先读错误信息里的实测数字，再决定是重试还是换成更保守的参数。",
  ].join("\n");
  const messages = [
    { role: "system", content: system },
    { role: "user", content: `任务：${taskText}\n请开始，第一步通常需要先复位场景。` },
  ];

  for (let step = 0; step < maxSteps; step++) {
    let reply;
    try {
      const response = await fetch("/api/plan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages, tools: toolSchema(runner.catalogue) }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error ?? `plan HTTP ${response.status}`);
      reply = payload.message;
    } catch (error) {
      runner.ui?.onMessage?.(`LLM 规划不可用（${error.message}），改用本地规划器继续。`);
      return runLocal(runner, { taskText });
    }

    messages.push(reply);
    const call = reply.tool_calls?.[0];
    if (!call) {
      runner.ui?.onMessage?.(reply.content || "模型没有选择工具，任务结束。");
      return { ok: false, attempts: [] };
    }
    let args = {};
    try {
      args = call.function.arguments ? JSON.parse(call.function.arguments) : {};
    } catch {
      args = {};
    }
    onStep?.(step, call.function.name, args);
    try {
      const result = await runner.call(call.function.name, args);
      messages.push({
        role: "tool",
        tool_call_id: call.id,
        content: JSON.stringify({ ok: true, summary: result.summary, data: stripImages(result.data) }),
      });
      if (result.stage !== undefined && result.stage !== null) runner.ui?.onStageVerified?.(result.stage);
      if (call.function.name === "verify_goal") {
        runner.ui?.onMessage?.(result.summary, { note: "由 verify_goal 实测判定，不是模型自述。" });
        return { ok: true, attempts: [] };
      }
    } catch (error) {
      messages.push({
        role: "tool",
        tool_call_id: call.id,
        content: JSON.stringify({ ok: false, error: error instanceof Error ? error.message : String(error) }),
      });
      runner.ui?.onMessage?.(`工具失败：${error.message}`);
    }
  }
  return { ok: false, attempts: [] };
}

/**
 * Images stay with the operator.
 *
 * The observation is shown in the interface either way; what is dropped here is
 * only the copy that would go back into the model's prompt as a base64 blob.
 * The planner gets the measured numbers instead - which is the part it has to
 * reason about - and the picture stays where a human can check it.
 */
function stripImages(data) {
  if (!data || typeof data !== "object") return data;
  const copy = { ...data };
  if (copy.image) copy.image = "<已显示在界面>";
  return copy;
}
