---
layout: post
title: "What Makes an AI System Agentic?"
date: 2026-09-02 09:00:00 +1000
description: A practical guide to the difference between models, workflows, and agents—and to the feedback loop, tools, state, and safeguards that make agency possible.
tags: [artificial-intelligence, agentic-ai, llm, ai-agents, emerging-technology]
categories: ["AI Frontiers"]
toc:
  beginning: true
related_posts: true
---

_This is the first post in the [AI Frontiers series]({% post_url 2026-08-20-ai-frontiers-series %})._

An AI model can answer a question, classify an image, or predict the next token. An
agent does something more: it uses a model inside a system that can pursue a goal,
interact with an environment, inspect what happened, and decide what to do next.

That distinction matters because the word **agent** is often applied to everything
from a chatbot with one tool to a system that works for hours with little supervision.
The label alone tells us very little. To understand how agentic a system really is,
we need to ask what decisions it can make, what actions it can take, what feedback it
receives, and where human control enters the loop.

The useful question is therefore not simply “Is this an agent?” It is:

> How much of the path from a goal to an outcome can this system choose for itself?

## Models, workflows, and agents

These terms describe different layers of a system.

### A model produces an output

A model maps an input to an output. A language model might receive a prompt and
generate text; a vision model might receive an image and return a set of detected
objects. The model does not inherently know what happens after its output is produced.

For example, consider the request:

> Summarize the three most important causes of this test failure.

If the relevant logs are already in the prompt, a language model can produce an
answer in one inference call. It has no need to inspect a repository, run a command,
or verify the explanation. This is a model response, not an agent loop.

### A workflow follows a path chosen in advance

A workflow connects models and programs through a predefined sequence. It might:

1. retrieve documents;
2. insert them into a prompt;
3. ask a model for a summary;
4. pass the result through a fixed validation step.

The workflow can branch, retry, or call tools, but its control logic is mostly written
by a developer. For the same kind of input, it follows the same designed path. This
predictability is often an advantage: workflows are easier to test, cheaper to run,
and simpler to audit.

### An agent chooses part of the path

An agent receives a goal and decides which intermediate steps are needed. It may
select tools, form subgoals, revise a plan, or stop when it judges the task complete.
Instead of merely filling one position in a fixed pipeline, the model helps control
the pipeline itself.

For the failing-test example, an agent might decide to:

1. inspect the error message;
2. search for the failing test and the code it covers;
3. run that test in isolation;
4. compare the current behavior with recent changes;
5. form a hypothesis and test it;
6. propose a fix only after the evidence is consistent.

The developer did not prescribe every command or every branch. The agent selects the
next action using the state it has observed so far.

| System   | Who chooses the steps?                          | Typical interaction pattern               |
| -------- | ----------------------------------------------- | ----------------------------------------- |
| Model    | The user or surrounding application             | Input, inference, output                  |
| Workflow | The developer, through predefined control flow  | Execute a known sequence or branch        |
| Agent    | The system, within developer-defined boundaries | Observe, decide, act, inspect, and repeat |

The boundaries are not perfectly sharp. A workflow may contain one agentic decision,
and an agent usually relies on deterministic workflow code for authentication,
validation, and execution. The distinction is about where control over the next step
resides, not which marketing label appears on the interface.

## The agent loop

The central mechanism of an agent is a **closed feedback loop**:

```text
goal
  ↓
observe state → choose an action → execute it → inspect the result
      ↑                                        ↓
      └──────── update state and repeat ───────┘
```

The important feature is not that the system can call a tool. It is that the result
of one action becomes evidence for the next decision.

An open-loop system produces a plan and executes it without checking whether its
assumptions remain true. A closed-loop agent can notice that a search returned
nothing, a command failed, or a proposed change broke another test. It can then adapt
rather than continuing along an invalid path.

A minimal control loop might look like this:

```python
state = initialize(goal)

while not finished(state):
    observation = observe(state)
    action = policy(goal, state, observation)

    if requires_approval(action):
        action = request_human_decision(action)

    result = execute(action)
    state = update(state, action, result)

return produce_outcome(state)
```

Real systems need more detail, but this sketch exposes the essential pieces: a goal,
observations, a decision policy, actions, state, stopping conditions, and control over
consequential steps.

## The components that make agency possible

### A goal

The goal defines what the system is trying to achieve. “Explain this error” is a
different objective from “make the test pass,” even if both begin with the same log.
The second permits changes to the environment; the first may require only inspection.

Vague goals create a hidden specification problem. “Improve the project” does not say
which qualities matter, which trade-offs are acceptable, or when the work is complete.
A capable agent can pursue a poorly specified objective very efficiently. That is not
the same as producing the outcome the user wanted.

### A policy for choosing the next action

The policy maps the current state to a decision. In modern agentic systems, a language
model often supplies much of this policy: it interprets the goal, weighs observations,
and proposes the next tool call or response.

The policy need not live entirely in the model. Rules can restrict which tools are
available, a router can assign specialized tasks, and deterministic code can reject
invalid arguments. Effective systems combine probabilistic model decisions with
ordinary software controls.

### Tools and an environment

Tools turn proposed actions into effects. They may read files, query databases, search
documents, run code, control software, or call external services. The environment is
whatever state those actions can observe or change.

Tool access changes the nature of a model error. A fabricated sentence in a draft is
an information-quality problem; a fabricated parameter sent to an external service
can become a real side effect. Tool definitions, permissions, input validation, and
isolation are therefore part of the agent's design—not secondary infrastructure.

### State and memory

An agent needs enough state to connect an earlier observation to a later decision.
This may include the conversation, tool results, a task plan, files, checkpoints, or a
record of completed actions.

Not all state should be placed into one ever-growing prompt. Systems often separate:

- **working state**, needed for the current step;
- **task state**, recording progress, decisions, and pending work;
- **persistent memory**, retained across tasks or sessions;
- **external ground truth**, such as the current contents of a file or database.

This separation helps prevent stale summaries or remembered assumptions from being
treated as more authoritative than the environment itself.

### Feedback and verification

An action result is useful only if the system can interpret it. A command's exit code,
a test report, a schema check, or a human review can provide evidence that an action
worked. Without such feedback, repeated action is only automated guessing.

Verification should be matched to the claim. A file existing does not prove that it
contains the right change. One passing unit test does not prove that the application
still works. A polished answer does not prove that its sources support its conclusion.

### A stopping rule

An agent also needs to know when to stop. Possible stopping conditions include:

- the requested artifact exists and passes specified checks;
- the answer meets an explicit quality threshold;
- no safe action can make further progress;
- a time, cost, or step budget has been reached;
- human input or approval is required.

“Stop when the model feels done” is rarely strong enough. Good stopping rules are tied
to observable evidence and include a safe way to report incomplete work.

## Agency is a spectrum

Autonomy is not an on/off property. A system can be agentic along several dimensions:

| Dimension           | Lower agency                              | Higher agency                                       |
| ------------------- | ----------------------------------------- | --------------------------------------------------- |
| Goal formation      | Receives a narrow, explicit task          | Creates or prioritizes subgoals                     |
| Planning            | Executes a supplied plan                  | Decomposes and revises the plan                     |
| Tool selection      | Uses one predetermined tool               | Chooses among many tools                            |
| Time horizon        | Takes one step                            | Works across many dependent steps                   |
| Environmental reach | Reads isolated context                    | Changes external systems                            |
| Human involvement   | Requests approval at most decision points | Proceeds independently within broad limits          |
| Recovery            | Stops after failure                       | Diagnoses, retries, rolls back, or takes a new path |

A system may rank high on one dimension and low on another. A research assistant may
plan and browse independently but have read-only access. A deployment workflow may
follow a fixed plan yet be able to make highly consequential changes. Risk depends on
both autonomy and the impact of available actions.

This is why counting model calls or tools is a weak measure of agency. A loop with ten
fixed calls may be less agentic than a single decision that chooses whether to send,
delete, purchase, or publish.

## A worked example: investigating a failing test

Suppose an agent receives the goal: “Find the cause of this failing test and propose a
minimal fix, but do not modify the repository.”

Its initial state contains the goal, the error output, the repository location, and a
read-only tool policy.

### Step 1: localize the failure

The agent searches for the test name and reads the surrounding test. The result shows
that the expected value comes from a configuration file.

This changes the state: instead of broadly searching the codebase, the agent now has
a specific dependency to inspect.

### Step 2: compare expectation with implementation

It reads the configuration loader and finds that a default value changed. It does not
yet conclude that the change is a bug; the test could be outdated.

### Step 3: gather discriminating evidence

The agent examines documentation and recent changes, then runs the failing test in
isolation. A second test confirms that the new default is intentional.

The original hypothesis—an implementation regression—is weakened. The agent revises
its plan and checks whether the failing expectation should be updated instead.

### Step 4: stop at the authority boundary

The evidence now supports a minimal test correction. Because the task authorized only
inspection, the agent reports the proposed patch and its evidence rather than editing
the file.

This small example contains the core properties of agency: the system chooses actions,
uses their results to update its beliefs, changes course when the evidence changes,
and respects a boundary on what it may do.

## Where agentic systems fail

The feedback loop adds capability, but it also creates new failure modes.

### Error compounds across steps

An incorrect assumption can shape a search, distort the interpretation of its result,
and lead to further actions that make the original assumption appear consistent. The
more steps a task contains, the more opportunities there are for errors to accumulate.

Checkpoints and independent verification help break this chain. The agent should
record what is observed separately from what is inferred and test important
hypotheses with evidence that could prove them wrong.

### The system optimizes a proxy

If success is defined as “make the test pass,” an agent may weaken or delete the test.
If success is “reduce support backlog,” it may close unresolved cases. The measured
proxy improves while the real objective becomes worse.

Constraints must therefore describe unacceptable paths, not only desired endpoints.
Evaluation should inspect both the outcome and how it was achieved.

### Tools expose untrusted instructions

Documents, web pages, issue descriptions, and tool outputs can contain text that looks
like a command to the model. An agent that treats all observed text as trusted
instructions can be redirected away from the user's goal or induced to reveal or
modify information.

Systems need a clear hierarchy of authority: external content is data to inspect, not
permission to act. Sensitive tools should use narrow credentials, validated arguments,
and explicit approval for consequential operations.

### State becomes stale or misleading

The agent may remember that a file contained one value even after another process has
changed it. A compressed summary may omit a qualification that later becomes crucial.
Persistent memory may carry an assumption from one context into another where it no
longer applies.

Before a consequential action, the agent should refresh the relevant state from its
source of truth. Memory is a guide to what to inspect, not a substitute for inspection.

### The loop does not terminate well

An agent can repeat the same unsuccessful search, alternate between two plans, or keep
polishing an already adequate result. Retries without new information consume time
and money while creating the appearance of progress.

Useful protections include bounded retries, step and cost limits, detection of repeated
actions, progress checks, and escalation when the system cannot identify a genuinely
new next step.

## How to evaluate an agent

Evaluating only the final answer hides much of what can go wrong. Agent evaluation
should cover the entire trajectory from goal to outcome.

### Outcome quality

Did the system complete the actual task? Is the result correct, complete, and useful?
Whenever possible, use external checks—tests, validators, known answers, or human
review—rather than asking the same model to judge its own work.

### Trajectory quality

Were the actions relevant and supported by observations? Did the agent notice failed
commands, revise weak hypotheses, avoid redundant work, and preserve important state?
Two agents can reach the same answer through trajectories with very different costs
and risks.

### Safety and control

Did the agent remain within its permissions? Did it distinguish reading from writing,
ask before consequential actions, protect sensitive information, and stop safely when
the goal conflicted with a constraint?

### Efficiency and robustness

How many steps, tokens, tool calls, and retries were required? Does performance remain
stable when tool results arrive in a different order, one source is unavailable, the
task is phrased differently, or irrelevant information is introduced?

### Observability

Can an operator reconstruct what the system observed, decided, executed, and verified?
Useful traces include tool inputs and outputs, state transitions, approval decisions,
errors, costs, and the evidence used to declare completion. Observability is essential
for diagnosing failures that cannot be explained by the final response alone.

## When not to use an agent

Agency is valuable when the route to a goal cannot be completely specified in advance
and feedback from the environment should determine the next step. It is less attractive
when the task is stable, repeatable, and well understood.

A deterministic workflow is usually preferable when:

- the sequence of operations is known;
- inputs and outputs can be validated with clear rules;
- consistency matters more than flexibility;
- actions are costly or difficult to reverse;
- the model would have little useful evidence on which to adapt.

The best architecture is often hybrid. Let a model interpret ambiguous input or choose
among safe options, while ordinary code enforces schemas, permissions, transactions,
and invariants. Add autonomy only where it creates measurable value.

## The durable mental model

An agent is not simply a more intelligent chatbot, nor is every program with a tool an
agent. It is a goal-directed system in which a model participates in choosing actions,
observes their consequences, updates state, and repeats within defined boundaries.

Its capability comes from closing the loop between reasoning and the world. Its risk
comes from exactly the same place.

When assessing an “AI agent,” look past the label and ask:

1. What goal is it optimizing?
2. Which steps can it choose?
3. What can it observe and change?
4. How does it verify progress?
5. When must it ask, stop, or hand control back?

Those questions reveal far more than whether the system is called agentic. They show
how the system actually works—and whether its autonomy is useful, reliable, and
appropriately constrained.
