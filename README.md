# Gemma 4 12B teacher trace SFT

This package trains one PEFT adapter from verified passing teacher traces using
the single-agent SWEGemma harness and its six Python tools. Hard distillation
uses recorded teacher responses as supervised targets; no teacher runs during
training. Tool counts and task duration are retained for analysis, not rewards.

## Install and configure

Use Python 3.12 in a dedicated environment. Install the appropriate ROCm or CUDA
PyTorch build first, then install from the repository root:

```bash
python -m pip install -e 'training[train,test]'
swe-train config init --profile rocm32 --output train.yaml
```

Profiles are `rocm32` (one device, CPU offload), `nvidia48x2` (two devices,
GPU-only), and `l4x4` (four devices, GPU-only). These are starting configurations,
not guarantees that a given sequence length fits. GPU-only failures do not
silently enable offload or shorten sequences. TPU execution is not implemented.
Model weights are dequantized to BF16 for differentiable training; the QAT
checkpoint does not imply 4-bit training memory usage. No bitsandbytes optimizer
or model loader is used.

The annotated template is packaged at
`src/swegemma_training/templates/train.example.yaml`. Edit the generated YAML:
set `data.traces`, `data.outcomes`, and `harness.submission_dir`. Every relative
path is relative to that YAML file. The default harness points to the Nemotron
project's current `sample_submission`, not the older three-agent harness.
Install its SWEGemma tool-schema provider and ADK wheels separately for
preparation; they are private packages and are not downloaded by this package.
The serving runtime, including vLLM, is not a training dependency.

```bash
swe-train config validate --config train.yaml
swe-train config show --config train.yaml
swe-train doctor --config train.yaml
swe-train prepare --config train.yaml
swe-train train --config train.yaml
```

`doctor` reports installed versions, accessible GPUs, memory and runtime settings
without loading weights. On GPU-only profiles, `doctor --smoke` additionally
loads the actual base and runs a four-token forward/backward. This is not a
full-context memory benchmark. Offload training uses FSDP2 hooks and must be
exercised through `train` rather than inference device placement.
The initial loader conservatively dequantizes one complete text base on CPU per
worker before sharding. Allow roughly 24 GB of host weight storage per worker,
plus loading and runtime overhead; `doctor` reports this estimate. Four L4s do
not imply that the host has enough RAM for this loading path.

Explicit overrides are available for experiments:

```bash
swe-train config show --config train.yaml --set memory.offload=none
swe-train train --config train.yaml --set optimizer.learning_rate=0.00005
```

Defaults are overridden by YAML, then by repeated `--set dotted.key=YAML_VALUE`
options. Unknown keys and invalid values fail validation. Uniform-rank trials
set all seven entries of `adapter.rank_pattern`; the default projection-specific
pattern is q/o 12, k/v 8, gate 24, up/down 16. Alpha defaults to twice each rank,
with explicit overrides available. The actual trainable count and matched
module names are saved; some Gemma layers share K/V projections.

`optimizer.warmup_ratio` remains a fraction in this YAML. The runner passes it
to the current TRL/Transformers `warmup_steps` argument, which accepts fractions
below one. Trainer arguments are validated before loading the base weights.

## Teacher data and labels

Point `data.traces` at ATIF JSON files such as `results/traces/trace_fastapi_123.json`.
`data.outcomes` accepts `task_results.jsonl`, a directory containing that file,
or a Kaggle `.log` with evaluation summaries containing both `resolved=True`
and `exit_code=0`. Decoded Kaggle JSON event logs and plain text logs are accepted.
Choose one collection attempt per task: ambiguous duplicate outcomes/traces fail.

ATIF must retain the initial user prompt and complete, correctly paired tool
calls and results. A recorded system instruction must match the configured
harness. Mixed-agent traces, unknown tools, orphan responses and ambiguous
thinking-only events are excluded. Reasoning is preserved when recorded, never
invented. Templates that silently discard recorded reasoning are rejected.
Traces containing context compaction are also excluded when the original model
request context cannot be recovered from ATIF. Exact teacher request capture is
needed to support those traces faithfully.

By default (`data.sequence_mode: turn`), preparation renders the student chat
template and labels only the current assistant response, including its reasoning, tool calls or final answer. Prompt,
previous turns and tool results are context with labels `-100`. Task IDs are
split before examples are constructed. At least two accepted passing tasks are
required. Completion/thinking budgets come from `configs/sampling.yaml` through
the harness include loader. The initial teacher contract is context 32768,
completion 12288 and thinking 5120; teacher and student use their own tokenizers.

Overlength targets are quarantined, never truncated. An invalid example excludes
that task's trajectory. The manifest records exclusions, per-split counts,
supervised tokens, source hashes, tokenizer identity, the resolved harness and
task memberships. A prepared directory is immutable: choose a new
`data.prepared` directory to regenerate it. Historical teacher budget enforcement
cannot be proved from an ATIF file alone and is labeled accordingly.

## Trajectory regimes and SoFT

The cleaned conversation export in `examples/trajectory-sft-harness.yaml` uses
`data.format: harness_jsonl`, explicit `data.train_file` and
`data.validation_file`, and `data.sequence_mode: trajectory`. It preserves the
provided split and recorded system prompts. Do not concatenate its recovery or
partial files into positive SFT. Source-verified successes are accepted using
the successful-demonstration flag, validated environment, and passing
fail-to-pass/pass-to-pass results. Original exit codes remain in provenance;
this source policy does not change SWEGemma evaluation's success criterion.

The importer checks the recorded text against the pinned student's native
template, independently derives assistant ownership from completed prefixes,
and cross-checks the source target spans. It handles adjacent narration and
tool-call chunks without inserting new assistant headers. Tool observations,
system/user content, and the injected tool-response prefix are masked. Source
message order is preserved, including interleaved calls; these are annotated
instead of moved across observations. Unsupported templates or token boundaries
are reported as exclusions. This format currently supports full trajectories,
with either SFT or SoFT; ATIF retains all existing sequence modes.

`rendered.enable_thinking` records student chat formatting, rather than the
teacher's internal reasoning setting. Missing reasoning is never invented.
Stored patches, outcomes and other metadata are provenance, not model targets.
Input files, source/effective prompts and renderer settings are fingerprinted;
changing a source or prompt requires a new prepared directory.

For independent training experiments, optionally set `data.prompt_override` to
a Markdown file. The original prompt remains in provenance and only the training
system context changes. This is an explicit context adaptation of fixed teacher
actions, rather than newly collected demonstrations under that prompt. Use
separate YAML files, prepared directories and run directories for each run;
there is no special A/B training mode. The evaluation harness remains configured
independently in `harness.submission_dir`, so both adapters can be evaluated on
the same tasks, prompt, sampling and budgets. For external datasets, explicitly
set `evaluation.task_ids` to held-out IDs present in the SWEGemma `tasks.jsonl`;
their supervised-validation task IDs may belong to a different task collection.

Generate a complete YAML for the chosen sequence mode/objective and hardware:

```bash
swe-train config init --regime trajectory-sft --profile nvidia48x2 --output trajectory.yaml
swe-train config init --regime windowed-sft --profile rocm32 --output windows.yaml
swe-train config init --regime windowed-soft --profile rocm32 --output soft.yaml
```

Complete examples also live in `examples/`. Existing files retain their `turn`
SFT behavior; these commands never overwrite an existing configuration. Paths
are relative to the generated YAML. Edit trace, outcome, and harness locations
before preparation. To use SoFT with a complete trajectory, set
`data.sequence_mode: trajectory` in the SoFT recipe.

`trajectory` makes each accepted full conversation one training example. Every
assistant turn is supervised, including recorded reasoning, tool calls, and final
output; instructions, task text, tool observations and padding are masked. The
context limit applies to the entire rendered conversation, while completion and
thinking limits still apply per response. Overlength trajectories are excluded.

`windowed` keeps the original system instruction, tool schemas and task, plus a
contiguous suffix of recent complete exchanges. `data.window.max_tokens: 8192`
includes all of those tokens; `history_tokens: 2048` is the additional budget
for recent history beyond the pinned task. New exchanges are greedily grouped
into windows. Repeated history is masked and each eligible assistant target is
supervised once. An assistant response and its following tool observations stay
whole. Oversized exchanges are excluded with explicit coverage records. A pinned
prompt that already fills the window is rejected. Windows may omit old exchanges;
that omission is recorded and is a deliberate change to the training context.
Smaller windows are not a guarantee of fitting 32 GB.

The renderer verifies token spans against the student chat template. Some Gemma
templates move assistant narration after subsequent tool results, or remove
reasoning before a later user message. Such targets would leak future context
or lose recorded supervision. Complete trajectories containing these rewrites
are rejected; windowed preparation reports and skips affected exchanges. Review
`coverage` and `excluded` before training. No reasoning or observations are
silently truncated or invented, and no hidden training-only chat template is
substituted. All splits are made by task before constructing windows.

SoFT follows [Soft Targets for Generalizable LLM Fine-Tuning, v2](https://arxiv.org/html/2609.32493v2),
using its top-K/gold/tail approximation. It requires recorded teacher text and a
frozen initial student, including any initial warm-start adapter. It does not
require the Nemotron teacher or teacher logits during training.

```bash
swe-train prepare --config soft.yaml
swe-train cache-reference --config soft.yaml
swe-train train --config soft.yaml
```

Reference caching uses the configured GPU topology/offload setup, with the
reference in evaluation mode and gradients disabled. It records the top-32
probabilities, demonstrated-token probability, and residual tail mass at each
supervised position using that example's exact context. Across the unique target
tokens of each source trajectory, bisection finds a probability floor that meets
`objective.soft.rstar`. Windows therefore share a trajectory-level calibration,
while their reference predictions reflect their own reduced contexts. The cache
also stores the resulting soft targets; training loads those compact targets
and needs no reference model. Student vocabulary projections are processed in
`memory.loss_chunk_tokens` chunks rather than constructing sequence-wide logits.

`rstar: 0.5` is an experimental starting point to tune on held-out tasks; `0`
preserves the reference distribution and `1` recovers hard targets. `top_k: 32`
is the paper's practical approximation. The tail is an aggregate bucket, not a
uniform distribution. The exact full-vocabulary objective is used as a small
numerical test oracle, rather than offered as a large-model runtime mode.

Reference caches are immutable and checksummed. Changing the dataset, windows,
tokenizer/template, pinned model revision, initial adapter, package versions,
K or budget requires a new `objective.soft.cache_dir`. Training validates all
cache shards before loading the base. Failed cache jobs do not publish a final
cache directory; their torchrun logs remain alongside the cache for diagnosis.
Cached probabilities, calibration statistics, and targets remain local.

Prepared v2 manifests report trajectories, examples/windows, supervised tokens,
excluded targets, window boundaries and omitted history. Epochs count examples
in the selected mode: six full trajectories give six examples, whereas windows
produce a different number. `microbatch_size` is still examples per device.
The regimes are teacher-forced imitation of successful teacher trajectories;
recovery from student-generated errors requires downstream rollout training.

GPU memory use, multi-GPU throughput, and held-out SWEGemma gains must be measured
on the deployment hardware. Compare base/adapter harness results with the same
tasks, sampling, context limits, and seeds. Report resolution rate, invalid tool
calls, runtime and tool counts; SFT loss alone does not establish agent quality.
The comparison report counts unknown tool names and malformed/non-object JSON
arguments as invalid calls. It reports this metric as unavailable if any task
trace is missing; schema validation and runtime tool failures remain in traces.
No automatic context shortening or objective changes occur after an OOM.

## Checkpoints and progress

Each run writes a resolved YAML, harness snapshot, dataset manifest, environment
report, module/rank report, `metrics.jsonl`, resumable checkpoints, and adapter
releases. tqdm displays preparation progress and the Trainer's rank-zero training
progress. Set `progress.enabled: false` for noninteractive logs. JSONL metrics
remain enabled; optional TensorBoard uses `training.tensorboard: true` and the
`training[tensorboard]` extra. MLflow is not required.

The main tqdm counter advances once per optimizer step, after gradient
accumulation completes. Microbatch completion messages and a heartbeat every
`progress.heartbeat_seconds` (default 30) report the current forward/loss or
backward phase. `RUN/progress.json` stores the same status. A heartbeat shows
where execution is waiting; it does not prove that a GPU kernel is progressing.
These messages do not change the accumulation factor or optimization settings.

Initialization also reports phases before the first training callback, including
token loading, base decompression, Trainer construction, and distributed model
preparation. `RUN/startup.json` contains the latest snapshot; `startup.jsonl`
retains phase changes and heartbeats. Startup and training snapshots include
process RSS and available/shared system RAM. The startup monitor hands off when
the training callbacks begin. It does not query or synchronize the GPU driver.
Prepared token columns are converted row by row into a disk-backed Arrow cache
under `data.prepared/arrow-cache`, before model loading. This avoids retaining all
JSONL provenance and token lists in Python RAM alongside the decompressed model.
Example order, SoFT cache IDs, tokens and supervised masks are preserved.

FSDP2 model saving filters trainable adapter parameters before gathering shards;
it does not gather the frozen base to create a PEFT export. Resumable checkpoints
contain adapter weights in both PEFT and Accelerate's native formats, adapter
optimizer state, scheduler, RNG and Trainer state. The pinned base is reconstructed
before resuming. Shared frozen embeddings are excluded as well. Heartbeats mark
`saving_model`, `saving_optimizer_and_scheduler`, and `checkpoint_complete`, so
time spent saving is not reported as an ongoing training microbatch.
Activation checkpoint wrapper names are normalized before gathering. FSDP2
exports write this recipe's projection LoRAs directly in PEFT format, avoiding
PEFT's module-name filtering while wrappers are present; empty or incomplete
adapter tensors are rejected.

The loss computes vocabulary logits in configurable token chunks and preserves
Gemma's logit softcap. Legacy turn SFT averages microbatch token means. Trajectory
and windowed modes, and SoFT in any sequence mode, divide the summed assistant
token loss by the supervised token count across the entire optimizer update,
including all accumulation microbatches, ranks, and partial final updates. There is no tool-count
or trajectory-length bonus. Activation checkpointing is configurable.

All sequence modes and both objectives check for NaN/Inf in raw training and
evaluation loss, gradients before clipping/updating, the clipping norm, and
adapter weights and optimizer state after an update. An invalid update stops
the run before a new checkpoint or release is accepted. Previous healthy
checkpoints remain available. Distributed checks inspect local shards and share
a failure flag without gathering the frozen base. Non-finite metrics are also
rejected; Transformers' NaN/Inf log filtering is disabled so invalid losses
cannot appear as reassuring zero losses.

SoFT additionally checks reference logits/probabilities and target construction.
Cached targets must be finite, nonnegative, and normalized, even when their file
checksums match. These checks run before loading training weights. Resume checks
saved adapter tensors and restored optimizer state; warm starts, exports and
publishing reject non-finite adapters, including files created by older versions
of this harness. YAML float settings also reject NaN and infinity.

`failure.json` records caught failures with the optimizer step, completed
microbatches and current sequence length. Numerical guards stop corruption from
propagating; they do not repair an unstable GPU kernel or recover already
corrupted weights. Do not resume or warm-start from a checkpoint containing
NaN/Inf. Native GPU faults can still bypass Python checks entirely.

```bash
swe-train train --config train.yaml --resume latest
swe-train export --config train.yaml
```

Checkpoints save optimizer, scheduler, RNG, data position and Trainer state at the
configured step interval and last training step. SIGINT/SIGTERM requests a save
at an optimizer-step boundary. A `complete.json` marker identifies checkpoints
whose save finished. Resume requires unchanged gradient-affecting configuration,
dataset, package versions and process count. For a hardware/topology change,
set `adapter.warm_start` to an exported release and start a new run; this explicitly
resets optimizer state. Resume must not also specify a warm start.

The template saves every optimizer step because a single full-context update can
take several minutes. Existing YAML files retain their explicit interval: set
`training.save_steps: 1` for frequent checkpoints. Fatal GPU/process aborts cannot
save a checkpoint during the crash. Updates since the last complete checkpoint
are lost.

Worker stdout/stderr is mirrored to `RUN/worker-logs/` and the terminal. Python
fatal-error stack capture and unbuffered output are enabled by default, while
explicit environment overrides are respected. If torchrun fails, the parent
writes `RUN/launcher-failure.json` with the actual list of complete checkpoints;
this record can survive a native worker abort that bypasses Python handlers.
For AMDGPU page faults, correlate worker PID/time with the kernel journal before
attributing the problem to memory capacity or a particular kernel. Synchronous
HIP execution is a diagnostic option, with a performance cost, rather than an
automatic training setting.

A frozen 12B BF16 base still needs roughly 24 GB before offload setup. FSDP can
temporarily retain the original CPU state while creating the offloaded state;
the initial weight estimate from `doctor` is not a peak-RAM estimate. Pinned host
allocations cannot be swapped out. `memory.cpu_offload_pin_memory: false` uses
pageable offload storage through PyTorch's FSDP2 policy, potentially trading
transfer speed for less locked-RAM pressure. This does not eliminate temporary
weight copies or guarantee that a run fits. New templates set it to false;
existing configurations that omit it retain the previous true default. The same
policy applies to SoFT reference caching. Changing it requires a new run rather
than resuming a checkpoint with a different configuration fingerprint.

If the desktop freezes and the kernel reports `Out of memory: Killed process`,
that is host-memory exhaustion. Check `journalctl -k -b --grep 'Out of memory'`
and the startup snapshots. A SIGKILL can prevent both worker and launcher failure
records from being written. The OS may kill unrelated desktop applications too.
On a Linux desktop with systemd and about 64 GB RAM, bound the next attempt in
a transient user scope (adjust limits for your machine):

```bash
systemd-run --user --scope \
  -p MemoryMax=40G -p MemorySwapMax=4G \
  "$VIRTUAL_ENV/bin/swe-train" train --config "$TRAIN_CONFIG"
```

This scope includes the launcher and its workers. It can terminate training when
the limit is reached while reserving room for the desktop; it does not reduce
the model's memory requirement or prevent an independent GPU/driver fault. If
the bounded run exceeds its limit, use its startup reports to decide whether
more host RAM or a different offload strategy is needed. Avoid an unbounded
repeat of the same desktop-wide OOM.
Avoid a `MemoryHigh` soft limit below the initialization working set: it can
cause prolonged reclaim throttling before `MemoryMax` is reached. For an existing
scope, `memory.events` records these `high` events and `memory.pressure` records
time stalled on reclaim. Removing the soft limit does not increase the hard cap
or guarantee that initialization fits.

Portable releases live under `RUN/releases/step-…` and contain adapter weights,
tokenizer files, provenance and file checksums. `RUN/latest-release.json` points
to the latest release; resuming does not overwrite an older release. Base weights,
optimizer state and raw teacher traces are excluded. Keep full local checkpoints
when exact resumption is needed. The configured checkpoint retention limit applies
to optimizer checkpoints; exported releases are retained.
For a unified base checkpoint, export translates text-training adapter keys to
the original `model.language_model` namespace. Warm starts reverse that mapping;
training checkpoints retain their native text-only layout. Local base-model paths
must remain immutable; remote sources are pinned to their resolved Hub commit.

## Harness evaluation and publishing

Run evaluation in a separate vLLM-capable environment with the private harness
packages installed. Install this package there without its `train` extra; the
training and serving Transformers version constraints may differ. Set
`harness.data_root` to the task dataset, with `tasks.jsonl`, `snapshots`, `wheels`,
`graphs` and `embeddings`, and select the parser names supported by that vLLM build.

```bash
swe-train evaluate --config train.yaml --adapter /path/to/exported/release
```

Evaluation starts and stops a local vLLM server, loads the exported adapter, and
compares the base and adapter through the same prompts, tools and sampling YAML.
By default it uses validation task IDs. Results include test outcomes, tool calls,
durations, traces and errors. Native packed-weight inference, text-backbone adapter
name mapping, variable-rank LoRA support and exact reasoning-budget transport must
be validated against the serving build before treating a release as deployable.

Everything stays local by default. Enable either publisher in YAML with its
`repo_id`, install `training[publish]`, and authenticate via `hf auth login` or
the Kaggle CLI. Credentials are not written into YAML or release artifacts.
Enabled targets are attempted after a successful uninterrupted run. HF creates
new repositories private; Kaggle creates new datasets private. To update an
existing Kaggle dataset, set its `mode: version`.

```bash
swe-train publish --config train.yaml --target hf
swe-train publish --config train.yaml --target kaggle
```

An upload failure leaves the local release intact and records a retryable status
beside it. Only the release allowlist is uploaded. Uploading intermediate optimizer
checkpoints is not automatic. The optional notebook is a thin CLI entrypoint;
it does not maintain a second training implementation.

## Validation and extension

```bash
python -m pytest training/tests -q
```

CPU tests exercise real tiny Gemma models, LoRA gradients, frozen base weights,
chunked-loss equivalence, checkpoint resume, tool/trace validation, task splits,
configuration and publishing safeguards. Full 12B loading, ROCm/CUDA offload,
multi-GPU execution, full-budget VRAM use and vLLM deployment require hardware
validation. No production or GPU-fit claim follows from the CPU tests.
Numerical tests run all six sequence-mode/objective combinations with real tiny
Gemma/PEFT/TRL training. They inject invalid losses, gradients, clipping norms,
optimizer updates/state, and reference caches; they also cover saturated SoFT
probabilities, valid checkpoint retention, and invalid optimizer resume. A
two-rank CPU/Gloo shard test requires permission to open local sockets and is
skipped in restricted environments. This does not replace GPU/FSDP validation.

The package separates trace data, runtime contracts, model/adapter loading,
run artifacts and the SFT runner. Future GRPO/RLVR stages can warm-start from the
same releases and reuse those components. Soft distillation and TPU backends are
not part of this first implementation.
