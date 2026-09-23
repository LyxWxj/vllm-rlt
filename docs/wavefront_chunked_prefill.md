# Wavefront Chunked Prefill Proposal

**Status: proposal.** This document describes a possible finer-grained prefill scheduler. It does not describe behavior implemented by the current runner.

## Goal

Allow ready prompt chunks at different loop depths to share one recurrent-core batch. Smaller PREFILL invocations also create more scheduler boundaries, where the engine can admit new prefill work or choose a separate decode batch instead of waiting for every chunk in the current prefill batch to finish all loop depths.

The optimization must preserve the current causal attention and KV semantics. It must not change generated tokens or the full-depth computation used to produce the first output token.

## Current Execution

The scheduler selects a prompt range per request. Its length is bounded by the remaining token budget, `prefill_chunk_size`, and the unprocessed prompt length. It packs selected ranges into one PREFILL batch.

For `LAST_EXITED`, the runner embeds the selected tokens and executes the complete recurrent depth over that batch before returning. The engine then advances each request's prefill frontier and requeues any request with prompt tokens remaining. The final chunk enters CODA only after the full prompt is complete. See `Scheduler._make_scheduled_item()`, `ModelRunner._prefill_tokens()`, and `LLMEngine._update()`.

For `SHARED`, the runner completes all loops for one prompt position before advancing that request to its next position. This is required by the layout's semantics: later positions use the earlier positions' final KV. Different requests can still be batched at each position wave.

Consequently, current chunking bounds the amount of prompt work selected per scheduler batch; it does not expose each `(chunk, loop depth)` as an independently schedulable task. The current prefill runner supplies one common depth to all rows on each recurrent call.

## Proposed Work Unit

Represent prefill work as a task keyed by:

```text
(request_id, token_start, token_count, loop_depth)
```

Each task carries its input hidden states. At depth zero these are token embeddings; at later depths they are the previous depth's output for the same token range. A task becomes runnable only when both conditions hold:

1. Its input hidden states are available.
2. Causal KV for earlier prompt positions is ready in the cache plane used by this task's depth.

Attention handles causal ordering among positions inside the task's chunk. The scheduler packs ready prefill tasks under the existing sequence and token limits. A recurrent invocation may therefore contain prefill rows from different requests, chunks, and loop depths. The first implementation does not combine prefill and decode rows in the same invocation; decode remains a separate batch selected at a scheduler boundary.

For example, after `B chunk 0 @ depth 0` completes, a subsequent batch may contain `B chunk 0 @ depth 1` and `B chunk 1 @ depth 0`. The first task uses the saved depth-zero hidden state. The second uses embeddings and reads chunk 0's depth-zero KV as its causal prefix. This is valid only while chunk 0 still has work at depth one; a chunk that already completed every depth must not be scheduled again.

The final prompt position may enter CODA only after every preceding prompt position has completed the required full prefill depth. Chunk completion and request completion therefore remain distinct states.

## Cache Layout Constraints

### LAST_EXITED

This is the first target for same-request wavefront scheduling. Each recurrence depth has a separate KV plane, so a later chunk at depth `d` can read earlier chunks' KV at that same depth. The scheduler must enforce that dependency before launching the task.

`KVCacheManager._prepare_batch()` already accepts per-row depth metadata, and the model's recurrent core uses those depths to select KV namespaces. The current prefill path does not use mixed depths: it explicitly builds a depth vector filled with the same value for each call.

### SHARED

Do not apply same-request cross-chunk wavefront scheduling under the current semantics. A later position needs earlier positions' final KV, and intermediate prefill loops do not persist those KV entries. Keep the existing per-position full-depth order. Mixed-depth batches across independent requests may be considered separately because their KV histories do not depend on one another.

## Technical Route

1. **Instrument the baseline.** Record per-core-call effective token rows, loop depth, batch composition, arrival-to-schedule delay, TTFT, decode inter-token latency, and peak memory. Include mixed prompt lengths and decode traffic.
2. **Add explicit prefill task state.** Track each active chunk's depth and hidden-state ownership separately from the request's contiguous completed-token frontier. Maintain readiness dependencies for previous-depth hidden states and same-depth prefix KV.
3. **Add mixed-depth prefill execution.** Build per-row request IDs, positions, and depths; gather the correct hidden input for each task; run one shared recurrent core; write KV to the matching depth plane; retain the output only until that chunk advances or completes. Preserve packed causal attention for tokens within each chunk.
4. **Make scheduling dependency-aware.** Select only ready tasks, form batches within `max_num_seqs` and `max_num_batched_tokens`, and let the scheduler choose a separate decode batch between PREFILL invocations. Revisit prefill fairness accounting so the allowance reflects token rows and loop work, not only the number of scheduler calls.
5. **Roll out narrowly.** Start with `LAST_EXITED`, synchronous execution, and deterministic fixed-depth replay. Keep the existing path for `SHARED`, asynchronous execution, preemption, prefix caching, and disaggregated serving until their task-state and event-lifetime contracts are covered.
6. **Consider cross-stage fusion separately.** Combining prefill and decode rows in one recurrent invocation requires a mixed-work batch contract, plus separate handling for prefill hidden-state progression and decode exit signals. It is not required to validate the wavefront scheduler and should be evaluated only if separate batches leave meaningful launch or occupancy overhead.
7. **Optimize after correctness.** Add backend-specific mixed-depth metadata and buffer reuse. Consider asynchronous scheduling or CUDA Graph support only after eager execution matches the reference.

### State and Memory

The runner currently retains a final hidden state per request between stages. Wavefront prefill needs intermediate hidden states for every in-flight chunk that has more loop depths to execute. The initial implementation should measure and bound this storage explicitly. A ready-task window can cap the number of partially advanced chunks; recomputing earlier depths is an alternative tradeoff, but adds model work and should be measured separately.

KV reservation remains a separate concern. Smaller execution tasks do not by themselves reduce KV capacity: with the default non-incremental allocation mode, admission still reserves the request's logical capacity up front.

## Expected Benefits

| Workload | Expected effect | Main condition |
| --- | --- | --- |
| Long prompts with short requests arriving during prefill | Reduce the wait for a short request to join a batch and potentially lower its TTFT | There must be ready work and unused batch capacity at loop boundaries |
| Low or uneven concurrency | Increase average active token rows per recurrent-core call | Mixed-depth ready tasks must fill otherwise idle batch capacity |
| Prefill concurrent with decode | Reduce decode queueing and improve inter-token latency tails | The scheduler must select a separate recurrent decode batch between finer-grained PREFILL invocations |
| Uniform long prompts with a saturated batch | Little expected benefit | Current batches already keep the recurrent core occupied |

The model performs the same per-token loop computation. The expected gain comes from better prefill batch occupancy and shorter scheduling delays, not from fewer FLOPs. No numeric speedup is claimed before measurement. Additional scheduler work, launches, and retained hidden states may outweigh the benefit for small batches or short prompts.

## Validation Plan

Compare the current path and the wavefront path with the same model, prompt set, sampling configuration, cache layout, and hardware. Use deterministic fixed-depth execution first, then repeat with the normal exit policy.

Measure:

- Prompt tokens per second and aggregate requests per second.
- TTFT and decode inter-token latency at p50, p95, and p99.
- Effective token rows per core call, separated by depth.
- Scheduler delay from task-ready time to launch.
- Peak KV and intermediate-hidden memory, plus recurrent-core launch count.

Cover a single long prompt, short prompts arriving behind a long prompt, bimodal prompt lengths, and continuous prefill arrivals while decode requests are active. Verify output token IDs and exit depths against the existing full-depth prefill path. Also compare KV contents or deterministic logits at the prompt boundary so matching sampled tokens cannot hide a cache mismatch.

Treat the change as beneficial only if it preserves correctness and improves the targeted service metric on the intended workloads without unacceptable single-request TTFT or memory regressions. If effective batch occupancy rises but throughput and latency do not improve, the added scheduling state is not justified.

## Benchmark Command

The controlled benchmark compares the same branch with `wavefront_prefill` disabled and enabled. It reports total time, output throughput, prefill rows per batch, recurrent graph captures/replays, and can export a Chrome trace:

```bash
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.wavefront_prefill \
  --device cuda --attention-backend triton --async-scheduling --cuda-graphs \
  --prompt-length 96 --prefill-chunk-size 2 --max-num-batched-tokens 4 \
  --profile /tmp/wavefront-prefill.json.gz --output /tmp/wavefront-prefill.json
```

The workload targets a low-concurrency long prompt where one chunk has only a few token rows. The benchmark checks output equality before reporting speedup; a speedup below `1.0` is a valid result and means this workload did not benefit from the finer scheduling granularity.
