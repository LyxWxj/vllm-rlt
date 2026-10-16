"""CPU scheduling at stage/loop boundaries, independent of model execution."""

from collections import deque
from dataclasses import dataclass, replace
from enum import Enum, auto

from vllm_rlt.config import SchedulerConfig
from vllm_rlt.core.scheduling_policy import create_scheduling_policy
from vllm_rlt.request import FinishReason, Request, Stage

# Items of these stages cover a contiguous span of prompt tokens, so their KV
# frontier is the span end rather than the request's decode position.
_SPAN_STAGES = (
    Stage.PREFILL,
    Stage.PREFILLING_PRELUDE,
    Stage.PREFILLING_RECURRENT,
    Stage.SPECULATIVE,
)
# Stages that must reserve KV capacity for that frontier before selection.
_CAPACITY_STAGES = _SPAN_STAGES + (Stage.PRELUDE, Stage.RECURRENT)


@dataclass(frozen=True)
class ScheduledRequest:
    request: Request
    # PREFILL and SPECULATIVE use a contiguous span; ordinary decode uses one row.
    token_start: int = 0
    token_count: int = 1
    # Recurrence position/depth are snapshotted for both prompt and generation.
    # The engine must not infer them from mutable request state after submission.
    depth: int | None = None
    chunk_start: int | None = None
    completes_prompt: bool = False
    phase: Stage | None = None


@dataclass(frozen=True)
class SchedulerOutput:
    stage: Stage
    requests: list[ScheduledRequest]

    @property
    def is_speculative(self) -> bool:
        """Execution-mode flag for this batch, not the engine-wide configuration."""
        return self.stage == Stage.SPECULATIVE

    def phase_for(self, scheduled_request):
        return scheduled_request.phase or self.stage

    @property
    def num_tokens(self) -> int:
        return sum(scheduled_request.token_count for scheduled_request in self.requests)


@dataclass(frozen=True)
class PrefillProgress:
    """Logical host-side progress for a submitted split-prefill request.

    These flags describe request frontiers, not GPU completion. Device work may
    still be pending when the chunk or prompt becomes logically complete.
    """

    chunk_complete: bool
    prompt_complete: bool


class ResumeResult(Enum):
    NOT_PREEMPTED = auto()
    RESTORED = auto()
    BLOCKED = auto()


@dataclass(frozen=True)
class AdmissionPlan:
    """One candidate's allocation inputs and admission budget, not a reservation."""

    capacity_tokens: int
    initial_tokens: int
    prefix_blocks: tuple[tuple[int, ...], ...]
    cached_tokens: int
    required_blocks: int
    reserved_growth_blocks: int
    admission_headroom_blocks: int
    cached_claim_blocks: int

    @property
    def total_budget_blocks(self) -> int:
        return (
            self.required_blocks
            + self.reserved_growth_blocks
            + self.admission_headroom_blocks
            + self.cached_claim_blocks
        )


class Scheduler:
    def __init__(
        self,
        config: SchedulerConfig,
        cache_manager,
        speculative_config=None,
        atomic_prefill: bool = True,
        *,
        accepts_prompt: bool = True,
        produces_tokens: bool = True,
    ):
        self.config = config
        self.cache_manager = cache_manager
        self.speculative_config = speculative_config
        self.requests: dict[str, Request] = {}
        self.queues: dict[Stage, deque[str]] = {s: deque() for s in Stage}
        # Bound by Engine when preemption is enabled. These callbacks can copy
        # device state and mutate queues; they are operations, not predicates.
        self.preempt_callback = None
        self.resume_callback = None
        # Optional connector readiness predicate. Never mutates queues or blocks.
        self.execution_ready = None
        # One scheduling boundary per chunk (atomic) or per recurrence depth (split).
        # The policy is derived from the same flag so the two cannot disagree.
        self.atomic_prefill = atomic_prefill
        self.accepts_prompt = accepts_prompt
        self.policy = create_scheduling_policy(
            config, speculative_config, atomic_prefill, produces_tokens=produces_tokens
        )

    def add_request(self, request: Request):
        if request.request_id in self.requests:
            raise ValueError(f"duplicate request ID: {request.request_id}")
        self.requests[request.request_id] = request
        self.queues[Stage.WAITING].append(request.request_id)

    def enqueue(self, request: Request, stage: Stage):
        request.stage = stage
        self.queues[stage].append(request.request_id)

    def suspend_for_transfer(self, request):
        """Stop scheduling without freeing resources that a transfer still borrows."""
        if self.requests.get(request.request_id) is not request:
            return False
        for queue in self.queues.values():
            while request.request_id in queue:
                queue.remove(request.request_id)
        request.stage = Stage.TRANSFERRING
        return True

    def finish(self, request: Request, reason: FinishReason):
        # A delayed EOS can stop a request already queued for its next stage.
        for queue in self.queues.values():
            while request.request_id in queue:
                queue.remove(request.request_id)
        self.cache_manager.free(request.request_id)
        request.stage = Stage.FINISHED
        request.finish_reason = reason
        request.hidden_state = None
        request.input_token_tensor = None
        request.num_output_placeholders = 0
        request.num_speculative_rounds = 0
        self.requests.pop(request.request_id)

    def abort(self, request_id: str) -> Request:
        request = self.requests[request_id]
        self.finish(request, FinishReason.ABORT)
        return request

    @property
    def has_unfinished_requests(self) -> bool:
        return bool(self.requests)

    def _order_waiting_requests(self) -> deque[str]:
        """Return the live waiting queue in admission order.

        Priority is supplied by the caller, not estimated from prompt length.
        Lower values go first: -5, 0, 10. Equal values retain current queue order.
        FCFS preserves queue order, but admission may still bypass blocked requests.
        """
        waiting = self.queues[Stage.WAITING]
        if self.config.policy == "priority":
            ordered = sorted(waiting, key=lambda rid: self.requests[rid].sampling_params.priority)
            waiting.clear()
            waiting.extend(ordered)
        return waiting

    def _ensure_active_slot(self, request: Request, active_count: int) -> int | None:
        """Return the updated active count, or None when admission must stop.

        A full engine may make room for a higher-priority arrival. The callback
        suspends another request and moves it to WAITING; it does not admit the
        request. This operation can synchronize the runner and copy KV to CPU.
        Admission runs before any batch is selected, so no request is excluded
        from preemption here.
        """
        if active_count < self.config.max_num_seqs:
            return active_count
        if (
            self.config.policy == "priority"
            and self.preempt_callback is not None
            and self.preempt_callback(request, priority_only=True, excluded=frozenset())
        ):
            return active_count - 1
        return None

    def _try_resume(self, request: Request) -> ResumeResult:
        """Translate the existing restoration callback's three-way result.

        None: no snapshot, continue fresh admission.
        True: resources/state restored and the original stage already enqueued.
        False: snapshot exists, but restoration cannot obtain capacity yet.
        """
        restored = self.resume_callback(request) if self.resume_callback is not None else None
        if restored is None:
            return ResumeResult.NOT_PREEMPTED
        return ResumeResult.RESTORED if restored else ResumeResult.BLOCKED

    def _reserved_growth_blocks(self, *, reserve_outputs: bool) -> int:
        """Budget future growth of existing requests without allocating it.

        Example: a request needs 12 blocks eventually but owns 4. Without
        preemption, reserve the remaining 8 even though they still look free.
        With preemption, protect in-progress prompts; decode growth may instead
        reclaim capacity by suspending another request later.
        """
        cache = self.cache_manager
        reserved = 0
        for request in self.requests.values():
            active = request.stage not in (Stage.WAITING, Stage.RECEIVING)
            if request.stage not in (
                Stage.PREFILL,
                Stage.PREFILLING_PRELUDE,
                Stage.PREFILLING_RECURRENT,
            ) and not (reserve_outputs and active):
                continue
            tokens = len(request.prompt_token_ids)
            if reserve_outputs:
                tokens += request.sampling_params.max_tokens - 1
            allocated = cache.allocated_blocks(request.request_id)
            reserved += max(0, cache.required_blocks(tokens) - allocated)
        return reserved

    def _plan_admission(self, request: Request, active_count: int) -> AdmissionPlan:
        """Describe prefix reuse, physical allocation and logical admission cost.

        With 8 prompt tokens and 5 outputs, capacity is 12: the last sampled
        output is never fed back into the model. Incremental allocation may
        acquire fewer pages now, while budgeting the remaining growth.

        Prefix lookup can publish completed prefixes and update cache LRU order;
        planning does not allocate request pages, but is not a pure cache read.
        All block counts include the cache's storage-depth multiplier.
        """
        cache = self.cache_manager
        capacity = len(request.prompt_token_ids) + request.sampling_params.max_tokens - 1
        prefix = cache.lookup_prefix(request.prompt_token_ids)
        cached_tokens = len(prefix) * cache.block_size
        initial_tokens = min(
            len(request.prompt_token_ids), cached_tokens + self.config.prefill_chunk_size
        )
        if not cache.incremental_allocation:
            initial_tokens = capacity
        reserve_outputs = cache.incremental_allocation and self.preempt_callback is None
        budget_tokens = (
            len(request.prompt_token_ids)
            if cache.incremental_allocation and not reserve_outputs
            else capacity
        )
        required = cache.required_blocks(budget_tokens) - len(prefix) * cache.storage_depths
        reserved = self._reserved_growth_blocks(reserve_outputs=reserve_outputs)
        # Allow an otherwise idle engine to start a large request.
        admission_headroom_blocks = cache.watermark_blocks if active_count else 0
        # num_free_blocks includes evictable prefix-only blocks. Reusing those
        # blocks makes them non-evictable: do not count the same capacity twice.
        # Example: free=8 fresh + 4 cached, demand=12 with a 4-block prefix.
        # New demand is 8, but the cached claim is 4: admission costs 12, not 8.
        cached_claim = cache.exclusive_prefix_blocks(prefix)
        return AdmissionPlan(
            capacity_tokens=capacity,
            initial_tokens=initial_tokens,
            prefix_blocks=prefix,
            cached_tokens=cached_tokens,
            required_blocks=required,
            reserved_growth_blocks=reserved,
            admission_headroom_blocks=admission_headroom_blocks,
            cached_claim_blocks=cached_claim,
        )

    def _commit_admission(self, request: Request, plan: AdmissionPlan) -> bool:
        """Allocate pages and enter PREFILL only when the whole budget fits."""
        cache = self.cache_manager
        if plan.total_budget_blocks > cache.num_free_blocks:
            return False
        if not cache.allocate(
            request.request_id,
            plan.capacity_tokens,
            initial_tokens=plan.initial_tokens,
            prefix=plan.prefix_blocks,
        ):
            return False
        request.num_prefilled_tokens = plan.cached_tokens
        # A reused prompt prefix is already prefilled at every depth, so the
        # staged window starts closed at that frontier and opens for the rest.
        request.num_preluded_tokens = plan.cached_tokens
        request.prefill_chunk_start = plan.cached_tokens
        if request.prefill_depth_ranges:
            request.prefill_depth_ranges[:] = Request.prefill_boundaries(
                len(request.prompt_token_ids),
                len(request.prefill_depth_ranges),
                start=plan.cached_tokens,
            )
        if self.atomic_prefill:
            self.enqueue(request, Stage.PREFILL)
        else:
            self.enqueue(request, Stage.PREFILLING_PRELUDE)
        return True

    def _admit(self):
        """Admit waiting requests: order -> slot -> restore/allocate -> defer/enter.

        A blocked long request may let a short one pass. Each successful fresh
        admission increments earlier blocked requests' bypass counts; reaching
        the limit stops further bypasses so active requests can drain.
        """
        active_count = sum(
            r.stage not in (Stage.WAITING, Stage.RECEIVING) for r in self.requests.values()
        )
        waiting = self._order_waiting_requests()
        deferred = []
        scan_count = min(len(waiting), self.config.admission_scan_limit)
        for _ in range(scan_count):
            if not waiting:
                break
            # Peek before popping: preemption may append its victim to WAITING.
            request = self.requests[waiting[0]]
            available_count = self._ensure_active_slot(request, active_count)
            if available_count is None:
                break
            active_count = available_count
            request_id = waiting.popleft()
            request = self.requests[request_id]

            resume_result = self._try_resume(request)
            if resume_result == ResumeResult.RESTORED:
                active_count += 1
                continue
            if resume_result == ResumeResult.BLOCKED:
                deferred.append(request_id)
                continue

            plan = self._plan_admission(request, active_count)
            if not self._commit_admission(request, plan):
                deferred.append(request_id)
                if request.admission_bypasses >= self.config.max_admission_bypasses:
                    break
                continue

            active_count += 1
            for blocked_id in deferred:
                self.requests[blocked_id].admission_bypasses += 1
            if any(
                self.requests[rid].admission_bypasses >= self.config.max_admission_bypasses
                for rid in deferred
            ):
                break
        # [A, B] deferred must precede the untouched tail in the same order.
        waiting.extendleft(reversed(deferred))

    def _make_scheduled_request(
        self, request: Request, stage: Stage, token_budget: int
    ) -> ScheduledRequest:
        """Snapshot one request's execution range without advancing its progress.

        Prompt length 10, prefilled 4, chunk 4, budget 6 -> range [4, 8).
        Decode stages contribute one position; its loop may differ by request.
        """
        if stage == Stage.PREFILL:
            start = request.num_prefilled_tokens
            count = min(
                token_budget, self.config.prefill_chunk_size, len(request.prompt_token_ids) - start
            )
            return ScheduledRequest(request, start, count)
        # The two split stages are unreachable while atomic prefill is enabled.
        if stage == Stage.PREFILLING_PRELUDE:
            start = request.num_preluded_tokens
            count = min(
                token_budget,
                self.config.prefill_chunk_size,
                len(request.prompt_token_ids) - start,
            )
            return ScheduledRequest(request, start, count, chunk_start=start)
        if stage == Stage.PREFILLING_RECURRENT:
            start = request.num_prefilled_tokens
            if start >= request.num_preluded_tokens:
                return ScheduledRequest(
                    request, start, 0, depth=0, chunk_start=request.prefill_chunk_start
                )
            first_token_depth = request.prefill_depth(start)
            # One run per depth frontier: every token of the run already completed
            # this depth, and no earlier token is missing a shallower one, so the
            # plane's causal prefix covers the run. Tokens past num_preluded_tokens
            # have no resident prelude and cannot run yet.
            boundary = request.prefill_depth_ranges[first_token_depth]
            count = min(
                token_budget,
                self.config.prefill_chunk_size,
                request.num_preluded_tokens - start,
                boundary[1] - start,
                len(request.prompt_token_ids) - start,
            )
            return ScheduledRequest(
                request,
                start,
                count,
                depth=first_token_depth,
                chunk_start=request.prefill_chunk_start,
                completes_prompt=(
                    first_token_depth == len(request.prefill_depth_ranges) - 1
                    and start + count == len(request.prompt_token_ids)
                ),
            )
        if stage == Stage.SPECULATIVE:
            # K candidates plus one bonus distribution. At the output limit,
            # K=0 is an ordinary fixed-depth step and needs no extra KV slot.
            remaining = request.sampling_params.max_tokens - request.num_scheduled_outputs
            count = min(self.speculative_config.num_speculative_tokens + 1, remaining, token_budget)
            return ScheduledRequest(request, request.position, count)
        if stage == Stage.RECURRENT:
            return ScheduledRequest(request, request.position, depth=request.loops_done)
        return ScheduledRequest(request)

    def _ensure_execution_capacity(
        self, request: Request, frontier: int, *, selected: set[str]
    ) -> bool:
        """Grow KV for this step; if needed, preempt one other request and retry.

        frontier is an exclusive token count, not a loop count. Another loop at
        the same token position normally uses existing capacity. selected holds
        the requests already chosen for this batch; preemption must not suspend
        them, so it is handed a frozen snapshot of that batch-local selection
        instead of reading shared Scheduler state.
        """
        cache = self.cache_manager
        if cache.ensure_capacity(request.request_id, frontier):
            return True
        if self.preempt_callback is None or not self.preempt_callback(
            request, excluded=frozenset(selected)
        ):
            return False
        return cache.ensure_capacity(request.request_id, frontier)

    def _prelude_window_open(self, request: Request) -> bool:
        """Whether the next prompt chunk may be staged for this request.

        Split prefill keeps one staged chunk: deeper depths of the current chunk
        still read its hidden rows, so the next chunk is staged only after the
        current one reached the last depth. Wavefront chunked prefill widens the
        window to several chunks, and then this bound becomes a chunk count.
        """
        return request.num_preluded_tokens == request.num_prefilled_tokens

    def has_ready(self, phase: Stage) -> bool:
        """A blocked transport queue must not hide another runnable phase."""
        queue = self.queues[phase]
        if self.execution_ready is None:
            return bool(queue)
        return any(self.execution_ready(self.requests[rid]) for rid in queue)

    def has_ready_phases(self, *phases: Stage) -> bool:
        """Return whether every listed phase has at least one runnable request."""
        return bool(phases) and all(self.has_ready(phase) for phase in phases)

    def _take(self, stage: Stage) -> SchedulerOutput | None:
        return self.take_stages((stage,), label=stage)

    def take_stages(self, stages, *, label, eligible=None):
        """Select compatible requests across queues under one budget/reservation set."""
        eligible = eligible if eligible is not None else self.execution_ready
        token_budget = self.config.max_num_batched_tokens
        scheduled_requests = []
        selected: set[str] = set()
        for stage in stages:
            queue = self.queues[stage]
            remaining = len(queue)
            while (
                queue
                and token_budget
                and len(scheduled_requests) < self.config.max_num_seqs
                and remaining
            ):
                remaining -= 1
                request = self.requests[queue.popleft()]
                if eligible is not None and not eligible(request):
                    queue.append(request.request_id)
                    continue
                if stage == Stage.PREFILLING_PRELUDE and not self._prelude_window_open(request):
                    queue.append(request.request_id)
                    continue
                if stage == Stage.SPECULATIVE and (
                    request.num_speculative_rounds >= 2
                    or request.num_scheduled_outputs >= request.sampling_params.max_tokens
                ):
                    queue.append(request.request_id)
                    continue
                scheduled_request = self._make_scheduled_request(request, stage, token_budget)
                if (
                    stage in (Stage.PREFILLING_PRELUDE, Stage.PREFILLING_RECURRENT)
                    and scheduled_request.token_count <= 0
                ):
                    queue.append(request.request_id)
                    continue
                if stage in _CAPACITY_STAGES:
                    # A scheduled span declares a prompt frontier; decode stages advance the
                    # request's own position by one.
                    frontier = (
                        scheduled_request.token_start + scheduled_request.token_count
                        if stage in _SPAN_STAGES
                        else request.position + 1
                    )
                    if not self._ensure_execution_capacity(request, frontier, selected=selected):
                        queue.append(request.request_id)
                        continue
                # Selecting A must prevent B's later capacity check from evicting A.
                selected.add(request.request_id)
                scheduled_requests.append(replace(scheduled_request, phase=stage))
                token_budget -= scheduled_request.token_count
        if not scheduled_requests:
            return None
        # Fairness follows the requests actually selected, including a mixed attempt
        # whose other queue could not reserve capacity.
        selected_phases = {scheduled_request.phase for scheduled_request in scheduled_requests}
        recorded = (
            Stage.RECURRENT if Stage.RECURRENT in selected_phases else scheduled_requests[0].phase
        )
        self.policy.record_batch(recorded)
        return SchedulerOutput(label, scheduled_requests)

    def schedule(self, *, prefer_recurrent=False) -> SchedulerOutput | None:
        """Choose queued requests or an admission opportunity, then build a batch.

        An unfinished request is not necessarily runnable (e.g. PD RECEIVING).
        Conversely, one long active request must not prevent new requests from
        filling available slots. The stage policy decides when to call _admit;
        merely checking whether requests exist must not allocate or preempt.
        """
        if not self.requests:
            return None
        if not self.accepts_prompt and self.queues[Stage.WAITING]:
            raise ValueError("decode stage requests must enter through reserve_transfer")
        return self.policy.schedule(self, prefer_recurrent=prefer_recurrent)
