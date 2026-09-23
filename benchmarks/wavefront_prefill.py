"""Compare legacy and wavefront prefill on a controlled long-prompt workload.

The default workload is intentionally small-batch: one long prompt is split
into short chunks so wavefront tasks can expose multiple depth rows to one
recurrent call. This is an engineering comparison, not a paper-speed claim.

Examples:
  python -m benchmarks.wavefront_prefill --device cpu --output /tmp/wavefront.json
  CUDA_VISIBLE_DEVICES=0 python -m benchmarks.wavefront_prefill --device cuda \
      --attention-backend triton --cuda-graphs --output /tmp/wavefront.json
"""

import argparse
import json
import time
from pathlib import Path

import torch

from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroConfig, OuroForCausalLM
from vllm_rlt.request import Stage


def _workload(prompt_length, max_tokens):
    prompts = {"long": [1 + (index % 63) for index in range(prompt_length)]}
    trace = {"long": [4] * max_tokens}
    params = SamplingParams(
        max_tokens=max_tokens,
        min_loops=4,
        max_loops=4,
        exit_threshold=1.0,
        ignore_eos=True,
    )
    return prompts, trace, params


def _make_engine(
    device,
    backend,
    wavefront,
    prompts,
    trace,
    params,
    *,
    prefix,
    async_scheduling,
    cuda_graphs,
    chunk_size,
    batch_tokens,
):
    torch.manual_seed(123)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = OuroForCausalLM(OuroConfig.tiny()).to(device=device, dtype=dtype)
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(
            num_blocks=256,
            block_size=16,
            enable_prefix_caching=prefix,
            incremental_allocation=prefix,
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=8,
            max_num_batched_tokens=batch_tokens,
            prefill_chunk_size=chunk_size,
            wavefront_prefill=wavefront,
        ),
        exit_config=ExitConfig("trace", depths_by_request=trace),
        execution_config=ExecutionConfig(
            async_scheduling=async_scheduling,
            static_buffers=async_scheduling,
            pad_to_power_of_two=async_scheduling,
            cuda_graphs=cuda_graphs,
        ),
        attention_backend=backend,
    )
    for request_id, prompt in prompts.items():
        engine.add_request(request_id, prompt, params)
    return engine


def _drive(engine, *, profile_path=None):
    outputs = {}
    first_output_s = {}
    steps = 0
    prefill_batches = 0
    prefill_rows = 0
    recurrent_batches = 0
    recurrent_rows = 0
    start = time.perf_counter()

    def loop():
        nonlocal steps, prefill_batches, prefill_rows, recurrent_batches, recurrent_rows
        while engine.has_unfinished_requests():
            for output in engine.step():
                if output.request_id not in first_output_s:
                    first_output_s[output.request_id] = time.perf_counter() - start
                if output.finished:
                    outputs[output.request_id] = output
            steps += 1
            batch = engine.last_schedule
            if batch is None:
                continue
            if batch.stage == Stage.PREFILL:
                prefill_batches += 1
                prefill_rows += batch.num_tokens
            elif batch.stage == Stage.RECURRENT:
                recurrent_batches += 1
                recurrent_rows += len(batch.items)
        engine.model_runner.synchronize()

    if profile_path is None:
        loop()
    else:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if engine.model_runner.device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, record_shapes=False) as prof:
            loop()
        prof.export_chrome_trace(str(profile_path))

    elapsed = time.perf_counter() - start
    return dict(
        seconds=elapsed,
        first_output_s=first_output_s,
        output_tokens=sum(len(output.token_ids) for output in outputs.values()),
        tokens_per_second=sum(len(output.token_ids) for output in outputs.values()) / elapsed,
        steps=steps,
        prefill_batches=prefill_batches,
        prefill_rows=prefill_rows,
        prefill_rows_per_batch=prefill_rows / max(prefill_batches, 1),
        recurrent_batches=recurrent_batches,
        recurrent_rows=recurrent_rows,
        graph_captures=getattr(engine.model_runner.graphs, "captures", 0),
        graph_replays=getattr(engine.model_runner.graphs, "replays", 0),
        outputs={rid: output.token_ids for rid, output in outputs.items()},
    )


def benchmark(
    *,
    device="cpu",
    attention_backend=None,
    prompt_length=128,
    max_tokens=4,
    chunk_size=2,
    batch_tokens=4,
    prefix=True,
    async_scheduling=False,
    cuda_graphs=False,
    repeats=3,
    profile_path=None,
):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but CUDA is unavailable")
    backend = attention_backend or ("triton" if device == "cuda" else "torch")
    prompts, trace, params = _workload(prompt_length, max_tokens)
    results = []
    for wavefront in (False, True):
        samples = []
        for repeat in range(repeats):
            engine = _make_engine(
                device,
                backend,
                wavefront,
                prompts,
                trace,
                params,
                prefix=prefix,
                async_scheduling=async_scheduling,
                cuda_graphs=cuda_graphs,
                chunk_size=chunk_size,
                batch_tokens=batch_tokens,
            )
            if device == "cuda":
                torch.cuda.synchronize()
            profile = Path(profile_path) if profile_path and repeat == repeats - 1 else None
            sample = _drive(engine, profile_path=profile)
            if sample["outputs"] != {"long": sample["outputs"]["long"]}:
                raise AssertionError("unexpected benchmark request IDs")
            samples.append(sample)
        median = sorted(samples, key=lambda item: item["seconds"])[len(samples) // 2]
        median["wavefront_prefill"] = wavefront
        results.append(median)

    legacy, wavefront_result = results
    if legacy["outputs"] != wavefront_result["outputs"]:
        raise AssertionError("wavefront output mismatch")
    return dict(
        workload=dict(
            device=device,
            backend=backend,
            prompt_length=prompt_length,
            max_tokens=max_tokens,
            chunk_size=chunk_size,
            batch_tokens=batch_tokens,
            prefix=prefix,
            async_scheduling=async_scheduling,
            cuda_graphs=cuda_graphs,
            repeats=repeats,
        ),
        results=results,
        speedup_seconds=legacy["seconds"] / wavefront_result["seconds"],
        speedup_tokens_per_second=wavefront_result["tokens_per_second"]
        / legacy["tokens_per_second"],
        note=(
            "The legacy row is the same branch with wavefront_prefill disabled; "
            "no paper speedup claim."
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--attention-backend", choices=["torch", "triton"], default=None)
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--prefill-chunk-size", type=int, default=2)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4)
    parser.add_argument("--no-prefix-caching", action="store_true")
    parser.add_argument("--async-scheduling", action="store_true")
    parser.add_argument("--cuda-graphs", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = benchmark(
        device=args.device,
        attention_backend=args.attention_backend,
        prompt_length=args.prompt_length,
        max_tokens=args.max_tokens,
        chunk_size=args.prefill_chunk_size,
        batch_tokens=args.max_num_batched_tokens,
        prefix=not args.no_prefix_caching,
        async_scheduling=args.async_scheduling,
        cuda_graphs=args.cuda_graphs,
        repeats=args.repeats,
        profile_path=args.profile,
    )
    args.output.write_text(json.dumps(report, indent=2))
    for result in report["results"]:
        print(
            "wavefront=",
            result["wavefront_prefill"],
            "seconds=",
            f"{result['seconds']:.4f}",
            "tokens/s=",
            f"{result['tokens_per_second']:.2f}",
            "prefill_rows/batch=",
            f"{result['prefill_rows_per_batch']:.2f}",
        )
    print("speedup_seconds=", f"{report['speedup_seconds']:.3f}")
    print("speedup_tokens_per_second=", f"{report['speedup_tokens_per_second']:.3f}")


if __name__ == "__main__":
    main()
