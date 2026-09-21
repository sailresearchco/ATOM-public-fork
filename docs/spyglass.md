# Spyglass programmable probes

Install Spyglass in the worker environment. Set `SPYGLASS_PERF_HOOKS` to an
absolute JSON config available on every worker. Registration happens before
model construction; disabled workers do not import Spyglass. The config selects
callables and optional metadata providers; no per-model instrumentation is needed
in ATOM. Use the Spyglass `examples/performance/atom-k3*.json` config matching the
loaded transport, adapting targets to the engine revision.

With `capture_control: "engine"`, existing start/stop profiler endpoints also
start/stop bounded probe reports. Set `report_dir` (or `handshake_dir`) to a writable
worker-visible directory. Reports include rank, process, observed/sample counts,
explicit metadata errors, overflow counts, and local parent-child IDs. Join each
report event's `trace_label` to the native ROCm Chrome trace. IDs and host clocks
are local; they do not establish matched invocations or synchronized ranks.

Providers are external Python callables receiving `phase`, `args`, `kwargs`, and
`result`. Return bounded JSON facts. They must not synchronize or copy GPU tensors
during measurement. Inspect shapes/dtypes and existing host metadata instead.
Provider errors are recorded without replacing model results/exceptions. ATOM's
existing profiler remains responsible for GPU activity and trace export.

Python hooks observe eager execution and graph construction, not inner HIP graph
replay. Use the outer forward scope and device trace for replay; registration
does not prove execution. Compiled/opaque callables may need a different external
target. Async callback scopes mark host execution, not device completion.
Instrumented timing cannot establish an unprofiled speedup. These changes do not
supply replay/reset, numerical references, or production-wide validation.

The programmable lifecycle API requires Spyglass commit `ea8a8f3` or newer from
`sailresearchco/spyglass` (`codex/atom-programmable-probes`). Install that revision
in every worker image before enabling the hook config. No Spyglass dependency is
required with capture/performance hooks disabled.

Upstream maintenance: this integration is independent of K3 serving changes.
Keep `upstream` pointing at `ROCm/ATOM`; merge upstream updates into fork `main`
and preserve this small optional integration. Do not merge the entire serving
branch merely to carry probes forward.
