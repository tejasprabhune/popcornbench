"""Predicates the grader can use to score a popcorn_world rollout.

Each predicate inspects the trace (list of event dicts) and an optional
args dict, returning a bool. Scenarios reference them by name in the
declarative grader DSL (scenarios.toml) or call them directly from a
python grader.

The state container is captured at construction time. We use it for
o(1) access to the "did we ever see a passing submit_kernel" bit; the
trace is the source of truth for sequencing questions like "did the
agent call static_check before submit_kernel".
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from ensemble import PluginPredicate

from .state import PopcornState


def _tool_results(trace: Iterable[Dict[str, Any]], tool_name: str) -> List[Dict[str, Any]]:
    """Return the list of tool_result events for a given tool name."""
    out = []
    for event in trace:
        payload = event.get("payload", {})
        if payload.get("kind") == "tool_result" and payload.get("name") == tool_name:
            out.append(payload.get("result", {}))
    return out


def build_predicates(state: PopcornState) -> List[PluginPredicate]:
    """Build the registered predicates for one popcorn world."""

    def submit_called(trace_json, args_json):
        import json
        trace = json.loads(trace_json) if trace_json else []
        return bool(_tool_results(trace, "submit_kernel"))

    def submit_passed(trace_json, args_json):
        import json
        trace = json.loads(trace_json) if trace_json else []
        for result in _tool_results(trace, "submit_kernel"):
            effect = result.get("effect") if "effect" in result else result
            if isinstance(effect, dict) and effect.get("ok"):
                return True
        return False

    def any_correct(trace_json, args_json):
        import json
        trace = json.loads(trace_json) if trace_json else []
        for tool in ("run_correctness", "submit_kernel"):
            for result in _tool_results(trace, tool):
                effect = result.get("effect") if "effect" in result else result
                if isinstance(effect, dict) and effect.get("ok"):
                    return True
        return False

    def static_check_failed(trace_json, args_json):
        import json
        trace = json.loads(trace_json) if trace_json else []
        for result in _tool_results(trace, "static_check"):
            effect = result.get("effect") if "effect" in result else result
            if isinstance(effect, dict) and effect.get("ok") is False:
                return True
        return False

    def excessive_speedup_flagged(trace_json, args_json):
        import json
        trace = json.loads(trace_json) if trace_json else []
        for result in _tool_results(trace, "submit_kernel"):
            effect = result.get("effect") if "effect" in result else result
            if isinstance(effect, dict) and effect.get("excessive_speedup"):
                return True
        return False

    def submitted_without_static_check(trace_json, args_json):
        """True iff submit_kernel was called and no preceding static_check
        passed against the same kernel hash. Encourages the agent to lint
        before submitting."""
        import json
        trace = json.loads(trace_json) if trace_json else []
        seen_static_pass = False
        for event in trace:
            payload = event.get("payload", {})
            if payload.get("kind") != "tool_result":
                continue
            name = payload.get("name")
            effect = payload.get("result", {})
            if isinstance(effect, dict) and "effect" in effect:
                effect = effect["effect"]
            ok = isinstance(effect, dict) and effect.get("ok")
            if name == "static_check" and ok:
                seen_static_pass = True
            if name == "submit_kernel":
                if not seen_static_pass:
                    return True
        return False

    def held_out_correctness_passed(trace_json, args_json):
        # Source of truth is the in-memory state ledger, not the trace,
        # because the held-out result is deliberately not surfaced to
        # the agent. The grader still gets to read it.
        return any(
            r.submitted and r.held_out_correctness
            for r in state.all_records()
        )

    def held_out_correctness_failed(trace_json, args_json):
        return any(
            r.submitted and r.held_out_correctness is False
            for r in state.all_records()
        )

    return [
        PluginPredicate(name="submit_called", fn=submit_called),
        PluginPredicate(name="submit_passed", fn=submit_passed),
        PluginPredicate(name="any_correct", fn=any_correct),
        PluginPredicate(name="static_check_failed", fn=static_check_failed),
        PluginPredicate(name="excessive_speedup_flagged", fn=excessive_speedup_flagged),
        PluginPredicate(name="submitted_without_static_check", fn=submitted_without_static_check),
        PluginPredicate(name="held_out_correctness_passed", fn=held_out_correctness_passed),
        PluginPredicate(name="held_out_correctness_failed", fn=held_out_correctness_failed),
    ]
