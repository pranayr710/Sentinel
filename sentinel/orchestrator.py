"""The Incident Commander — SENTINEL's core agentic loop.

Lifecycle of an incident:

  DETECTED -> DIAGNOSING -> VERIFYING (self-heal loop until tests green)
           -> AWAITING_APPROVAL (only if confidence below threshold)
           -> DEPLOYING -> REPORTING -> RESOLVED
                                     -> ESCALATED (rejected / failed / timed out)

The diagnose/patch/verify phase is a manual Claude tool-use loop: the agent
reads logs and source, proposes patches into an isolated sandbox, and runs
the test suite there. It can only submit a resolution after a green test run.
Every thought, tool call and result is published to the event bus and the
audit log.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from sentinel import confidence, llm
from sentinel.audit import audit
from sentinel.config import settings
from sentinel.events import TraceEvent, bus
from sentinel.tools import actions, observability
from sentinel.tools.sandbox import Sandbox

# ── incident model ────────────────────────────────────────────────────────

STATUSES = (
    "DETECTED", "DIAGNOSING", "VERIFYING", "AWAITING_APPROVAL",
    "DEPLOYING", "REPORTING", "RESOLVED", "ESCALATED",
)


@dataclass
class Incident:
    id: str
    symptom: str
    opened_at: float = field(default_factory=time.time)
    status: str = "DETECTED"
    diagnosis: str = ""
    fix_summary: str = ""
    diff: str = ""
    test_summary: str = ""
    iterations: int = 0
    verdict: Optional[dict] = None
    actions: list[dict] = field(default_factory=list)
    approval_note: str = ""
    closed_at: Optional[float] = None
    _approval: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    _approved: bool = field(default=False, repr=False)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "symptom": self.symptom,
            "opened_at": self.opened_at,
            "status": self.status,
            "diagnosis": self.diagnosis,
            "fix_summary": self.fix_summary,
            "diff": self.diff,
            "test_summary": self.test_summary,
            "iterations": self.iterations,
            "verdict": self.verdict,
            "actions": self.actions,
            "approval_note": self.approval_note,
            "closed_at": self.closed_at,
        }


# ── agent tool schema ─────────────────────────────────────────────────────

TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_health",
        "description": "Probe the live service's /health endpoint. Returns status code and body.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "read_logs",
        "description": "Tail the service's application log. Call this first — errors and tracebacks land here.",
        "input_schema": {
            "type": "object",
            "properties": {"lines": {"type": "integer", "description": "How many trailing lines (default 80)."}},
            "additionalProperties": False,
        },
    },
    {
        "name": "list_files",
        "description": "List all files in the service codebase (sandbox copy).",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "read_file",
        "description": "Read a source file with line numbers. Path is relative to the service root.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_code",
        "description": "Regex search across the service codebase. Returns file, line number and matching text.",
        "input_schema": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}},
            "required": ["pattern"],
            "additionalProperties": False,
        },
    },
    {
        "name": "propose_patch",
        "description": (
            "Apply an exact string replacement to a file in the verification sandbox. "
            "`old` must match exactly one location. Make the minimal change that fixes the root cause."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old": {"type": "string", "description": "Exact text to replace (must be unique in the file)."},
                "new": {"type": "string", "description": "Replacement text."},
            },
            "required": ["path", "old", "new"],
            "additionalProperties": False,
        },
    },
    {
        "name": "run_tests",
        "description": "Run the service's full pytest suite against the patched sandbox. You MUST get a passing run before submitting a resolution.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "submit_resolution",
        "description": (
            "Submit your final resolution. Only call this AFTER run_tests has passed. "
            "Be honest in `confidence`: it feeds an autonomy gate that decides whether "
            "your fix deploys automatically or waits for human approval."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "root_cause": {"type": "string", "description": "One-paragraph root cause analysis."},
                "fix_summary": {"type": "string", "description": "One-sentence description of the fix."},
                "confidence": {"type": "number", "description": "Your calibrated confidence in this fix, 0.0-1.0."},
            },
            "required": ["root_cause", "fix_summary", "confidence"],
            "additionalProperties": False,
        },
    },
]

SYSTEM_PROMPT = """You are SENTINEL, an autonomous incident commander for production services.

A service you monitor is unhealthy. Your job:
1. Read the logs and health status to understand the symptom.
2. Locate the root cause in the source code (read files, search).
3. Apply the MINIMAL patch that fixes the root cause in the verification sandbox.
4. Run the test suite. If it fails, read the failures, revise your patch, and re-run until green.
5. Submit your resolution with an honest, calibrated confidence score.

Rules:
- Fix root causes, never symptoms. Do not weaken, skip, or modify tests.
- Do not refactor, rename, or "improve" code beyond the fix.
- Keep patches minimal — the blast radius of your change affects whether it can auto-deploy.
- Your confidence score feeds a real autonomy gate; overstating it erodes trust, understating it wastes human time.
- Every step you take is recorded in an audit trail reviewed by humans."""


# ── commander ─────────────────────────────────────────────────────────────

class IncidentCommander:
    def __init__(self) -> None:
        self.incidents: dict[str, Incident] = {}

    # — event helper —
    async def _emit(self, incident: Incident, type_: str, title: str, detail: dict | None = None) -> None:
        event = TraceEvent(incident_id=incident.id, type=type_, title=title, detail=detail or {})
        audit.append(event)
        await bus.publish(event)

    async def _set_status(self, incident: Incident, status: str) -> None:
        incident.status = status
        await self._emit(incident, "incident.status", f"Status → {status}", {"status": status})

    # — public API —
    def active_incident(self) -> Optional[Incident]:
        for inc in self.incidents.values():
            if inc.status not in ("RESOLVED", "ESCALATED"):
                return inc
        return None

    def resolve_approval(self, incident_id: str, approved: bool, note: str = "") -> bool:
        inc = self.incidents.get(incident_id)
        if not inc or inc.status != "AWAITING_APPROVAL":
            return False
        inc._approved = approved
        inc.approval_note = note
        inc._approval.set()
        return True

    async def open_incident(self, symptom: str) -> Incident:
        incident = Incident(id=f"INC-{time.strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}", symptom=symptom)
        self.incidents[incident.id] = incident
        await self._emit(
            incident, "incident.opened",
            f"Incident opened: {symptom.splitlines()[0][:120]}",
            {"symptom": symptom},
        )
        asyncio.create_task(self._run(incident))
        return incident

    # — the pipeline —
    async def _run(self, incident: Incident) -> None:
        try:
            sandbox = Sandbox(incident.id)
            await self._set_status(incident, "DIAGNOSING")
            resolution = await self._agent_loop(incident, sandbox)
            if resolution is None:
                await self._escalate(incident, "agent could not produce a verified fix")
                return

            incident.diagnosis = resolution["root_cause"]
            incident.fix_summary = resolution["fix_summary"]
            incident.diff = sandbox.full_diff()

            # Autonomy gate
            verdict = confidence.assess(
                tests_passed=True,
                agent_confidence=float(resolution["confidence"]),
                iterations_used=max(incident.iterations, 1),
                files_touched=len(sandbox.touched_files),
                lines_changed=sandbox.lines_changed(),
            )
            incident.verdict = verdict.to_dict()
            await self._emit(
                incident, "confidence.verdict",
                f"Confidence {verdict.score:.2f} vs threshold {verdict.threshold:.2f} — "
                + ("autonomous deploy" if verdict.autonomous else "human approval required"),
                incident.verdict,
            )

            if not verdict.autonomous:
                await self._set_status(incident, "AWAITING_APPROVAL")
                await self._emit(
                    incident, "approval.requested",
                    "Awaiting human approval in the dashboard",
                    {"diff": incident.diff, "verdict": incident.verdict},
                )
                try:
                    await asyncio.wait_for(incident._approval.wait(), timeout=settings.approval_timeout)
                except asyncio.TimeoutError:
                    await self._escalate(incident, "approval timed out")
                    return
                if not incident._approved:
                    await self._emit(incident, "approval.rejected", "Human rejected the fix",
                                     {"note": incident.approval_note})
                    await self._escalate(incident, f"fix rejected by human: {incident.approval_note}")
                    return
                await self._emit(incident, "approval.granted", "Human approved the fix",
                                 {"note": incident.approval_note})

            # Deploy
            await self._set_status(incident, "DEPLOYING")
            promoted = sandbox.promote()
            await self._emit(incident, "deploy.promoted", f"Patched files deployed: {', '.join(promoted)}",
                             {"files": promoted})
            await _request_hot_reload()

            recovered = await self._await_recovery(incident)
            if not recovered:
                await self._escalate(incident, "service did not recover after deploy")
                return

            # Report
            await self._set_status(incident, "REPORTING")
            await self._report(incident)

            incident.closed_at = time.time()
            await self._set_status(incident, "RESOLVED")
            await self._emit(
                incident, "incident.resolved",
                f"Resolved in {incident.closed_at - incident.opened_at:.0f}s: {incident.fix_summary}",
                {"duration_s": round(incident.closed_at - incident.opened_at, 1)},
            )
            sandbox.cleanup()
        except Exception as exc:  # noqa: BLE001 — pipeline errors become escalations
            await self._emit(incident, "incident.error", f"Pipeline error: {exc}", {"error": str(exc)})
            await self._escalate(incident, f"pipeline error: {exc}")

    # — agent loop —
    async def _agent_loop(self, incident: Incident, sandbox: Sandbox) -> Optional[dict]:
        messages: list[dict[str, Any]] = [{
            "role": "user",
            "content": (
                f"Incident {incident.id}.\n\nDetected symptom:\n{incident.symptom}\n\n"
                "Diagnose the root cause, patch it in the sandbox, verify with the test "
                "suite, then submit your resolution."
            ),
        }]
        last_tests_passed = False

        for _turn in range(settings.max_agent_turns):
            response = await asyncio.to_thread(llm.complete, SYSTEM_PROMPT, messages, TOOLS)

            # Surface reasoning + narration to the glass box.
            for block in response.content:
                if block.type == "thinking" and getattr(block, "thinking", ""):
                    await self._emit(incident, "agent.thinking", "Reasoning",
                                     {"text": block.thinking})
                elif block.type == "text" and block.text.strip():
                    await self._emit(incident, "agent.message", block.text.strip()[:200],
                                     {"text": block.text})

            if response.stop_reason != "tool_use":
                # Agent stopped without submitting — treat as failure.
                return None

            messages.append({"role": "assistant", "content": response.content})
            tool_results: list[dict[str, Any]] = []

            for block in response.content:
                if block.type != "tool_use":
                    continue
                name, args = block.name, dict(block.input)

                if name == "submit_resolution":
                    if not last_tests_passed:
                        result = {"error": "Refused: run_tests has not passed since your last patch. Verify first."}
                        await self._emit(incident, "tool.blocked",
                                         "submit_resolution blocked — tests not green", result)
                        tool_results.append(_tool_result(block.id, result, is_error=True))
                        continue
                    await self._emit(incident, "agent.resolution", args.get("fix_summary", "Resolution submitted"), args)
                    return args

                result, is_error = await self._execute_tool(incident, sandbox, name, args)
                if name == "run_tests":
                    incident.iterations += 1  # every verification attempt counts
                    last_tests_passed = bool(result.get("passed"))
                elif name == "propose_patch" and not is_error:
                    last_tests_passed = False  # any new patch must be re-verified
                tool_results.append(_tool_result(block.id, result, is_error=is_error))

            messages.append({"role": "user", "content": tool_results})

        await self._emit(incident, "agent.exhausted", "Agent turn limit reached without a resolution", {})
        return None

    async def _execute_tool(self, incident: Incident, sandbox: Sandbox,
                            name: str, args: dict) -> tuple[dict, bool]:
        await self._emit(incident, "tool.call", f"{name}({_short_args(args)})",
                         {"tool": name, "args": args})
        try:
            if name == "get_health":
                result = await observability.get_health()
            elif name == "read_logs":
                result = {"log": observability.read_logs(int(args.get("lines", 80)))}
            elif name == "list_files":
                result = {"files": sandbox.list_files()}
            elif name == "read_file":
                result = {"content": sandbox.read_file(args["path"])}
            elif name == "search_code":
                result = {"matches": sandbox.search(args["pattern"])}
            elif name == "propose_patch":
                diff = sandbox.apply_patch(args["path"], args["old"], args["new"])
                result = {"applied": True, "diff": diff}
                await self._emit(incident, "patch.applied", f"Patch applied to {args['path']}",
                                 {"path": args["path"], "diff": diff})
            elif name == "run_tests":
                if incident.status != "VERIFYING":
                    await self._set_status(incident, "VERIFYING")
                test = await asyncio.to_thread(sandbox.run_tests)
                result = {"passed": test.passed, "summary": test.summary, "output": test.output}
                await self._emit(
                    incident, "tests.result",
                    ("✅ " if test.passed else "❌ ") + test.summary,
                    {"passed": test.passed, "summary": test.summary, "output": test.output},
                )
            else:
                return {"error": f"unknown tool {name}"}, True
            return result, False
        except Exception as exc:  # noqa: BLE001 — tool failures go back to the agent
            await self._emit(incident, "tool.error", f"{name} failed: {exc}", {"error": str(exc)})
            return {"error": str(exc)}, True

    # — recovery, reporting, escalation —
    async def _await_recovery(self, incident: Incident, timeout: float = 45.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            health = await observability.get_health()
            if health.get("healthy"):
                await self._emit(incident, "deploy.verified", "Live service recovered — health check green",
                                 health)
                return True
            await asyncio.sleep(1.5)
        await self._emit(incident, "deploy.failed", "Service still unhealthy after deploy", {})
        return False

    async def _report(self, incident: Incident) -> None:
        title = f"[{incident.id}] {incident.fix_summary}"
        body = (
            f"**Root cause**\n{incident.diagnosis}\n\n"
            f"**Fix**\n{incident.fix_summary}\n\n"
            f"**Verification**\nFull test suite green in sandbox; live health check recovered.\n\n"
            f"**Confidence**\n{json.dumps(incident.verdict, indent=2)}\n\n"
            f"```diff\n{incident.diff}\n```"
        )
        pr = await asyncio.to_thread(actions.open_pull_request, incident.id, incident.fix_summary, body, incident.diff)
        incident.actions.append({"kind": "github_pr", **pr})
        await self._emit(incident, "action.github", f"PR ({pr['mode']}): {pr.get('url', pr.get('branch'))}", pr)

        card = await actions.create_wekan_card(incident.id, title, body)
        incident.actions.append({"kind": "wekan_card", **card})
        await self._emit(incident, "action.wekan", f"WeKan card ({card['mode']}): {card.get('card_id')}", card)

        slack = await actions.post_slack(
            incident.id,
            f":white_check_mark: *{incident.id} resolved* — {incident.fix_summary}\n"
            f"Confidence {incident.verdict['score']:.2f} | {pr.get('url', '')}",
        )
        incident.actions.append({"kind": "slack_post", **slack})
        await self._emit(incident, "action.slack", f"Slack update posted ({slack['mode']})", slack)

    async def _escalate(self, incident: Incident, reason: str) -> None:
        incident.closed_at = time.time()
        await self._set_status(incident, "ESCALATED")
        slack = await actions.post_slack(
            incident.id,
            f":rotating_light: *{incident.id} escalated to humans* — {reason}",
        )
        incident.actions.append({"kind": "slack_escalation", **slack})
        await self._emit(incident, "incident.escalated", f"Escalated: {reason}", {"reason": reason})


# ── helpers ───────────────────────────────────────────────────────────────

async def _request_hot_reload() -> None:
    """Best-effort: ask the service to reload immediately after a deploy."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            await client.post(f"{settings.service_url}/admin/reload")
    except Exception:  # noqa: BLE001 — uvicorn --reload is the fallback
        pass



def _tool_result(tool_use_id: str, result: dict, is_error: bool = False) -> dict:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": json.dumps(result, ensure_ascii=False),
        "is_error": is_error,
    }


def _short_args(args: dict) -> str:
    parts = []
    for k, v in args.items():
        s = str(v).replace("\n", " ")
        parts.append(f"{k}={s[:40]}{'…' if len(s) > 40 else ''}")
    return ", ".join(parts)


commander = IncidentCommander()
