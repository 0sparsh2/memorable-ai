"""LLM client guards, LLM-backed summaries, provider adapters and runtime helpers."""

import asyncio
import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from memorable_ai.core import llm_client
from memorable_ai.core.episodic import EpisodicMemory
from memorable_ai.core.providers import (
    AnthropicAdapter,
    AsyncStreamProxy,
    LiteLLMAdapter,
    OpenAIChatAdapter,
    OpenAIResponsesAdapter,
    SyncStreamProxy,
)
from memorable_ai.core.scope import BackgroundWorker, Scope, active_scope, current_user, run_sync
from memorable_ai.core.storage import Storage
from memorable_ai.core.summary import ConversationSummaryStore


def _reply(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def test_parse_json_tolerates_fences_and_prose():
    assert llm_client.parse_json('```json\n[{"a": 1}]\n```') == [{"a": 1}]
    assert llm_client.parse_json('Sure! Here you go: {"x": [1, 2]} hope that helps') == {"x": [1, 2]}
    with pytest.raises(ValueError):
        llm_client.parse_json("no json here")


def test_llm_disabled_values_and_circuit_breaker(monkeypatch):
    for off in (None, "", "none", "off", "disabled"):
        assert not llm_client.llm_enabled(off)
    llm_client.reset_circuit_breaker()
    assert llm_client.llm_enabled("acme/model")

    import litellm

    async def boom(**kwargs):
        raise Exception("AuthenticationError: invalid api key")

    monkeypatch.setattr(litellm, "acompletion", boom)
    with pytest.raises(Exception):
        asyncio.run(llm_client.acompletion("acme/model", [{"role": "user", "content": "hi"}]))
    assert not llm_client.llm_enabled("acme/model"), "auth failures trip the breaker"
    llm_client.reset_circuit_breaker()


@pytest.mark.asyncio
async def test_llm_episode_summary_and_session_fold(tmp_path, monkeypatch):
    storage = Storage(f"sqlite:///{tmp_path / 's.db'}")
    replies = iter(
        [
            json.dumps(
                {
                    "title": "Kyoto trip planning",
                    "summary": "User plans a 10-day Kyoto trip with Maria in April 2026.",
                    "key_points": ["Trip starts 2026-04-03", "Wants a day trip to Nara"],
                    "entities": ["Kyoto", "Maria", "Nara"],
                }
            ),
            "User is planning a Kyoto trip with Maria in April 2026.",
        ]
    )

    async def fake(**kwargs):
        return _reply(next(replies))

    monkeypatch.setattr(llm_client, "acompletion_with_fallback", fake)
    monkeypatch.setattr(llm_client, "acompletion", fake)
    from memorable_ai.graph.knowledge_graph import KnowledgeGraph

    kg = KnowledgeGraph(storage)
    ep = EpisodicMemory(storage, embedder="hash", llm_model="fake-model", knowledge_graph=kg)
    scope = Scope(user_id="u", session_id="s")
    await ep.ingest(scope, [{"role": "user", "content": "Help me plan Kyoto with Maria"}], "Sure!", datetime(2026, 1, 5))
    eid = ep.list_episodes(scope)[0]["id"]
    out = await ep.close_episode(eid)
    assert out["title"] == "Kyoto trip planning" and "Nara" in out["entities"]
    assert storage.lexical_search("episode", "Nara day trip", user_id="u")[0][0] == eid
    summaries = ConversationSummaryStore(storage, llm_model="fake-model")
    await summaries._save("s", "Earlier: user asked about flights.")
    merged = await summaries.fold_episode("s", out)
    assert "Kyoto" in merged
    assert await ConversationSummaryStore(storage).get_summary("s") == merged  # persisted per session


def test_adapters_inject_without_breaking_api_shapes():
    ctx = "<memory>x</memory>"
    oa = OpenAIChatAdapter().inject({"messages": [{"role": "user", "content": "hi"}]}, ctx)
    assert oa["messages"][0] == {"role": "system", "content": ctx}
    parts = OpenAIChatAdapter().inject(
        {"messages": [{"role": "developer", "content": [{"type": "text", "text": "rules"}]}]}, ctx
    )
    assert parts["messages"][0]["content"][-1] == {"type": "text", "text": ctx}
    an = AnthropicAdapter().inject({"system": [{"type": "text", "text": "s"}], "messages": []}, ctx)
    assert an["system"][-1]["text"] == ctx
    assert AnthropicAdapter().inject({"messages": []}, ctx)["system"] == ctx
    rs = OpenAIResponsesAdapter()
    assert rs.get_messages({"input": "hello"}) == [{"role": "user", "content": "hello"}]
    assert rs.get_hints({"conversation": "conv_1"})["session_id"] == "conv_1"
    args, kwargs = LiteLLMAdapter().bind(("gpt", [{"role": "user", "content": "x"}]), {})
    assert args == ("gpt",) and kwargs["messages"][0]["content"] == "x"
    assert OpenAIChatAdapter().get_hints({"user": "u7", "metadata": {"session_id": "s7"}}) == {
        "user_id": "u7",
        "session_id": "s7",
    }


def test_stream_proxies_collect_text():
    seen = []
    chunks = [{"choices": [{"delta": {"content": "a"}}]}, {"choices": [{"delta": {"content": "b"}}]}]
    proxy = SyncStreamProxy(iter(chunks), OpenAIChatAdapter(), seen.append)
    assert list(proxy) == chunks and seen == ["ab"]

    async def agen():
        yield {"type": "response.output_text.delta", "delta": "hi"}
        yield {"type": "response.completed"}

    got = []

    async def consume():
        async with AsyncStreamProxy(agen(), OpenAIResponsesAdapter(), got.append) as s:
            async for _ in s:
                pass

    asyncio.run(consume())
    assert got == ["hi"]


def test_scope_contextvars_and_run_sync():
    with active_scope(user_id="ctx-user"):
        assert current_user() == "ctx-user"

        async def inner():
            return current_user()

        assert run_sync(inner) == "ctx-user"

        async def nested():
            return run_sync(inner)  # sync call from inside a running loop

        assert asyncio.run(nested()) == "ctx-user"
    assert current_user() is None


def test_background_worker_runs_jobs_in_order():
    worker = BackgroundWorker(name="test-worker")
    out = []

    def job(i):
        async def run():
            await asyncio.sleep(0.001 * (5 - i))
            out.append(i)

        return run

    for i in range(5):
        worker.submit(job(i))
    worker.flush(timeout=5)
    worker.stop()
    assert out == [0, 1, 2, 3, 4]
