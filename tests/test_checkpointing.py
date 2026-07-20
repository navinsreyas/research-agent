"""
SQLite checkpointing (graph.py uses SqliteSaver, NOT Postgres).

Behavioural test: an interrupted run, resumed after a simulated process restart,
restores its accumulated state instead of starting over. Uses the same SqliteSaver
construction the production code uses:
    conn = sqlite3.connect(path, check_same_thread=False); SqliteSaver(conn)

Structural test: the real create_research_graph() is compiled with a SqliteSaver.
No LLM or web calls occur (graph construction only registers nodes).
"""

import operator
import sqlite3
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver


class _S(TypedDict):
    items: Annotated[list, operator.add]   # accumulative, like the real knowledge_base
    steps: int


def _node_a(state):
    return {"items": ["a"], "steps": 1}


def _node_b(state):
    return {"items": ["b"], "steps": 2}


def _build(saver):
    wf = StateGraph(_S)
    wf.add_node("a", _node_a)
    wf.add_node("b", _node_b)
    wf.set_entry_point("a")
    wf.add_edge("a", "b")
    wf.add_edge("b", END)
    # interrupt_before mirrors the production graph pausing before a downstream node
    return wf.compile(checkpointer=saver, interrupt_before=["b"])


def test_sqlite_checkpoint_resume_restores_accumulated_state(tmp_path):
    db = str(tmp_path / "cp.sqlite")
    cfg = {"configurable": {"thread_id": "t1"}}

    conn = sqlite3.connect(db, check_same_thread=False)
    graph = _build(SqliteSaver(conn))

    graph.invoke({"items": [], "steps": 0}, cfg)     # runs 'a', pauses before 'b'
    paused = graph.get_state(cfg)
    assert paused.values["items"] == ["a"]           # accumulated from 'a'
    assert paused.next == ("b",)                       # interrupted before 'b'
    conn.close()

    # Simulate a process restart: brand-new connection + saver on the SAME db file.
    conn2 = sqlite3.connect(db, check_same_thread=False)
    graph2 = _build(SqliteSaver(conn2))
    restored = graph2.get_state(cfg)
    assert restored.values["items"] == ["a"]         # durably restored from disk

    graph2.invoke(None, cfg)                          # resume -> runs 'b'
    final = graph2.get_state(cfg)
    assert final.values["items"] == ["a", "b"]       # accumulation preserved, not restarted
    conn2.close()


def test_production_graph_uses_sqlite_checkpointer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)                        # keep checkpoints.sqlite out of the repo
    import graph as graph_mod
    compiled = graph_mod.create_research_graph()
    assert isinstance(compiled.checkpointer, SqliteSaver)   # SQLite, not Postgres/memory
