"""
graph.py – LangGraph workflow definition.
Nodes are wired in dependency order.
Structure is kept so a Chroma retriever can be spliced in at v2
between knowledge_base and predictor without rewriting any other node.

Note on ordering: the graph uses a plain dict as its state, so two nodes that
write in the same step raise InvalidUpdateError ("can receive only one value per
step"). Every branch is therefore wired as a single chain. Parallelising the
independent analysers/generators later requires an Annotated reducer on the
state channels.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, TypedDict

from langgraph.graph import StateGraph, END

from nodes.classifier import classify_node
from nodes.syllabus_analyst import syllabus_node
from nodes.paper_analyst import paper_node
from nodes.teacher_analyst import teacher_node
from nodes.knowledge_base import knowledge_base_node
from nodes.predictor import predictor_node
from nodes.generators import (
    mock_paper_node,
    question_bank_node,
    model_answers_node,
    flashcards_node,
    study_plan_node,
    viva_node,
)
from nodes.examiner import examiner_node

logger = logging.getLogger(__name__)

# LangGraph only carries keys declared in the state schema between nodes. A bare
# ``StateGraph(dict)`` declares no channels, so each node receives just the
# previous node's return value — subject, documents and syllabus disappear after
# one hop. Declaring the schema keeps the whole state available to every node.
class GraphState(TypedDict, total=False):
    # Inputs
    subject: str
    exam_type: str
    exam_pattern: Dict[str, Any]
    difficulty_split: Dict[str, float]
    unit_focus: List[int]
    include_topics: List[str]
    exclude_topics: List[str]
    exam_date: Any
    hours_per_day: float
    mode: str
    language: str
    answer_length: Dict[str, str]
    outputs_wanted: List[str]
    pasted_text: str
    viva_count: int
    practical_focus: str
    course_level: str
    discipline: str

    # Pipeline intermediates
    raw_documents: List[Dict[str, Any]]
    classified_docs: Any
    syllabus: Any
    paper_analysis: Any
    teacher_analysis: Any
    knowledge_base: Dict[str, Any]
    predictions: Any
    priority_list: Any

    # Generated artifacts
    mock_paper: Any
    examiner_output: Any
    question_bank: Any
    model_answers: Any
    flashcards_output: Any
    study_plan: Any
    viva_output: Any

    # Bookkeeping
    warnings: List[str]
    errors: List[str]
    progress: Dict[str, str]


def build_graph(outputs_wanted: List[str], exam_type: str = "") -> StateGraph:
    """
    Build and compile the LangGraph workflow.
    Only generator nodes requested by the user are added.
    Practical/viva exam types always get the viva node.
    """
    g = StateGraph(GraphState)

    # ── Core pipeline (always runs) ───────────────────────────────────────────
    g.add_node("classifier", classify_node)
    g.add_node("syllabus_analyst", syllabus_node)
    g.add_node("paper_analyst", paper_node)
    g.add_node("teacher_analyst", teacher_node)
    g.add_node("knowledge_base", knowledge_base_node)
    g.add_node("predictor", predictor_node)

    g.set_entry_point("classifier")
    g.add_edge("classifier", "syllabus_analyst")
    g.add_edge("syllabus_analyst", "paper_analyst")
    g.add_edge("paper_analyst", "teacher_analyst")
    g.add_edge("teacher_analyst", "knowledge_base")
    g.add_edge("knowledge_base", "predictor")

    # ── Optional generators, chained in dependency order ─────────────────────
    wants_paper = "mock_paper" in outputs_wanted or "model_answers" in outputs_wanted
    wants_answers = "model_answers" in outputs_wanted
    wants_qbank = "question_bank" in outputs_wanted
    wants_cards = "flashcards" in outputs_wanted or "formula_sheet" in outputs_wanted
    wants_plan = "study_plan" in outputs_wanted
    wants_viva = "viva" in outputs_wanted or any(
        key in (exam_type or "").lower() for key in ("viva", "practical")
    )

    chain: List[str] = []
    if wants_paper:
        g.add_node("mock_paper", mock_paper_node)
        chain.append("mock_paper")
        g.add_node("examiner", examiner_node)
        chain.append("examiner")
    if wants_answers:
        g.add_node("model_answers", model_answers_node)
        chain.append("model_answers")
    if wants_qbank:
        g.add_node("question_bank", question_bank_node)
        chain.append("question_bank")
    if wants_cards:
        g.add_node("flashcards", flashcards_node)
        chain.append("flashcards")
    if wants_plan:
        g.add_node("study_plan", study_plan_node)
        chain.append("study_plan")
    if wants_viva:
        g.add_node("viva", viva_node)
        chain.append("viva")

    previous = "predictor"
    for node_name in chain:
        g.add_edge(previous, node_name)
        previous = node_name
    g.add_edge(previous, END)

    return g.compile()


def run_workflow(state: Dict[str, Any]) -> Dict[str, Any]:
    """Entry point: compile graph and stream execution."""
    outputs_wanted = state.get("outputs_wanted", [])
    graph = build_graph(outputs_wanted, exam_type=state.get("exam_type", ""))

    final_state = dict(state)
    for chunk in graph.stream(state, {"recursion_limit": 50}):
        for node_name, node_output in chunk.items():
            if isinstance(node_output, dict):
                final_state.update(node_output)
            logger.info(f"✓ Node '{node_name}' complete")

    return final_state
