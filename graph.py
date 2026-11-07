"""
graph.py – LangGraph workflow definition.
Nodes are wired in dependency order.
Structure is kept so a Chroma retriever can be spliced in at v2
between knowledge_base and predictor without rewriting any other node.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

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
)
from nodes.examiner import examiner_node

logger = logging.getLogger(__name__)

# State is a plain dict (compatible with LangGraph StateGraph)
# Typed WorkflowState (Pydantic) is used only for IDE hints and validation.


def build_graph(outputs_wanted: List[str]) -> StateGraph:
    """
    Build and compile the LangGraph workflow.
    Only generator nodes requested by the user are added.
    """
    g = StateGraph(dict)

    # Core pipeline (always runs)
    g.add_node("classifier", classify_node)
    g.add_node("syllabus_analyst", syllabus_node)
    g.add_node("paper_analyst", paper_node)
    g.add_node("teacher_analyst", teacher_node)
    g.add_node("knowledge_base", knowledge_base_node)
    g.add_node("predictor", predictor_node)

    # Edges: core pipeline
    g.set_entry_point("classifier")
    g.add_edge("classifier", "syllabus_analyst")
    # Analysts run after classifier; paper + teacher can run in parallel
    g.add_edge("syllabus_analyst", "paper_analyst")
    g.add_edge("syllabus_analyst", "teacher_analyst")
    g.add_edge("paper_analyst", "knowledge_base")
    g.add_edge("teacher_analyst", "knowledge_base")
    g.add_edge("knowledge_base", "predictor")

    # Optional generator nodes
    generator_nodes = []

    if "mock_paper" in outputs_wanted or "model_answers" in outputs_wanted:
        g.add_node("mock_paper", mock_paper_node)
        g.add_edge("predictor", "mock_paper")
        generator_nodes.append("mock_paper")

        g.add_node("examiner", examiner_node)
        g.add_edge("mock_paper", "examiner")
        generator_nodes.append("examiner")

    if "model_answers" in outputs_wanted:
        g.add_node("model_answers", model_answers_node)
        if "examiner" in generator_nodes:
            g.add_edge("examiner", "model_answers")
        else:
            g.add_edge("mock_paper", "model_answers")
        generator_nodes.append("model_answers")

    if "question_bank" in outputs_wanted:
        g.add_node("question_bank", question_bank_node)
        g.add_edge("predictor", "question_bank")
        generator_nodes.append("question_bank")

    if "flashcards" in outputs_wanted or "formula_sheet" in outputs_wanted:
        g.add_node("flashcards", flashcards_node)
        g.add_edge("predictor", "flashcards")
        generator_nodes.append("flashcards")

    if "study_plan" in outputs_wanted:
        g.add_node("study_plan", study_plan_node)
        g.add_edge("predictor", "study_plan")
        generator_nodes.append("study_plan")

    # Terminal edges – only leaf nodes (not sources of other edges) get END
    # Build set of nodes that are sources of an explicit edge
    source_nodes = set()
    if "mock_paper" in outputs_wanted or "model_answers" in outputs_wanted:
        source_nodes.add("mock_paper")
        source_nodes.add("examiner")
    if "model_answers" in outputs_wanted:
        source_nodes.add("examiner")

    # If no generators, end after predictor
    if not generator_nodes:
        g.add_edge("predictor", END)
    else:
        # Leaf nodes are those in generator_nodes that are not sources
        for node in generator_nodes:
            if node not in source_nodes:
                g.add_edge(node, END)

    return g.compile()


def run_workflow(state: Dict[str, Any]) -> Dict[str, Any]:
    """Entry point: compile graph and stream execution."""
    outputs_wanted = state.get("outputs_wanted", [])
    graph = build_graph(outputs_wanted)

    final_state = dict(state)
    for chunk in graph.stream(state, {"recursion_limit": 50}):
        for node_name, node_output in chunk.items():
            if isinstance(node_output, dict):
                final_state.update(node_output)
            logger.info(f"✓ Node '{node_name}' complete")

    return final_state
