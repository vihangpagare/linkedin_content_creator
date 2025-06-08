"""
Build and compile the LangGraph workflow using all nodes defined in agent_nodes.py.
Exports `enhanced_graph`, `InMemoryStore`, `MemorySaver`, and `RunnableConfig` for downstream use.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.runnables import RunnableConfig

from agent_nodes import (
    IntegratedContentState,
    master_node,
    update_profile,
    generate_topic_integrated,
    update_topic,
    select_single_topic,
    fetch_articles_for_topic,
    evaluate_articles,
    analyze_competitor_content,
    create_linkedin_content_with_articles,
    optimize_linkedin_content,
    post_to_linkedin,
    route_message,
    route_after_topic_generation,
    route_after_topic_selection_enhanced,
    route_after_article_fetching,
    route_after_article_evaluation,
    route_after_competitor_analysis,
    route_after_content_creation,
    route_after_optimization,
    route_after_approval_response,
)

# ─────── Build StateGraph ─────────────────────────────────────────────────────────
builder = StateGraph(IntegratedContentState)

# ── Add core nodes
builder.add_node("master_node", master_node)
builder.add_node("update_profile", update_profile)
builder.add_node("generate_topic", generate_topic_integrated)
builder.add_node("update_topic", update_topic)

# ── Add content workflow nodes
builder.add_node("select_single_topic", select_single_topic)
builder.add_node("fetch_articles_for_topic", fetch_articles_for_topic)
builder.add_node("evaluate_articles", evaluate_articles)
builder.add_node("analyze_competitor_content", analyze_competitor_content)
builder.add_node("create_linkedin_content_with_articles", create_linkedin_content_with_articles)
builder.add_node("optimize_linkedin_content", optimize_linkedin_content)

#builder.add_node("post_to_linkedin", post_to_linkedin)

# ─────── Define Edges & Routing ───────────────────────────────────────────────────
builder.add_edge(START, "master_node")
builder.add_conditional_edges("master_node", route_message)
builder.add_edge("update_profile", "master_node")

# Topic generation flow
builder.add_conditional_edges("generate_topic", route_after_topic_generation)
builder.add_conditional_edges("select_single_topic", route_after_topic_selection_enhanced)
builder.add_conditional_edges("fetch_articles_for_topic", route_after_article_fetching)
builder.add_conditional_edges("evaluate_articles", route_after_article_evaluation)
builder.add_conditional_edges("analyze_competitor_content", route_after_competitor_analysis)
builder.add_conditional_edges("create_linkedin_content_with_articles", route_after_content_creation)
#builder.add_conditional_edges("optimize_linkedin_content", route_after_optimization)
builder.add_edge("optimize_linkedin_content", END)
#builder.add_edge("post_to_linkedin", END)

builder.add_edge("update_topic", "master_node")

# ─────── Compile Graph ────────────────────────────────────────────────────────────
enhanced_memory = InMemoryStore()
checkpointer = MemorySaver()

enhanced_graph = builder.compile(
    checkpointer=checkpointer,
    store=enhanced_memory
)

# ─────── Export ───────────────────────────────────────────────────────────────────
__all__ = [
    "enhanced_graph",
    "InMemoryStore",
    "MemorySaver",
    "RunnableConfig"
]
