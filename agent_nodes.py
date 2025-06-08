
import os
import uuid
import json
import requests
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import List, Optional, Dict, Any, Literal

from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage, merge_message_runs
from langchain_core.runnables import RunnableConfig
from langchain.schema import AIMessage

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import interrupt
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver

from langchain_openai import AzureChatOpenAI
from exa_py import Exa
from trustcall import create_extractor

from prompts import (
    TOPIC_SELECTION_PROMPT,
    WEB_RESEARCH_PROMPT,
    COMPETITOR_CONTENT_ANALYSIS_PROMPT,
    CONTENT_OPTIMIZATION_PROMPT,
    ARTICLE_EVALUATION_PROMPT,
    ENHANCED_CONTENT_CREATION_PROMPT,
    SUMMARY_INSTRUCTION,
    TOPIC_GENERATION_INSTRUCTION,
    TRUSTCALL_INSTRUCTION,
    MODEL_SYSTEM_MESSAGE,
)


# ─────── Load environment variables ────────────────────────────────────────────────
load_dotenv()

AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
EXA_KEY = os.getenv("EXA_KEY")
LINKEDIN_ACCESS_TOKEN = os.getenv("LINKEDIN_ACCESS_TOKEN")
LINKEDIN_PERSON_URN = os.getenv("LINKEDIN_PERSON_URN")



# ─────── Instantiate the LLM model ────────────────────────────────────────────────
model = AzureChatOpenAI(
    api_version=AZURE_API_VERSION,
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    model="gpt4o",
)



# ─────── Pydantic Schemas ──────────────────────────────────────────────────────────
class Memory(BaseModel):
    content: str = Field(description="The main content of the memory.")

class MemoryCollection(BaseModel):
    memories: List[Memory]

class Profile(BaseModel):
    name: Optional[str] = Field(None, description="User's full name")
    current_work: Optional[str] = Field(None, description="Current work profile")
    academics: Optional[str] = Field(None, description="Academic background")
    previous_work: List[str] = Field(default_factory=list, description="Previous work experiences")
    company_website: Optional[str] = Field(None, description="Company website (if any)")
    why_build: Optional[str] = Field(None, description="Why the user built their project")
    known_as: Optional[str] = Field(None, description="Who the user wants to be known as")
    alternative_career: Optional[str] = Field(None, description="Alternative career choice")
    past_expertise: Optional[str] = Field(None, description="Expertise no longer pursued")
    superpower: Optional[str] = Field(None, description="User's superpower")
    entrepreneurship_journey: Optional[str] = Field(None, description="How they began entrepreneur journey")
    mentors: List[str] = Field(default_factory=list, description="List of mentors")
    linkedin_highlights: List[str] = Field(default_factory=list, description="Highlights for LinkedIn")
    target_list: List[str] = Field(default_factory=list, description="Companies/people to target")
    sample_post_links: List[str] = Field(default_factory=list, description="Links to posts they like")
    content_theme: Optional[str] = Field(None, description="Theme for content (e.g., AI for lead gen)")
    focus_topic: Optional[str] = Field(None, description="Focus topic right now")
    topics_per_week: Optional[int] = Field(None, description="Number of topics per week")
    weeks_count: Optional[int] = Field(None, description="Number of weeks planned")




# ─────── Trustcall Extractors ──────────────────────────────────────────────────────
profile_extractor = create_extractor(
    model,
    tools=[Profile],
    tool_choice="Profile",
)



# ─────── Extended Content State ───────────────────────────────────────────────────
from typing import Annotated
from operator import add

class IntegratedContentState(MessagesState):
    feedback: Annotated[List[str], add]
    confirmed: bool = False
    temporary_topics: List[List[str]] = []
    final_topics: List[str] = []
    selected_topic: str = ""
    competitor_insights: Dict[str, Any] = {}
    sentiment_data: Dict[str, Any] = {}
    content_draft: str = ""
    optimized_content: str = ""
    posted_content_id: str = ""
    evaluated_articles: List[Dict[str, Any]] = []
    fetched_articles: List[Any] = []
    good_articles: List[Dict[str, Any]] = []
    web_research_data: str = ""
    approved_for_posting: bool = False
    pending_approval: bool = False


# ─────── Node: Select Single Topic ─────────────────────────────────────────────────
def select_single_topic(state: IntegratedContentState, config: RunnableConfig, store: BaseStore):
    """Select one topic from generated topics for content creation"""
    user_id = config["configurable"]["user_id"]

    # Get user profile
    namespace = ("profile", user_id)
    profile_memories = store.search(namespace)
    user_profile = profile_memories[0].value if profile_memories else {}

    # Get generated topics from temporary_topics
    topics = state.get("final_topics", [])
    if not topics or not topics[0]:
        return {"messages": ["No topics found to select from"], "selected_topic": ""}

    topics_list = topics[0] if isinstance(topics[0], list) else [topics[0]]

    # Use LLM to select the best topic
    selection_prompt = TOPIC_SELECTION_PROMPT.format(
        topics=topics_list,
        user_profile=user_profile
    )

    selected_topic_response = model.invoke([SystemMessage(content=selection_prompt)])
    selected_topic = selected_topic_response.content.strip()

    return {
        "selected_topic": selected_topic,
        "messages": [f"Selected topic for content creation: {selected_topic}"]
    }


# ─────── Node: Analyze Competitor Content ──────────────────────────────────────────
def analyze_competitor_content(state: IntegratedContentState, config: RunnableConfig, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    topic = state.get("selected_topic", "")
    if not topic:
        return {"messages": ["No topic selected for competitor analysis"]}

    exa = Exa(api_key=EXA_KEY)
    # Web research
    try:
        research_prompt = WEB_RESEARCH_PROMPT.format(topic=topic)
        research_results = exa.search_and_contents(
            f"{topic} trending LinkedIn discussions 2025",
            num_results=10,
            start_published_date="2024-11-01T00:00:00.000Z",
            include_domains=["linkedin.com"],
            summary=True
        )
        web_research = " ".join([res.summary for res in research_results.results if res.summary])
    except Exception:
        web_research = f"Current discussions around {topic}"

    competitor_queries = [
        f'"{topic}" LinkedIn viral posts high engagement 2024 2025',
        f'"{topic}" thought leadership LinkedIn content',
        f'"{topic}" professional discussion LinkedIn'
    ]
    competitor_content = []
    today = datetime.today().date()
    prev = today - relativedelta(months=3)

    for query in competitor_queries:
        try:
            resp = exa.search_and_contents(
                query,
                num_results=25,
                use_autoprompt=True,
                start_published_date=prev.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                end_published_date=today.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                include_domains=["linkedin.com"],
                summary=True
            ).results
            competitor_content.extend(resp)
        except Exception:
            continue

    competitor_text = "\n\n".join([
        f"Title: {c.title}\nSummary: {c.summary}"
        for c in competitor_content[:20]
    ])

    analysis_prompt = COMPETITOR_CONTENT_ANALYSIS_PROMPT.format(
        topic=topic,
        competitor_content=competitor_text,
        web_research_data=web_research
    )
    analysis_response = model.invoke([SystemMessage(content=analysis_prompt)])
    try:
        insights = json.loads(analysis_response.content)
    except Exception:
        insights = {
            "high_performing_formats": ["story posts", "insight posts"],
            "viral_hooks": ["surprising statistics", "contrarian views"],
            "engagement_triggers": ["questions", "personal experience"],
            "optimal_tone": "professional yet conversational"
        }

    return {
        "competitor_insights": insights,
        "web_research_data": web_research,
        "messages": [f"Enhanced analysis completed: {len(competitor_content)} posts analyzed."]
    }

# ─────── Node: Optimize LinkedIn Content ───────────────────────────────────────────
def optimize_linkedin_content(state: IntegratedContentState, config: RunnableConfig, store: BaseStore):
    draft = state.get("content_draft", "")
    if not draft:
        return {"messages": ["No content draft found to optimize"]}

    optimization_prompt = CONTENT_OPTIMIZATION_PROMPT.format(content=draft)
    optimized_response = model.invoke([SystemMessage(content=optimization_prompt)])
    optimized_content = optimized_response.content.strip()

    return {
        "optimized_content": optimized_content,
        "messages": [f"{optimized_content}"]
    }

# ─────── Node: Fetch Articles for Topic ────────────────────────────────────────────
def fetch_articles_for_topic(state: IntegratedContentState, config: RunnableConfig, store: BaseStore):
    topic = state.get("selected_topic", "")
    if not topic:
        return {"messages": ["No topic selected for article fetching"]}

    exa = Exa(api_key=EXA_KEY)
    fetched = []
    today = datetime.today().date()
    prev = today - relativedelta(months=2)

    queries = [
        f'"{topic}" insights analysis trends',
        f'"{topic}" industry news developments',
        f'"{topic}" expert opinions research'
    ]
    for q in queries:
        try:
            resp = exa.search_and_contents(
                q,
                num_results=5,
                use_autoprompt=True,
                start_published_date=prev.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                end_published_date=today.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                summary=True
            ).results
            fetched.extend(resp)
        except Exception:
            continue

    return {
        "fetched_articles": fetched,
        "messages": [f"Fetched {len(fetched)} articles for topic: {topic}"]
    }

# ─────── Node: Evaluate Articles ──────────────────────────────────────────────────
def evaluate_articles(state: IntegratedContentState, config: RunnableConfig, store: BaseStore):
    articles_list = state.get("fetched_articles", [])
    if not articles_list:
        return {"messages": ["No articles to evaluate"]}

    evaluated = []
    good = []
    for art in articles_list:
        article_snippet = f"Title: {art.title}\nSummary: {art.summary}\nURL: {art.url}"
        eval_prompt = ARTICLE_EVALUATION_PROMPT.format(article=article_snippet)
        eval_resp = model.invoke([SystemMessage(content=eval_prompt)])
        try:
            parsed = json.loads(eval_resp.content)
            verdict = parsed.get("evaluation", "bad").lower()
        except Exception:
            verdict = "bad"

        entry = {
            "title": art.title,
            "summary": art.summary,
            "url": art.url,
            "evaluation": verdict
        }
        evaluated.append(entry)
        if verdict == "good":
            good.append(entry)

    return {
        "evaluated_articles": evaluated,
        "good_articles": good,
        "messages": [f"Evaluated {len(evaluated)} articles. Found {len(good)} good articles."]
    }

# ─────── Node: Create LinkedIn Content with Articles ──────────────────────────────
def create_linkedin_content_with_articles(state: IntegratedContentState, config: RunnableConfig, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    namespace = ("profile", user_id)
    profile_memories = store.search(namespace)
    user_profile = profile_memories[0].value if profile_memories else {}

    topic = state.get("selected_topic", "")
    competitor_insights = state.get("competitor_insights", {})
    good_articles = state.get("good_articles", [])

    if good_articles:
        article_insights = "Key insights from quality articles:\n"
        for i, a in enumerate(good_articles[:3], 1):
            article_insights += f"{i}. {a['title']}: {a['summary'][:200]}...\n"
    else:
        article_insights = "No high-quality articles found. Focus on original insights and competitor analysis."

    content_prompt = ENHANCED_CONTENT_CREATION_PROMPT.format(
        topic=topic,
        user_profile=json.dumps(user_profile),
        competitor_insights=json.dumps(competitor_insights),
        article_insights=article_insights
    )
    content_response = model.invoke([SystemMessage(content=content_prompt)])
    draft = content_response.content.strip()

    return {
        "content_draft": draft,
        "messages": [f"Created LinkedIn content incorporating {len(good_articles)} quality articles"]
    }
from typing import TypedDict, Literal
# Update memory tool
class UpdateMemory(TypedDict):
    """ Decision on what memory type to update """
    update_type: Literal['user','update_topic']
# ─────── Node: Master Node (Memory-driven) ────────────────────────────────────────
def master_node(state: IntegratedContentState, config: RunnableConfig, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    namespace = ("profile", user_id)
    mems = store.search(namespace)
    user_profile = mems[0].value if mems else None

    namespace = ("topic", user_id)
    topic_mems = store.search(namespace)
    topics = topic_mems[0].value if topic_mems else []

    system_msg = MODEL_SYSTEM_MESSAGE
    response = model.bind_tools([UpdateMemory], parallel_tool_calls=False).invoke(
        [SystemMessage(content=system_msg)] + state["messages"]
    )

    return {"messages": [response]}

# ─────── Node: Update Profile ─────────────────────────────────────────────────────
def update_profile(state: IntegratedContentState, config: RunnableConfig, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    namespace = ("profile", user_id)
    existing = store.search(namespace)
    existing_memories = (
        [(item.key, "Profile", item.value) for item in existing]
        if existing else None
    )
    TRUSTCALL_FMT = TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
    merged_msgs = list(merge_message_runs(
        messages=[SystemMessage(content=TRUSTCALL_FMT)] + state["messages"][:-1]
    ))

    result = profile_extractor.invoke({
        "messages": merged_msgs,
        "existing": existing_memories
    })
    for r, meta in zip(result["responses"], result["response_metadata"]):
        store.put(
            namespace,
            meta.get("json_doc_id", str(uuid.uuid4())),
            r.model_dump(mode="json")
        )

    tool_calls = state["messages"][-1].tool_calls
    return {"messages": [{"role": "tool", "content": "updated profile", "tool_call_id": tool_calls[0]["id"]}]}

# ─────── Node: Update Topic ───────────────────────────────────────────────────────
def update_topic(state: IntegratedContentState, config: RunnableConfig, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    namespace = ("topic", user_id)
    existing = store.search(namespace)
    existing_topics = (
        [(item.key, "topic", item.value) for item in existing]
        if existing else None
    )
    TRUSTCALL_FMT = TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
    merged_msgs = list(merge_message_runs(
        messages=[SystemMessage(content=TRUSTCALL_FMT)] + state["messages"][:-1]
    ))

    result = topic_extractor.invoke({
        "messages": merged_msgs,
        "existing": existing_topics
    })
    for r, meta in zip(result["responses"], result["response_metadata"]):
        store.put(
            namespace,
            meta.get("json_doc_id", str(uuid.uuid4())),
            r.model_dump(mode="json")
        )

    tool_calls = state["messages"][-1].tool_calls
    return {"messages": [{"role": "tool", "content": "updated topics list", "tool_call_id": tool_calls[0]["id"]}]}

# ─────── Node: Post to LinkedIn ───────────────────────────────────────────────────
def post_to_linkedin(state: IntegratedContentState, config: RunnableConfig, store: BaseStore):
    optimized_content = state.get("optimized_content", "")
    
    try:
        # LinkedIn API credentials and settings
        ACCESS_TOKEN = LINKEDIN_ACCESS_TOKEN
        AUTHOR_URN = 'urn:li:person:4UPDA8Ukrx'
        
        url = 'https://api.linkedin.com/v2/ugcPosts'

        post_data = {
            "author": AUTHOR_URN,
            "lifecycleState": "PUBLISHED",
            "specificContent": {
                "com.linkedin.ugc.ShareContent": {
                    "shareCommentary": {
                        "text": optimized_content
                    },
                    "shareMediaCategory": "NONE"
                }
            },
            "visibility": {
                "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
            }
        }

        headers = {
            'Authorization': f'Bearer {ACCESS_TOKEN}',
            'Content-Type': 'application/json',
            'X-Restli-Protocol-Version': '2.0.0'
        }

        response = requests.post(url, headers=headers, json=post_data)
        
        if response.status_code == 201:
            return {
                "posted_content_id": response.headers.get('x-restli-id', 'unknown'),
                "messages": [f"✅ Successfully posted to LinkedIn!\n\nContent:\n{optimized_content}"]
            }
        else:
            return {"messages": [f"❌ Failed to post to LinkedIn. Error: {response.status_code} - {response.text}"]}

    except Exception as e:
        return {"messages": [f"❌ Error posting to LinkedIn: {str(e)}"]}

    

# ─────── Helper: Extract Topics from LLM Output ───────────────────────────────────
import re
def extract_topics(llm_output: str) -> List[str]:
    match = re.search(r"<topics>(.*?)</topics>", llm_output, re.DOTALL)
    if not match:
        return []
    inner = match.group(1).strip()
    lines = [line.strip("- ").strip() for line in inner.split("\n") if line.strip().startswith("-")]
    return lines



# ─────── Node: Generate Topics ────────────────────────────────────────────────────
def generate_topic_integrated(state: IntegratedContentState, config: RunnableConfig, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    # Fetch existing profile summary
    namespace = ("profile", user_id)
    existing_profile = store.search(namespace)
    user_profile = [(item.key, "Profile", item.value) for item in existing_profile] if existing_profile else None

    # Fetch existing topics
    namespace = ("topic", user_id)
    existing_top = store.search(namespace)
    existing_topics = existing_top[0].value if existing_top else []

    # Step 1: Generate profile summary
    summary_msg = SUMMARY_INSTRUCTION.format(user_profile=json.dumps(user_profile) if user_profile else "{}")
    summary_response = model.invoke([SystemMessage(content=summary_msg)])
    summary_paragraph = summary_response.content.strip()

    # Step 2: Generate new topics
    gen_msg = TOPIC_GENERATION_INSTRUCTION.format(
        summary_paragraph=summary_paragraph,
        topics=json.dumps(existing_topics),
        feedback=""
    )
    list_response = model.invoke([SystemMessage(content=gen_msg)])
    parsed = extract_topics(list_response.content)
    if not parsed:
        parsed = [list_response.content.strip()]

    
    return {
        "temporary_topics": [parsed],
        "final_topics": parsed,
        "messages": [f"Generated topics for content creation: {parsed}"]
    }

# ─────── Routing Functions ─────────────────────────────────────────────────────────
def route_after_topic_generation(state: IntegratedContentState) -> Literal["select_single_topic", END]:
    if state.get("final_topics"):
        return "select_single_topic"
    return END

def route_after_topic_selection_enhanced(state: IntegratedContentState) -> Literal["fetch_articles_for_topic", END]:
    if state.get("selected_topic"):
        return "fetch_articles_for_topic"
    return END

def route_after_article_fetching(state: IntegratedContentState) -> Literal["evaluate_articles", END]:
    if state.get("fetched_articles"):
        return "evaluate_articles"
    return END

def route_after_article_evaluation(state: IntegratedContentState) -> Literal["analyze_competitor_content"]:
    return "analyze_competitor_content"

def route_after_competitor_analysis(state: IntegratedContentState) -> Literal["create_linkedin_content_with_articles"]:
    return "create_linkedin_content_with_articles"

def route_after_content_creation(state: IntegratedContentState) -> Literal["optimize_linkedin_content"]:
    return "optimize_linkedin_content"

def route_after_optimization(state: IntegratedContentState) -> Literal["process_approval_response"]:
    return "process_approval_response"

def route_after_approval_response(state: IntegratedContentState) -> Literal["post_to_linkedin", END]:
    if state.get("approved_for_posting"):
        return "post_to_linkedin"
    return END

def route_message(state: IntegratedContentState, config: RunnableConfig, store: BaseStore) -> Literal[END, "update_profile", "generate_topic", "update_topic"]:
    msg = state["messages"][-1]
    if hasattr(msg, "content"):
        content = msg.content.lower()
        if "topic" in content:
            return "generate_topic"
        if content.strip() == "updated profile":
            return END
        if content.strip() == "updated topics list":
            return END
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        call = msg.tool_calls[0]
        if call["args"]["update_type"] == "user":
            return "update_profile"
        elif call["args"]["update_type"] == "update_topic":
            return "update_topic"
    return END
