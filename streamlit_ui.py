import streamlit as st
import uuid
from datetime import datetime
from typing import List, Dict, Any

from langchain_core.messages import HumanMessage

from agent import enhanced_graph, InMemoryStore, MemorySaver, RunnableConfig
from agent_nodes import post_to_linkedin, update_profile

# 
# --- Page Configuration ---
st.set_page_config(
    page_title="LinkedIn Content Creator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Session State Initialization ---
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "store" not in st.session_state:
    st.session_state.store = enhanced_graph.store
    st.session_state.saver = MemorySaver()
    st.session_state.status_log: List[Dict[str, Any]] = []
    st.session_state.workflow_data: Dict[str, Any] = {}

graph = enhanced_graph

def get_config() -> RunnableConfig:
    return {
        "configurable": {
            "thread_id": st.session_state.thread_id,
            "user_id": st.session_state.user_id
        }
    }

def add_to_log(message: str, msg_type: str = "info"):
    st.session_state.status_log.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "text": message,
        "type": msg_type
    })

def display_log():
    st.subheader("📋 Workflow Status")
    log = st.session_state.status_log
    if not log:
        st.info("No activity yet.")
        return
    mapping = {
        "info":    st.info,
        "success": st.success,
        "warning": st.warning,
        "error":   st.error
    }
    for entry in reversed(log[-10:]):
        fn = mapping.get(entry["type"], st.info)
        fn(f"[{entry['time']}] {entry['text']}")

def stream_workflow(messages: List[HumanMessage], action: str):
    cfg = get_config()
    try:
        add_to_log(f"▶️ {action} started")
        with st.spinner(f"{action}..."):
            prog = st.progress(0)
            step = 0
            for chunk in graph.stream({"messages": messages}, cfg, stream_mode="values"):
                step += 1
                prog.progress(min(step * 10, 100))
                for m in chunk.get("messages", []):
                    add_to_log(m.content)
                for key in [
                    "temporary_topics", "final_topics", "selected_topic",
                    "fetched_articles", "good_articles", "competitor_insights",
                    "content_draft", "optimized_content", "approved_for_posting",
                    "posted_content_id"
                ]:
                    if key in chunk:
                        st.session_state.workflow_data[key] = chunk[key]
                        add_to_log(f"{key} updated", "success")
            prog.progress(100)
        add_to_log(f"✅ {action} completed", "success")
    except Exception as e:
        add_to_log(f"❌ {action} error: {e}", "error")
        st.error(f"{action} failed: {e}")
def reset_topics():
    # 1. Clear UI workflow data
    st.session_state.workflow_data.clear()
    add_to_log("🔄 Workflow state reset", "info")

    # 2. Delete topic memories
    topic_ns = ("topic", st.session_state.user_id)
    for mem in st.session_state.store.search(topic_ns):
        st.session_state.store.delete(topic_ns, mem.key)
    add_to_log("🗑️ Cleared topic memory", "info")
# --- Login Screen ---
if st.session_state.user_id is None:
    st.title("🔑 Enter Your User ID")
    uid = st.text_input("User ID", "")
    if st.button("Submit"):
        if uid.strip():
            st.session_state.user_id = uid.strip()
            st.rerun()
        else:
            st.error("User ID cannot be empty")
    st.stop()

# --- Sidebar Controls ---
st.sidebar.header(f"Session {st.session_state.thread_id[:8]}… | User: {st.session_state.user_id}")
st.sidebar.markdown("### 👤 Profile & Actions")

# # Profile Input
# profile_text = st.sidebar.text_area(
#     "Paste your full LinkedIn profile",
#     placeholder="Name, role, company, background, interests..."
# )
# if st.sidebar.button("🔄 Update Profile", use_container_width=True):
#     if profile_text.strip():
#         stream_workflow([HumanMessage(content=profile_text)], "Profile Update")
#     else:
#         st.sidebar.warning("Please paste your profile first.")

# st.sidebar.divider()


profile_text = st.sidebar.text_area(
    "Paste your full LinkedIn profile",
    placeholder="Name, role, company, background, interests..."
)

if st.sidebar.button("🔄 Update Profile", use_container_width=True):
    if profile_text.strip():
        add_to_log("▶️ Profile Update started")
        with st.spinner("Updating profile…"):
            
            # Prepare the input for the graph
            profile_input = [HumanMessage(content=profile_text)]
            config = get_config()

            # Stream the graph directly
            for chunk in enhanced_graph.stream(
                {"messages": profile_input},
                config,
                stream_mode="values"
            ):
                # Log any messages from LLM/tool nodes
                for m in chunk.get("messages", []):
                    
                    add_to_log(m.content)
                # Update UI state with any new profile schema or other fields
                for key, val in chunk.items():
                    if key != "messages":
                        st.session_state.workflow_data[key] = val
                        #add_to_log(f"{key} updated", "success")
                # Immediately after your graph stream for profile update, dump the number of stored items
                
        profile_ns = ("profile", st.session_state.user_id)
        count = len(st.session_state.store.search(profile_ns))
        add_to_log(f"Profile entries in store: {count}", "info")
        st.sidebar.success("Profile schema updated via graph")
    else:
        st.sidebar.warning("Please paste your profile first.")


# Content Generation
if st.sidebar.button("📝 Generate Topics & Create Content", use_container_width=True):
    stream_workflow(
        [HumanMessage(content="Generate topics and create LinkedIn content with article research")],
        "Content Generation"
    )

st.sidebar.divider()

# Posting / Rejection
ready = "optimized_content" in st.session_state.workflow_data
user_id = st.session_state.user_id
store = st.session_state.store

if st.sidebar.button("✅ Approve & Post to LinkedIn", use_container_width=True, disabled=not ready):
    add_to_log("▶️ Posting to LinkedIn")
    result = post_to_linkedin(
        state=st.session_state.workflow_data,
        config=get_config(),
        store=store
    )
    for msg in result.get("messages", []):
        add_to_log(msg, "success" if msg.startswith("✅") else "error")
    if "posted_content_id" in result:
        st.sidebar.success(f"Posted! ID: {result['posted_content_id']}")
    else:
        st.sidebar.error("Posting failed.")
    # Clear UI state
    st.session_state.workflow_data.clear()
    reset_topics()
    st.rerun()
    #add_to_log("🔄 Workflow state reset after posting", "info")
    # Clear topics memory
    # topic_ns = ("topic", user_id)
    # for mem in store.search(topic_ns):
    #     store.delete(topic_ns, mem.key)
    # add_to_log("🗑️ Cleared topics memory", "info")

if ready and st.sidebar.button("❌ Reject Content", use_container_width=True):
    add_to_log("❌ Content rejected by user", "warning")
    st.sidebar.warning("Content workflow has been reset.")
    reset_topics()
    st.rerun()
    # Clear UI state
    # st.session_state.workflow_data.clear()
    # add_to_log("🔄 Workflow state reset after rejection", "info")
    # # Clear topics memory
    # topic_ns = ("topic", user_id)
    # for mem in store.search(topic_ns):
    #     store.delete(topic_ns, mem.key)
    # add_to_log("🗑️ Cleared topics memory", "info")

# --- Main Layout with Tabs ---
st.title("🚀 LinkedIn Content Creator")
tabs = st.tabs(["Dashboard", "Stored Profile"])

with tabs[0]:
    col1, col2 = st.columns([2, 1])
    with col1:
        display_log()
        if st.session_state.workflow_data:
            st.divider()
            st.subheader("📊 Workflow Results")
            data = st.session_state.workflow_data

            if topics := data.get("final_topics"):
                with st.expander("🎯 Generated Topics", expanded=True):
                    for i, t in enumerate(topics, 1):
                        st.write(f"{i}. {t}")

            if sel := data.get("selected_topic"):
                st.success(f"Selected Topic: **{sel}**")

            if arts := data.get("fetched_articles"):
                with st.expander("📰 Articles Fetched", expanded=False):
                    st.write(f"Total: {len(arts)}")
                    for a in arts:
                        st.write(f"- {getattr(a, 'title', '<no title>')}")
                    good = data.get("good_articles", [])
                    st.write(f"👍 Good: {len(good)} / {len(arts)}")

            if draft := data.get("content_draft"):
                with st.expander("✍️ Draft Content", expanded=True):
                    st.text_area("Draft", draft, height=200, disabled=True)

            if opt := data.get("optimized_content"):
                with st.expander("✨ Optimized Content", expanded=True):
                    st.text_area("Ready to Post", opt, height=250, disabled=True)

    with col2:
        st.subheader("📈 Analytics")
        steps = [
            ("Profile Updated",   "final_topics" in st.session_state.workflow_data),
            ("Topics Generated",  "final_topics" in st.session_state.workflow_data),
            ("Topic Selected",    "selected_topic" in st.session_state.workflow_data),
            ("Articles Fetched",  "fetched_articles" in st.session_state.workflow_data),
            ("Content Created",   "content_draft" in st.session_state.workflow_data),
            ("Content Optimized", "optimized_content" in st.session_state.workflow_data),
            ("Ready to Post",     "optimized_content" in st.session_state.workflow_data)
        ]
        completed = sum(1 for _, cond in steps if cond)
        st.metric("Progress", f"{completed}/{len(steps)}")
        st.progress(completed / len(steps))
        for name, cond in steps:
            (st.success if cond else st.info)(f"{'✅' if cond else '⏳'} {name}")

        if arts := st.session_state.workflow_data.get("fetched_articles"):
            total, good = len(arts), len(st.session_state.workflow_data.get("good_articles", []))
            rate = f"{good/total*100:.1f}%" if total else "N/A"
            st.metric("Articles Analyzed", total)
            st.metric("High-Quality", good)
            st.metric("Quality Rate", rate)

with tabs[1]:
    st.subheader("👤 Stored Profile Data")
    namespace = ("profile", st.session_state.user_id)
    mems = st.session_state.store.search(namespace)
    if not mems:
        st.info("No profile stored yet.")
    else:
        for mem in mems:
            st.markdown(f"**Key:** `{mem.key}`")
            st.json(mem.value)

st.divider()
with st.expander("🔍 Debug Info"):
    st.json({
        "thread_id": st.session_state.thread_id,
        "user_id": st.session_state.user_id,
        "workflow_keys": list(st.session_state.workflow_data.keys()),
        "log_count": len(st.session_state.status_log),
        "config": get_config()
    })
