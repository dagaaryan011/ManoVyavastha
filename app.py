"""
================================================================================
FILE: app.py
COMPONENT: Streamlit Web App - Cognitive-Aware Task Scheduler
================================================================================
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import os
import plotly.graph_objects as go
import plotly.express as px  # <--- ADD THIS IMPORT
from typing import Dict, List

from llm_task_decomposer import LLMTaskDecomposer
from smart_scheduler import SmartScheduler
from rl_agent_decomposed import (
    TimeSchedulingAgent,
    DurationAllocationAgent,
    BreakSchedulingAgent
)
from user_profile_manager import UserProfileManager


AGENT_SAVE_DIR = "saved_agents"
PROFILE_SAVE_PATH = "user_profile.json"

# ============================================================
# VISUAL STYLING (MICROSOFT TEAMS STYLE)
# ============================================================

def load_custom_css():
    st.markdown("""
    <style>
        /* Light Theme Teams Card Styling */
        .teams-card {
            background-color: #FFFFFF; /* White Card */
            border-left: 6px solid #6264A7;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1); /* Softer shadow */
            display: flex;
            align-items: center;
            justify-content: space-between;
            border: 1px solid #E0E0E0; /* Light gray border */
        }
        .time-col {
            font-family: 'Source Sans Pro', sans-serif;
            font-weight: 700;
            color: #333333; /* Dark text */
            min-width: 90px;
            border-right: 2px solid #F0F0F0;
            padding-right: 12px;
            margin-right: 16px;
            text-align: right;
            font-size: 1.1rem;
        }
        .info-col {
            flex-grow: 1;
        }
        .task-title {
            font-size: 18px;
            font-weight: 600;
            color: #111111; /* Nearly black for title */
            margin-bottom: 4px;
        }
        .task-meta {
            font-size: 13px;
            color: #666666; /* Medium gray for meta */
            display: flex;
            gap: 12px;
            align-items: center;
        }
        .tag {
            background: #F3F3F3; /* Light gray tag bg */
            color: #444; /* Dark tag text */
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border: 1px solid #DDD;
        }
        /* Dynamic Border Colors */
        .border-green { border-left-color: #2ECC71 !important; }
        .border-yellow { border-left-color: #F1C40F !important; }
        .border-red { border-left-color: #E74C3C !important; }
        .border-break { border-left-color: #3498DB !important; border-style: dashed; background-color: #F8F9FA; }
    </style>
    """, unsafe_allow_html=True)

def render_fatigue_gauge(fatigue_level):
    """
    Renders a gauge chart showing fatigue level.
    Red = High Fatigue (Bad), Green = Low Fatigue (Good).
    """
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = fatigue_level,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Mental Fatigue", 'font': {'size': 24, 'color': "black"}}, # Changed to black
        number = {'suffix': "%", 'font': {'color': "black"}}, # Changed to black
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "black"}, # Changed to black
            'bar': {'color': "rgba(0,0,0,0)"}, 
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#333",
            'steps': [
                {'range': [0, 40], 'color': "#2ECC71"},
                {'range': [40, 70], 'color': "#F1C40F"},
                {'range': [70, 100], 'color': "#E74C3C"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4}, # Changed to black
                'thickness': 0.75,
                'value': fatigue_level
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "black", 'family': "Arial"}, # Global font color black
        margin=dict(l=20, r=20, t=50, b=20),
        height=250
    )
    
    st.plotly_chart(fig, use_container_width=True)
def render_teams_task_card(item):
    """Renders the HTML card for a task/break."""
    if item.type == "break":
        border_class = "border-break"
        icon = ""
        load_badge = "RECOVERY"
    elif item.cognitive_load >= 8:
        border_class = "border-red"
        icon = ""
        load_badge = f"HIGH LOAD ({item.cognitive_load})"
    elif item.cognitive_load >= 5:
        border_class = "border-yellow"
        icon = ""
        load_badge = f"MED LOAD ({item.cognitive_load})"
    else:
        border_class = "border-green"
        icon = ""
        load_badge = f"LOW LOAD ({item.cognitive_load})"

    start_str = item.start_time.strftime("%H:%M")
    end_str = item.end_time.strftime("%H:%M")
    
    st.markdown(f"""
    <div class="teams-card {border_class}">
        <div class="time-col">
            <div>{start_str}</div>
            <div style="font-weight:400; font-size:14px; opacity:0.7;">{end_str}</div>
        </div>
        <div class="info-col">
            <div class="task-title">{icon} {item.task_name}</div>
            <div class="task-meta">
                <span class="tag">{load_badge}</span>
                <span>⏱️ {item.duration_minutes()} min</span>
                <span style="font-style:italic; opacity:0.7; border-left: 1px solid #555; padding-left: 8px;">{item.reasoning}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# SESSION INITIALIZATION
# ============================================================

def initialize_session_state():
    if "initialized" not in st.session_state:
        os.makedirs(AGENT_SAVE_DIR, exist_ok=True)

        # Load or create agents ONLY ONCE
        time_agent = TimeSchedulingAgent()
        duration_agent = DurationAllocationAgent()
        break_agent = BreakSchedulingAgent()
        user_profile = UserProfileManager()

        time_agent.load(os.path.join(AGENT_SAVE_DIR, "time_agent.pkl"))
        duration_agent.load(os.path.join(AGENT_SAVE_DIR, "duration_agent.pkl"))
        break_agent.load(os.path.join(AGENT_SAVE_DIR, "break_agent.pkl"))
        user_profile.load_profile(PROFILE_SAVE_PATH)

        st.session_state.time_agent = time_agent
        st.session_state.duration_agent = duration_agent
        st.session_state.break_agent = break_agent
        st.session_state.user_profile = user_profile

        st.session_state.scheduler = SmartScheduler(
            time_agent,
            duration_agent,
            break_agent,
            user_profile
        )

        st.session_state.multi_day_schedule = {}
        st.session_state.sub_tasks = []
        st.session_state.history_log = []
        st.session_state.reward_log = []
        st.session_state.attempt_count = {}
        st.session_state.fatigue_log = [0]

        st.session_state.initialized = True


def save_agents():
    st.session_state.time_agent.save(os.path.join(AGENT_SAVE_DIR, "time_agent.pkl"))
    st.session_state.duration_agent.save(os.path.join(AGENT_SAVE_DIR, "duration_agent.pkl"))
    st.session_state.break_agent.save(os.path.join(AGENT_SAVE_DIR, "break_agent.pkl"))
    st.session_state.user_profile.save_profile(PROFILE_SAVE_PATH)


# ============================================================
# LOGIC HANDLERS (UNCHANGED)
# ============================================================

def process_success(task, actual_time, scheduled_duration):
    time_agent = st.session_state.time_agent
    duration_agent = st.session_state.duration_agent
    break_agent = st.session_state.break_agent
    user_profile = st.session_state.user_profile

    reward = 10.0
    if actual_time < scheduled_duration * 0.9:
        reward += 5.0
    elif actual_time > scheduled_duration * 1.1:
        reward -= 2.0

    update_result = user_profile.update_on_completion(
        task, actual_time, datetime.now().hour
    )

    task_type = task['task_type']
    difficulty = task['difficulty']
    cognitive_load = task['cognitive_load']
    current_hour = datetime.now().hour
    fatigue_before = st.session_state.fatigue_log[-1]

    time_state = time_agent.get_state(current_hour, task_type, difficulty)
    time_action = time_agent.get_action(time_state)

    eff = user_profile.profile.get(f"{task_type}_multiplier", 1.0)
    duration_state = duration_agent.get_state(task_type, difficulty, eff)
    duration_action = duration_agent.get_action(duration_state)

    break_state = break_agent.get_state(fatigue_before, cognitive_load, 0)
    break_action = break_agent.get_action(break_state)

    fatigue_after = min(100, fatigue_before + cognitive_load * 3.5)
    next_hour = (current_hour + actual_time // 60) % 24

    time_agent.update_q_value(time_state, time_action, reward, time_agent.get_state(next_hour, task_type, difficulty))
    duration_agent.update_q_value(duration_state, duration_action, reward, duration_agent.get_state(task_type, difficulty, eff))
    break_agent.update_q_value(break_state, break_action, reward, break_agent.get_state(fatigue_after, cognitive_load, 0))

    st.session_state.fatigue_log.append(fatigue_after)
    st.session_state.history_log.append({
        "timestamp": datetime.now(), "task_name": task["task_name"], "task_type": task_type,
        "outcome": "completed", "actual_time": actual_time, "reward": reward, "fatigue": fatigue_after
    })
    st.session_state.reward_log.append(reward)
    save_agents()
    return reward


def process_failure(task, feedback_list, actual_time, scheduled_duration):
    time_agent = st.session_state.time_agent
    duration_agent = st.session_state.duration_agent
    break_agent = st.session_state.break_agent
    user_profile = st.session_state.user_profile

    task_id = task["task_id"]
    attempts = st.session_state.attempt_count.get(task_id, 0)
    penalty = -15.0 * (0.3 if attempts == 0 else 0.7)
    st.session_state.attempt_count[task_id] = attempts + 1

    task_type = task["task_type"]
    difficulty = task["difficulty"]
    cognitive_load = task["cognitive_load"]
    hour = datetime.now().hour
    fatigue_before = st.session_state.fatigue_log[-1]

    time_state = time_agent.get_state(hour, task_type, difficulty)
    time_action = time_agent.get_action(time_state)

    eff = user_profile.profile.get(f"{task_type}_multiplier", 1.0)
    duration_state = duration_agent.get_state(task_type, difficulty, eff)
    duration_action = duration_agent.get_action(duration_state)

    break_state = break_agent.get_state(fatigue_before, cognitive_load, 0)
    break_action = break_agent.get_action(break_state)

    fatigue_after = min(100, fatigue_before + cognitive_load * 6.0)

    if "time_too_less" in feedback_list:
        duration_agent.update_q_value(duration_state, duration_action, penalty - 10.0, duration_state)
    if "too_tired" in feedback_list:
        break_agent.update_q_value(break_state, break_action, penalty - 20.0, break_state)
    if "wrong_time" in feedback_list:
        time_agent.update_q_value(time_state, time_action, penalty - 12.0, time_state)

    user_profile.update_on_failure(task, feedback_list, actual_time, hour)

    st.session_state.fatigue_log.append(fatigue_after)
    st.session_state.history_log.append({
        "timestamp": datetime.now(), "task_name": task["task_name"], "task_type": task_type,
        "outcome": "failed", "actual_time": actual_time, "reward": penalty, "fatigue": fatigue_after
    })
    st.session_state.reward_log.append(penalty)
    save_agents()
    return penalty


# ============================================================
# MAIN
# ============================================================

def main():
    st.set_page_config(layout="wide", page_title="Cognitive Scheduler")
    load_custom_css()  # Inject CSS
    initialize_session_state()

    scheduler = st.session_state.scheduler
    llm = LLMTaskDecomposer()

    st.title("Cognitive-Aware Task Scheduler")

    tabs = st.tabs(["Plan", "My Day", " Analytics", " History"])
    with st.sidebar:
        st.header("Cognitive Status")
        
        # Get current fatigue
        current_fatigue = st.session_state.fatigue_log[-1] if st.session_state.fatigue_log else 0
        
        # Render the Gauge
        render_fatigue_gauge(current_fatigue)
        
        # Dynamic advice based on fatigue
        if current_fatigue > 70:
            st.error("High Fatigue! Take a break.")
        elif current_fatigue > 40:
            st.warning("⚡ Energy dipping. Plan lighter tasks.")
        else:
            st.success("Brain is fresh. Tackle hard tasks!")

        st.divider()
        st.metric("Tasks Completed", sum(1 for h in st.session_state.history_log if h['outcome'] == 'completed'))
        
        if st.button("Save Progress"):
            save_agents() # (Using your wrapper function)
            st.success("Saved!")
    # ------------------ PLAN ------------------
    with tabs[0]:
        goal = st.text_area("New Goal")
        days = st.number_input("Deadline", 1, 30, 3)

        if st.button("Generate", type="primary"):
            with st.spinner("Analyzing..."):
                result = llm.decompose_goal(goal, days)
                
                # Multi-goal support: Extend instead of overwrite
                # If you prefer strict overwrite like your snippet, change .extend to =
                if not st.session_state.sub_tasks:
                     st.session_state.sub_tasks = []
                st.session_state.sub_tasks.extend(result["sub_tasks"])

                st.session_state.multi_day_schedule = scheduler.generate_multi_day_schedule(
                    st.session_state.sub_tasks,
                    deadline_days=days,
                    daily_capacity_limit=25
                )
                st.success("Schedule Generated")

    # ------------------ MY DAY ------------------
    with tabs[1]:
        schedule_map = st.session_state.multi_day_schedule

        if not schedule_map:
            st.info("Generate schedule first.")
        else:
            # Date Picker
            dates = sorted(schedule_map.keys())
            labels = [d.strftime("%a %b %d") for d in dates]
            sel = st.radio("Day", labels, horizontal=True, label_visibility="collapsed")
            sel_date = dates[labels.index(sel)]
            
            st.divider()

            for i, item in enumerate(schedule_map[sel_date]["schedule"]):
                # VISUAL: Use the Teams Card
                render_teams_task_card(item)

                # LOGIC: Keep your exact logic, just nested below the card
                if item.type == "task":
                    # Layout column to align buttons nicely
                    _, act_col = st.columns([0.05, 0.95])
                    with act_col:
                        status = st.radio(
                            "Status",
                            ["Pending", "Complete", "Failed"],
                            key=f"{sel_date}_{i}",
                            horizontal=True,
                            label_visibility="collapsed"
                        )

                        if status == "Complete":
                            if st.button("Confirm Complete", key=f"c_{i}"):
                                # Safe look-up using ID
                                task = next((t for t in st.session_state.sub_tasks if t["task_id"] == item.task_id), None)
                                if task:
                                    process_success(task, item.duration_minutes(), item.duration_minutes())
                                    st.success("Updated")
                                    st.rerun()

                        if status == "Failed":
                            with st.expander("Feedback", expanded=True):
                                reasons = st.multiselect(
                                    "Why?",
                                    ["time_too_less", "too_tired", "wrong_time"],
                                    key=f"f_{i}"
                                )
                                if st.button("Submit Failure", key=f"ff_{i}"):
                                    task = next((t for t in st.session_state.sub_tasks if t["task_id"] == item.task_id), None)
                                    if task:
                                        process_failure(task, reasons, item.duration_minutes(), item.duration_minutes())
                                        st.error("Failure Logged")
                                        st.rerun()

    # ------------------ ANALYTICS ------------------
    # TAB 3: Performance Analytics
    # ------------------ ANALYTICS ------------------
    # TAB 3: Performance Analytics
    with tabs[2]:
        st.header("Performance Analytics & Learning Progress")
        
        # FIX 1: Use 'history_log', not 'task_history'
        if len(st.session_state.history_log) == 0:
            st.info("""
            **No data yet!** Complete some tasks to see your learning analytics:
            1. Generate a schedule in Tab 1
            2. Mark tasks as complete/failed in Tab 2
            3. Come back here to see how the system learns YOUR cognitive patterns!
            """)
        else:
            # Convert task history to DataFrame for easy analysis
            df_history = pd.DataFrame(st.session_state.history_log)
            
            # Calculate key metrics
            total_tasks = len(df_history)
            completed_tasks = len(df_history[df_history['outcome'] == 'completed'])
            failed_tasks = len(df_history[df_history['outcome'] == 'failed'])
            completion_rate = completed_tasks / total_tasks if total_tasks > 0 else 0
            
            # FIX 2: Manually construct stats (since 'agent' wrapper doesn't exist)
            stats = {
                'time_agent_states': len(st.session_state.time_agent.q_table),
                'duration_agent_states': len(st.session_state.duration_agent.q_table),
                'break_agent_states': len(st.session_state.break_agent.q_table)
            }
            
            # ================================================================
            # SECTION 1: OVERVIEW METRICS
            # ================================================================
            st.subheader("system Learning Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Tasks Attempted", total_tasks)
            
            with col2:
                st.metric("Completion Rate", f"{completion_rate:.0%}")
            
            with col3:
                total_q_states = (stats['time_agent_states'] + 
                                 stats['duration_agent_states'] + 
                                 stats['break_agent_states'])
                st.metric("Q-States Learned", total_q_states)
            
            with col4:
                # Calculate schedule accuracy
                completed_df = df_history[df_history['outcome'] == 'completed']
                if len(completed_df) > 0:
                    errors = []
                    for _, task in completed_df.iterrows():
                        # Handle cases where estimated_duration might be missing in older logs
                        est = task.get('actual_time', 60) # Fallback to actual if missing
                        # In your history log you save 'actual_time', not 'actual_duration'
                        act = task['actual_time']
                        
                        # Note: In process_success you saved 'actual_time' but didn't explicitly save 'estimated_duration'
                        # We will approximate accuracy based on reward for now
                        if task['reward'] > 12: error = 0 # Early bonus
                        elif task['reward'] < 9: error = 20 # Late penalty
                        else: error = 5
                        
                        errors.append(error)
                    avg_error = sum(errors) / len(errors)
                    accuracy = max(0, 100 - avg_error)
                else:
                    accuracy = 0
                
                st.metric("Schedule Accuracy", f"{accuracy:.0f}%")
            
            st.markdown("---")
            
            # ================================================================
            # SECTION 2: DECOMPOSED RL AGENTS DASHBOARD
            # ================================================================
            st.subheader("Decomposed RL Agents")
            
            col_time, col_duration, col_break = st.columns(3)
            
            # FIX 3: Access agents directly from session_state
            with col_time:
                st.markdown("#### Time Agent")
                st.metric("States Learned", stats['time_agent_states'])
                st.info("**Learns:** WHEN to schedule tasks")
            
            with col_duration:
                st.markdown("####  Duration Agent")
                st.metric("States Learned", stats['duration_agent_states'])
                st.info("**Learns:** HOW MUCH time you need")
            
            with col_break:
                st.markdown("#### Break Agent")
                st.metric("States Learned", stats['break_agent_states'])
                st.info("**Learns:** WHEN you need breaks")
            
            st.markdown("---")
            
            # ================================================================
            # SECTION 3: COGNITIVE FINGERPRINT
            # ================================================================
            st.subheader(" Your Cognitive Fingerprint")
            
            # FIX 4: Use 'user_profile', not 'profile_manager'
            profile = st.session_state.user_profile.profile
            
            # Prepare efficiency data
            efficiency_data = {
                'Theory': profile.get('theory_multiplier', 1.0),
                'Problem Solving': profile.get('problem_solving_multiplier', 1.0),
                'Revision': profile.get('revision_multiplier', 1.0),
                'Exam Simulation': profile.get('exam_simulation_multiplier', 1.0),
                'Reading': profile.get('reading_multiplier', 1.0)
            }
            
            task_types = list(efficiency_data.keys())
            multipliers = list(efficiency_data.values())
            pct_diffs = [(m - 1.0) * 100 for m in multipliers]
            
            colors = ['green' if m < 0.85 else 'red' if m > 1.15 else 'gray' for m in multipliers]
            
            fig_efficiency = go.Figure()
            fig_efficiency.add_trace(go.Bar(
                y=task_types, x=pct_diffs, orientation='h',
                marker_color=colors, text=[f"{m:.2f}x" for m in multipliers],
                textposition='auto'
            ))
            
            fig_efficiency.update_layout(
                title="Time Multipliers (Left=Faster, Right=Slower)",
                xaxis=dict(range=[-30, 30])
            )
            st.plotly_chart(fig_efficiency, use_container_width=True)
            
            st.markdown("---")
            
            # ================================================================
            # SECTION 6: FATIGUE MANAGEMENT
            # ================================================================
            st.subheader(" Fatigue Evolution")
            
            if len(df_history) > 0:
                df_fatigue = df_history.copy()
                df_fatigue['task_num'] = range(1, len(df_fatigue) + 1)
                
                fig_fatigue = go.Figure()
                fig_fatigue.add_trace(go.Scatter(
                    x=df_fatigue['task_num'],
                    y=df_fatigue['fatigue'], # Uses 'fatigue' from history log
                    mode='lines+markers', name='Fatigue',
                    line=dict(color='darkred', width=3)
                ))
                
                # Add zones
                fig_fatigue.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.1)
                
                fig_fatigue.update_layout(title="Fatigue Score Over Time", yaxis=dict(range=[0, 105]))
                st.plotly_chart(fig_fatigue, use_container_width=True)
            
            st.markdown("---")
            
    with tabs[3]:
        if st.session_state.history_log:
            st.dataframe(pd.DataFrame(st.session_state.history_log))


if __name__ == "__main__":
    main()