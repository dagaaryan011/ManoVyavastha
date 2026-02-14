# 🧠 Cognitive-Aware Task Scheduler
### RL + LLM Hybrid System for Personalized Study Planning
**HackXcelerate 2026 - PS-03 Solution**

---

## 🎯 Problem Statement

Traditional task schedulers treat all users the same, but people work at different speeds and have different cognitive patterns. This system **learns your individual efficiency** and **adapts scheduling** to match your mental capacity.

## 💡 Key Innovation: Decomposed RL + Bidirectional Learning

### 1. **Three Specialized RL Agents** (Not One Monolithic Agent)
- **TimeSchedulingAgent**: Learns WHEN to schedule tasks (morning/afternoon/evening)
- **DurationAllocationAgent**: Learns HOW MUCH time you need per task type
- **BreakSchedulingAgent**: Learns WHEN you need breaks based on fatigue

### 2. **Targeted Updates** (Inspired by Backpropagation)
When you fail a task due to "insufficient time":
- ❌ Traditional: All agents penalized equally
- ✅ Our System: **ONLY DurationAgent** penalized
- Result: Faster convergence, better learning

### 3. **Bidirectional Learning**
- Task completed **early** → Reduce time estimates
- Task took **longer** → Increase time estimates
- Converges to accurate user-specific durations

### 4. **Task-Type Specific Efficiency**
- You might be 30% **faster** at coding
- But 20% **slower** at learning theory
- System tracks 5 separate multipliers

---

## 🏗️ System Architecture

```
USER INPUT → LLM DECOMPOSER → RL SCHEDULER → TIMETABLE → FEEDBACK → Q-TABLE UPDATES
     ↓              ↓                ↓              ↓           ↓            ↓
"Study DS     Breaks into      Queries 3 RL    Generated   User marks   Targeted
exam in       6 sub-tasks      agents +        schedule    complete/    updates to
3 days"       with params      user profile    with times  failed       specific
                                                                         agents only
```

---

## 📦 Installation

### 1. Clone/Download Files

Ensure you have all these files in the same directory:

```
project/
├── app.py
├── rl_agent_decomposed.py
├── user_profile_manager.py
├── llm_task_decomposer.py
├── smart_scheduler.py
├── requirements.txt
└── README.md
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add Groq API Key (Optional but Recommended)

Open `llm_task_decomposer.py` and update line ~28:

```python
DEFAULT_GROQ_API_KEY = "gsk_your_actual_key_here"
```

Get free key: https://console.groq.com

**Note:** System works perfectly without API key using rule-based fallback!

---

## 🚀 Running the App

```bash
streamlit run app.py
```

App opens at: http://localhost:8501

---

## 📖 User Guide

### Tab 1: Generate Schedule

1. **Enter your goal:**
   ```
   Prepare for Data Structures exam in 3 days.
   Focus on Graphs and Dynamic Programming.
   ```

2. **Set parameters:**
   - Deadline: 3 days
   - Available hours: 8 hours

3. **Click "Generate Schedule"**

**What happens:**
- LLM breaks goal into 5-7 sub-tasks
- RL agents optimize scheduling
- User profile applies efficiency multipliers
- Schedule generated with reasoning

### Tab 2: Current Schedule

**View your personalized schedule:**

```
09:00 - 09:30  ☕ Break (30 min)
09:30 - 11:45  🔴 Learn Graph Theory (135 min, load 8/10)
               💡 Duration increased 50% based on your theory speed
11:45 - 12:30  ☕ Break (45 min)
12:30 - 13:30  🟡 Practice problems (60 min, load 5/10)
```

**For each task:**
- ✅ **Done** → System learns you're faster/slower
- ❌ **Failed** → Targeted feedback collection

**If task failed:**
```
What went wrong?
☑ Not enough time allocated     → Updates DurationAgent only
☑ Too tired                      → Updates BreakAgent only
☐ Wrong time of day              → Updates TimeAgent only
☐ Too difficult                  → Updates success rate only
☐ Got distracted                 → Updates BreakAgent
```

**This is the innovation!** Only relevant agents learn from each type of failure.

**Regenerate Button:**
- After marking tasks, click "🔄 Regenerate"
- System creates new schedule with updated learning
- See how durations/breaks adapt!

### Tab 3: Performance Analytics

**Track learning progress:**
- Q-table states learned (grows as you use the system)
- Your efficiency profile:
  ```
  🚀 Problem Solving: 0.85x (15% faster than average!)
  🐢 Theory Learning: 1.15x (15% slower, needs more time)
  ```
- Fatigue evolution chart
- Task completion rate
- Reward distribution

### Tab 4: History

**Review all completed tasks:**
- Timestamp, task name, outcome
- Actual time spent
- Reward earned
- Fatigue after task
- Export to CSV

---

## 🎬 Demo Script (5 Minutes)

### 1. Problem Introduction (30s)
"Traditional schedulers ignore mental capacity. They don't know if you're fast at coding but slow at theory."

### 2. LLM Decomposition (30s)
```
Input: "Prepare for DS exam in 3 days. Focus on Graphs and DP."
Output: 7 intelligent sub-tasks with prerequisites
```

### 3. Initial Schedule (30s)
"Notice all tasks start with 1.0x multipliers - system hasn't learned yet."

### 4. Simulate Early Completion (30s)
```
Task: 90 min theory task
User: Completed in 70 min
System: ✅ Reduces theory multiplier to 0.93x
```

### 5. Simulate Failure with Targeted Update (60s)
```
Task: 90 min coding task
User: Failed after 60 min
Feedback: ☑ Not enough time allocated
System: ONLY updates DurationAgent (+10% penalty)
         TimeAgent and BreakAgent untouched!
```

### 6. Show Adaptation (30s)
```
Click "Regenerate"
New schedule: 
  - Theory tasks: 84 min (reduced!)
  - Coding tasks: 99 min (increased!)
  - System learned YOUR cognitive fingerprint
```

### 7. Show Q-Table Growth (20s)
```
Performance tab:
  - Time Agent: 12 states learned
  - Duration Agent: 15 states learned  
  - Break Agent: 8 states learned
```

### 8. Closing (20s)
"The system learns your INDIVIDUAL cognitive patterns. Not generic - personalized to YOU."

---

## 🔬 Technical Details

### RL Algorithm: Q-Learning

```python
Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
```

- Learning rate (α): 0.1
- Discount factor (γ): 0.9
- Exploration rate (ε): 0.2

### User Profile: Exponential Moving Average

```python
new_multiplier = 0.8 * old_multiplier + 0.2 * (actual_time / estimated_time)
```

- β = 0.8: 80% weight to history
- 1-β = 0.2: 20% weight to new observation

### State Spaces

| Agent | State Dimensions | Total States |
|-------|-----------------|--------------|
| Time | (hour_block × task_type × difficulty) | 3 × 5 × 3 = 45 |
| Duration | (task_type × difficulty × efficiency_bucket) | 5 × 3 × 3 = 45 |
| Break | (fatigue × cognitive_load × consecutive) | 3 × 3 × 3 = 27 |

---

## 🎯 Key Features

✅ **LLM Task Decomposition** - Groq API (Llama 3.3 70B) with rule-based fallback  
✅ **Decomposed RL Agents** - Three specialized learners, not one monolithic agent  
✅ **Targeted Updates** - Only relevant agents penalized based on feedback type  
✅ **Bidirectional Learning** - Updates from both early and late completion  
✅ **Task-Type Efficiency** - Separate multipliers for theory/coding/revision/etc  
✅ **Fatigue Management** - Intelligent break insertion based on cognitive load  
✅ **Peak Hours Learning** - Tracks when you perform best for each task type  
✅ **Prerequisite Handling** - Ensures logical task ordering  
✅ **Staged Penalties** - Encourages retries (30% penalty first, 70% if second fails)  
✅ **Persistent Learning** - Q-tables and profiles saved to disk  
✅ **Schedule Regeneration** - One-click regenerate with updated learning  
✅ **Performance Analytics** - Visual learning progress tracking  

---

## 📊 Example Output

### Initial Schedule (Before Learning):
```
Theory task: 90 min (1.0x multiplier)
Coding task: 90 min (1.0x multiplier)
```

### After 5 Tasks (System Learned):
```
Theory task: 108 min (1.2x - you're slower at theory)
Coding task: 63 min (0.7x - you're fast at coding!)
```

**System adapted to YOUR cognitive fingerprint!**

---

## 🐛 Troubleshooting

### "Groq API Error"
- Either add your Groq API key
- Or ignore - fallback mode works great!

### "No module named 'streamlit'"
```bash
pip install streamlit
```

### "Q-tables not saving"
- Check write permissions in current directory
- Agents save to `./saved_agents/` folder

### "Schedule looks wrong"
- Remember: System learns over time
- Generate multiple schedules and provide feedback
- After 3-5 tasks, system adapts significantly

---

## 🚀 Future Enhancements

- 📄 PDF syllabus upload for context-aware decomposition
- 📅 Google Calendar integration
- 👥 Team scheduling for group study
- ⌚ Wearable integration (real-time fatigue from heart rate)
- 🧠 Deep Q-Networks (replace Q-table with neural network)
- 🔄 Automatic continuous rescheduling

---

## 📝 License & Credits

**Team:** [Your Team Name]  
**Hackathon:** HackXcelerate 2026  
**Problem Statement:** PS-03 - Cognitive-Aware Task Scheduling  

**Technologies:**
- Reinforcement Learning (Q-Learning)
- Large Language Models (Groq/Llama 3.3)
- Streamlit (Web UI)
- Python 3.10+

---

## 🎓 Academic References

- Sutton & Barto - Reinforcement Learning: An Introduction
- Silver et al. - Mastering the Game of Go with Deep RL
- OpenAI - GPT-based Task Decomposition
- Groq - Fast LLM Inference

---

**Built with ❤️ for cognitive optimization and personalized learning**

🧠 *Because everyone learns differently - your scheduler should too!*
