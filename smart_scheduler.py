"""
================================================================================
FILE: smart_scheduler.py
COMPONENT: Smart Scheduler - Cognitive-Aware Task Scheduler
================================================================================

Coordinates three RL agents + UserProfileManager to generate optimal schedules.

This is the orchestration layer that brings together:
1. TimeSchedulingAgent - Learns WHEN to schedule tasks
2. DurationAllocationAgent - Learns HOW MUCH time users need
3. BreakSchedulingAgent - Learns WHEN users need breaks
4. UserProfileManager - Tracks user-specific efficiency patterns

Key Innovation: Adaptive Scheduling
------------------------------------
- Applies learned user efficiency multipliers
- Queries RL agents for optimal decisions
- Adjusts for urgency and prerequisites
- Inserts intelligent breaks to prevent burnout
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
from dataclasses import dataclass


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ScheduledItem:
    """
    Represents a single item in the schedule (task or break).
    
    Attributes:
        type: "task" or "break"
        task_id: Task identifier (None for breaks)
        task_name: Human-readable task name (None for breaks)
        start_time: When this item begins
        end_time: When this item ends
        cognitive_load: Mental effort required (None for breaks)
        reasoning: Explanation for scheduling decision
    """
    type: str
    task_id: Optional[str]
    task_name: Optional[str]
    start_time: datetime
    end_time: datetime
    cognitive_load: Optional[int]
    reasoning: str
    
    def duration_minutes(self) -> int:
        """Calculate duration in minutes."""
        return int((self.end_time - self.start_time).total_seconds() / 60)


# ============================================================================
# MAPPING DICTIONARIES
# ============================================================================

TASK_TYPE_MAP = {
    "theory": 0,
    "problem_solving": 1,
    "revision": 2,
    "exam_simulation": 3,
    "reading": 4
}

DIFFICULTY_MAP = {
    "easy": 0,
    "medium": 1,
    "hard": 2
}

PRIORITY_MAP = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "urgent": 3
}


# ============================================================================
# SMART SCHEDULER
# ============================================================================

class SmartScheduler:
    """
    Orchestrates RL agents and user profile to generate intelligent schedules.
    
    The scheduler:
    - Sorts tasks by priority and dependencies
    - Queries RL agents for optimal scheduling decisions
    - Applies learned user efficiency multipliers
    - Inserts breaks to prevent cognitive overload
    - Provides reasoning for each scheduling decision
    
    Attributes:
        time_agent: TimeSchedulingAgent instance
        duration_agent: DurationAllocationAgent instance
        break_agent: BreakSchedulingAgent instance
        user_profile: UserProfileManager instance
    """
    
    def __init__(
        self,
        time_agent,
        duration_agent,
        break_agent,
        user_profile_manager
    ) -> None:
        """
        Initialize with RL agents and user profile manager.
        
        Args:
            time_agent: TimeSchedulingAgent for optimal timing
            duration_agent: DurationAllocationAgent for time allocation
            break_agent: BreakSchedulingAgent for fatigue management
            user_profile_manager: UserProfileManager for efficiency tracking
        """
        self.time_agent = time_agent
        self.duration_agent = duration_agent
        self.break_agent = break_agent
        self.user_profile = user_profile_manager
    
    def generate_multi_day_schedule(
        self,
        tasks: List[Dict],
        deadline_days: int,
        daily_capacity_limit: int = 25,
        daily_start_hour: int = 9,
        daily_end_hour: int = 22,
        max_high_load_tasks: int = 2,
        buffer_minutes: int = 60
    ) -> Dict[datetime, Dict]:
        """
        Generate multi-day schedule using Iterative Greedy Selection.
        
        Fixes implemented:
        1. Soft Cap logic (in selection).
        2. Global Memory (remembers completed tasks across days).
        3. Accurate Metrics (counts scheduled tasks, not just selected ones).
        """
        
        # 1. Initialize master list of work
        remaining_tasks = [t for t in tasks]
        # SYSTEM FIX: Track completed tasks globally across days
        global_completed_tasks = set()
        multi_day_schedule = {}
        
        today = datetime.now().date()
        
        print(f"\n{'='*80}")
        print(f"ITERATIVE SCHEDULE GENERATION")
        print(f"Tasks to schedule: {len(remaining_tasks)}")
        print(f"{'='*80}\n")
        
        # 2. Iterate through each available day
        for day_offset in range(deadline_days):
            if not remaining_tasks:
                print("✅ All tasks scheduled early!")
                break
                
            current_date = today + timedelta(days=day_offset)
            print(f"Planning {current_date}...")

            # 3. Select tasks for THIS SPECIFIC DAY (Soft Cap Logic)
            candidate_tasks, _ = self._select_tasks_by_daily_capacity(
                remaining_tasks, 
                daily_capacity_limit
            )
            
            if not candidate_tasks:
                print("   ⚠️ No tasks fit capacity today.")
                # Save empty entry to maintain calendar structure
                multi_day_schedule[current_date] = {
                    "schedule": [],
                    "total_load": 0,
                    "tasks_scheduled": 0
                }
                continue

            # 4. Attempt to schedule them with Global Memory
            scheduled_items, day_dropped = self._generate_single_day_schedule(
                candidate_tasks,
                datetime.combine(current_date, datetime.min.time()),
                daily_start_hour, 
                daily_end_hour,
                daily_capacity_limit,
                max_high_load_tasks,
                buffer_minutes,
                previously_completed=global_completed_tasks  # Pass history
            )
            
            # 5. Extract IDs and Stats of successfully scheduled tasks
            scheduled_ids_today = set()
            real_load = 0
            real_task_count = 0
            
            for item in scheduled_items:
                if item.type == 'task':
                    scheduled_ids_today.add(item.task_id)
                    real_load += item.cognitive_load
                    real_task_count += 1
            
            # 6. Update Global Memory
            global_completed_tasks.update(scheduled_ids_today)
            
            # 7. CRITICAL STEP: Remove scheduled tasks from master list
            remaining_tasks = [t for t in remaining_tasks if t['task_id'] not in scheduled_ids_today]
            
            # 8. Save this day's results (Accurate metrics)
            multi_day_schedule[current_date] = {
                "schedule": scheduled_items,
                "total_load": real_load,
                "tasks_scheduled": real_task_count
            }
            
            print(f"   Scheduled: {real_task_count} tasks | Load: {real_load}")
            print(f"   Remaining: {len(remaining_tasks)} tasks")

        # 9. Handle Overflow
        if remaining_tasks:
            print(f"\n⚠️ DEADLINE EXCEEDED: {len(remaining_tasks)} tasks could not be scheduled.")
            
        return multi_day_schedule
    
    def _generate_single_day_schedule(
        self,
        selected_tasks: List[Dict],
        planning_date: datetime,
        daily_start_hour: int,
        daily_end_hour: int,
        daily_capacity_limit: int,
        max_high_load_tasks: int,
        buffer_minutes: int,
        previously_completed: Set[str] = None  # New Argument
    ) -> tuple[List[ScheduledItem], List[Dict]]:
        """
        Generate schedule for a single day.
        
        Fixes:
        - Accepts 'previously_completed' to check prerequisites correctly across days.
        - Uses dampened fatigue accumulation (4.0 multiplier).
        """
        # Initialize completed set with history from previous days
        completed_task_ids = set(previously_completed) if previously_completed else set()
        
        now = datetime.now()
        
        # Determine start time
        if planning_date.date() == now.date():
            # Planning for today - start from current time
            start_time = now
        else:
            # Planning for future date - start at daily_start_hour
            start_time = planning_date.replace(
                hour=daily_start_hour,
                minute=0,
                second=0,
                microsecond=0
            )
        
        # End time for this day
        end_time = planning_date.replace(
            hour=daily_end_hour,
            minute=0,
            second=0,
            microsecond=0
        )
        
        # Reserve buffer time
        effective_end_time = end_time - timedelta(minutes=buffer_minutes)
        
        # Safety check
        available_minutes = (effective_end_time - start_time).total_seconds() / 60
        
        if available_minutes < 30:
            return [], selected_tasks  # Not enough time
        
        # Sort selected tasks
        sorted_tasks = self._sort_tasks(selected_tasks)
        
        # Initialize scheduling state (fresh each day)
        current_time = start_time
        micro_fatigue = 0.0  # Reset each day
        high_load_count = 0
        consecutive_high_load = 0
        schedule: List[ScheduledItem] = []
        dropped_tasks: List[Dict] = []
        
        # Schedule tasks
        for task in sorted_tasks:
            # Check prerequisites (Now checks against global history too)
            prerequisites = task.get('prerequisites', [])
            if not all(prereq in completed_task_ids for prereq in prerequisites):
                dropped_tasks.append(task)
                continue
            
            # Get adjusted duration
            adjusted_duration = self._get_adjusted_duration(task)
            
            # High load limit check
            if task['cognitive_load'] >= 8 and high_load_count >= max_high_load_tasks:
                dropped_tasks.append(task)
                continue
            
            # Time fit check
            task_end = current_time + timedelta(minutes=adjusted_duration)
            if task_end > effective_end_time:
                dropped_tasks.append(task)
                continue
            
            # Query RL agents
            current_hour = current_time.hour
            time_state = self.time_agent.get_state(
                current_hour,
                task['task_type'],
                task['difficulty']
            )
            time_action = self.time_agent.get_action(time_state)
            
            # Break scheduling
            break_duration = self._should_insert_break(
                micro_fatigue,
                task,
                consecutive_high_load
            )
            
            if task['priority'] == 'urgent' and break_duration > 0:
                break_duration = break_duration // 2
            
            # Insert break
            if break_duration > 0:
                break_end = current_time + timedelta(minutes=break_duration)
                
                if break_end <= effective_end_time:
                    break_item = ScheduledItem(
                        type="break",
                        task_id=None,
                        task_name=f"{break_duration}-minute break",
                        start_time=current_time,
                        end_time=break_end,
                        cognitive_load=None,
                        reasoning=f"Recovery break (micro-fatigue: {micro_fatigue:.1f})"
                    )
                    
                    schedule.append(break_item)
                    micro_fatigue = max(0, micro_fatigue - (break_duration / 3))
                    current_time = break_end
                    consecutive_high_load = 0
            
            # Schedule task
            task_start = current_time
            task_end = task_start + timedelta(minutes=adjusted_duration)
            
            # Build reasoning
            reasoning_parts = [
                f"Selected within daily capacity ({daily_capacity_limit})"
            ]
            
            if task['cognitive_load'] >= 8:
                reasoning_parts.append(f"High-load task {high_load_count + 1}/{max_high_load_tasks}")
            
            original_duration = task.get('estimated_duration', 0)

            if original_duration > 0 and adjusted_duration != original_duration:
                diff_pct = ((adjusted_duration - original_duration) / original_duration) * 100
                reasoning_parts.append(
                    f"Duration {diff_pct:+.0f}% (your {task['task_type']} speed)"
                )

            if self.user_profile.is_peak_hour(task['task_type'], current_hour):
                reasoning_parts.append(f"Peak hour ({current_hour}:00)")
            
            if task['priority'] in ['urgent', 'high']:
                reasoning_parts.append(f"{task['priority'].title()} priority")
            
            reasoning = ". ".join(reasoning_parts)
            
            scheduled_task = ScheduledItem(
                type="task",
                task_id=task['task_id'],
                task_name=task['task_name'],
                start_time=task_start,
                end_time=task_end,
                cognitive_load=task['cognitive_load'],
                reasoning=reasoning
            )
            
            schedule.append(scheduled_task)
            
            # Update state
            current_time = task_end
            
            # SYSTEM CHANGE: Reduced micro-fatigue multiplier to 4.0
            micro_fatigue += task['cognitive_load'] * 4.0
            
            if task['cognitive_load'] >= 8:
                consecutive_high_load += 1
                high_load_count += 1
            else:
                consecutive_high_load = 0
            
            completed_task_ids.add(task['task_id'])
        
        return schedule, dropped_tasks
    
    def generate_next_day_schedule(
        self,
        tasks: List[Dict],
        planning_date: datetime,
        daily_start_hour: int = 9,
        daily_end_hour: int = 22,
        daily_capacity_limit: int = 25,
        max_high_load_tasks: int = 2,
        buffer_minutes: int = 60
    ) -> tuple[List[ScheduledItem], List[Dict]]:
        """
        Generate schedule for a specific day with capacity-aware task selection.
        Wrapper for single-day use cases.
        """
        # 1. Capacity Selection
        selected_tasks, capacity_dropped = self._select_tasks_by_daily_capacity(
            tasks,
            daily_capacity_limit
        )
        
        if not selected_tasks:
            return [], tasks
            
        # 2. Generate Schedule
        schedule, scheduling_dropped = self._generate_single_day_schedule(
            selected_tasks,
            planning_date,
            daily_start_hour,
            daily_end_hour,
            daily_capacity_limit,
            max_high_load_tasks,
            buffer_minutes
        )
        
        # Combine dropped tasks
        all_dropped = capacity_dropped + scheduling_dropped
        
        return schedule, all_dropped
    
    def _select_tasks_by_daily_capacity(
        self,
        tasks: List[Dict],
        daily_capacity_limit: int
    ) -> tuple[List[Dict], List[Dict]]:
        """
        Select optimal subset of tasks using Greedy Knapsack algorithm with Soft Cap.
        """
        # SYSTEM CHANGE: Allow 20% overflow buffer
        OVERFLOW_BUFFER = 1.2 
        max_load = daily_capacity_limit * OVERFLOW_BUFFER
        
        # Group tasks by priority tier
        priority_groups = {
            'urgent': [],
            'high': [],
            'medium': [],
            'low': []
        }
        
        for task in tasks:
            priority = task.get('priority', 'medium')
            if priority in priority_groups:
                priority_groups[priority].append(task)
            else:
                priority_groups['medium'].append(task)
        
        # Sort each tier by cognitive_load ascending
        for priority in priority_groups:
            priority_groups[priority].sort(key=lambda t: t.get('cognitive_load', 5))
        
        # Greedy selection
        selected_tasks = []
        dropped_tasks = []
        current_total_load = 0
        
        # Process tiers in priority order
        for priority in ['urgent', 'high', 'medium', 'low']:
            for task in priority_groups[priority]:
                task_load = task.get('cognitive_load', 5)
                
                # Check soft limit OR ensure at least 1 task
                if (current_total_load + task_load <= max_load) or (len(selected_tasks) == 0):
                    selected_tasks.append(task)
                    current_total_load += task_load
                else:
                    dropped_tasks.append(task)
        
        return selected_tasks, dropped_tasks
    
    def _sort_tasks(self, tasks: List[Dict]) -> List[Dict]:
        """
        Sort tasks for optimal scheduling.
        Priority > Prerequisites > Cognitive Load
        """
        def sort_key(task):
            priority = PRIORITY_MAP.get(task.get('priority', 'medium'), 1)
            prereq_count = len(task.get('prerequisites', []))
            cognitive_load = task.get('cognitive_load', 5)
            
            # Return tuple for sorting
            return (-priority, prereq_count, -cognitive_load)
        
        return sorted(tasks, key=sort_key)
    
    def _get_adjusted_duration(self, task: Dict) -> int:
        """
        Calculate adjusted task duration using user profile + RL agent.
        """
        base_duration = task['estimated_duration']
        task_type = task['task_type']
        difficulty = task['difficulty']
        
        # Step 1: Get user's learned efficiency for this task type
        user_adjusted = self.user_profile.get_adjusted_duration(task_type, base_duration)
        
        # Step 2: Query DurationAllocationAgent
        duration_state = self.duration_agent.get_state(
            task_type,
            difficulty,
            self.user_profile.profile.get(f'{task_type}_multiplier', 1.0)
        )
        duration_action = self.duration_agent.get_action(duration_state)
        rl_multiplier = self.duration_agent.get_duration_multiplier(duration_action)
        
        # Step 3: Combine both adjustments
        final_duration = int(user_adjusted * rl_multiplier)
        
        # Ensure minimum duration
        final_duration = max(15, final_duration)
        
        return final_duration
    
    def _should_insert_break(
        self,
        fatigue: float,
        task: Dict,
        consecutive_high_load: int
    ) -> int:
        """
        Query BreakSchedulingAgent to determine if break is needed.
        """
        break_state = self.break_agent.get_state(
            fatigue,
            task['cognitive_load'],
            consecutive_high_load
        )
        break_action = self.break_agent.get_action(break_state)
        return self.break_agent.get_break_duration(break_action)
    
    def _update_fatigue(self, fatigue: float, cognitive_load: int) -> float:
        """
        Update fatigue score based on cognitive load.
        """
        new_fatigue = fatigue + (cognitive_load * 8)
        return max(0, min(100, new_fatigue))


# ============================================================================
# TESTING AND DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("TESTING SMART SCHEDULER")
    print("=" * 80)
    print()
    
    # Import required components
    from rl_agent_decomposed import (
        TimeSchedulingAgent,
        DurationAllocationAgent,
        BreakSchedulingAgent
    )
    from user_profile_manager import UserProfileManager
    
    # Initialize components
    time_agent = TimeSchedulingAgent()
    duration_agent = DurationAllocationAgent()
    break_agent = BreakSchedulingAgent()
    user_profile = UserProfileManager()
    
    # Create scheduler
    scheduler = SmartScheduler(
        time_agent=time_agent,
        duration_agent=duration_agent,
        break_agent=break_agent,
        user_profile_manager=user_profile
    )
    
    # Define test tasks
    tasks = [
        {
            "task_id": "DS_001",
            "task_name": "Learn Graph Theory fundamentals",
            "task_type": "theory",
            "difficulty": "hard",
            "cognitive_load": 8,
            "estimated_duration": 90,
            "prerequisites": [],
            "priority": "high",
            "flexibility": "splittable"
        },
        {
            "task_id": "DS_002",
            "task_name": "Practice 10 easy Graph problems",
            "task_type": "problem_solving",
            "difficulty": "easy",
            "cognitive_load": 5,
            "estimated_duration": 60,
            "prerequisites": ["DS_001"],
            "priority": "medium",
            "flexibility": "splittable"
        },
        {
            "task_id": "DS_003",
            "task_name": "Practice 15 medium Graph problems",
            "task_type": "problem_solving",
            "difficulty": "medium",
            "cognitive_load": 7,
            "estimated_duration": 120,
            "prerequisites": ["DS_001", "DS_002"],
            "priority": "high",
            "flexibility": "splittable"
        }
    ]
    
    # Run test
    print("Generating schedule...")
    schedule = scheduler.generate_multi_day_schedule(
        tasks,
        deadline_days=3,
        daily_capacity_limit=25
    )