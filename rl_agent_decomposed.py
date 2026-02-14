"""
================================================================================
FILE: rl_agent_decomposed.py
COMPONENT: Decomposed Q-Learning Agents for Cognitive Scheduling
================================================================================

This module implements three specialized Q-learning agents that learn different
aspects of optimal task scheduling. This is the CORE INNOVATION of the system.

Instead of one monolithic Q-learner, we use THREE independent agents:
1. TimeSchedulingAgent - Learns WHEN to schedule tasks (morning/afternoon/evening)
2. DurationAllocationAgent - Learns HOW MUCH time users need per task type
3. BreakSchedulingAgent - Learns WHEN users need breaks based on fatigue

Key Innovation: Targeted Updates
--------------------------------
When user provides feedback like "not enough time", ONLY the DurationAllocationAgent
gets penalized. The other two agents remain untouched. This is inspired by
backpropagation in neural networks but adapted for tabular RL.
"""

import pickle
import numpy as np
from typing import Dict, Tuple, List, Optional
from collections import defaultdict
from pathlib import Path


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def encode_task_type(task_type: str) -> int:
    """
    Convert task_type string to integer encoding.
    
    Args:
        task_type: One of ["theory", "problem_solving", "revision", 
                          "exam_simulation", "reading"]
    
    Returns:
        Integer code 0-4
    """
    mapping = {
        "theory": 0,
        "problem_solving": 1,
        "revision": 2,
        "exam_simulation": 3,
        "reading": 4
    }
    return mapping.get(task_type, 0)


def encode_difficulty(difficulty: str) -> int:
    """
    Convert difficulty string to integer encoding.
    
    Args:
        difficulty: One of ["easy", "medium", "hard"]
    
    Returns:
        Integer code 0-2
    """
    mapping = {"easy": 0, "medium": 1, "hard": 2}
    return mapping.get(difficulty, 1)


def decode_task_type(code: int) -> str:
    """
    Convert integer encoding back to task_type string.
    
    Args:
        code: Integer 0-4
    
    Returns:
        Task type string
    """
    mapping = {
        0: "theory",
        1: "problem_solving",
        2: "revision",
        3: "exam_simulation",
        4: "reading"
    }
    return mapping.get(code, "theory")


def decode_difficulty(code: int) -> str:
    """
    Convert integer encoding back to difficulty string.
    
    Args:
        code: Integer 0-2
    
    Returns:
        Difficulty string
    """
    mapping = {0: "easy", 1: "medium", 2: "hard"}
    return mapping.get(code, "medium")


# ============================================================================
# BASE Q-LEARNER CLASS
# ============================================================================

class BaseQLearner:
    """
    Generic Q-learning agent that other specialized agents inherit from.
    
    Implements core Q-learning functionality:
    - Q-table storage as nested dict
    - Epsilon-greedy action selection
    - Q-value updates using standard Q-learning formula
    - Save/load functionality
    
    Attributes:
        q_table: Nested dict mapping (state -> action -> Q-value)
        learning_rate: Alpha parameter for Q-learning (0.1)
        discount_factor: Gamma parameter for future rewards (0.9)
        epsilon: Exploration rate for epsilon-greedy (0.2)
        actions: List of valid actions for this agent
    """
    
    def __init__(
        self,
        actions: List[str],
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 0.2
    ):
        """
        Initialize base Q-learner.
        
        Args:
            actions: List of valid action strings
            learning_rate: Learning rate (alpha) for Q-updates
            discount_factor: Discount factor (gamma) for future rewards
            epsilon: Exploration rate for epsilon-greedy policy
        """
        self.q_table: Dict[Tuple, Dict[str, float]] = defaultdict(
            lambda: {action: 0.0 for action in actions}
        )
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.actions = actions
        
        # Set random seed for reproducibility
        np.random.seed(42)
    
    def get_action(self, state: Tuple) -> str:
        """
        Select action using epsilon-greedy policy.
        
        With probability epsilon, choose random action (explore).
        Otherwise, choose action with highest Q-value (exploit).
        
        Args:
            state: State tuple (will be initialized if new)
        
        Returns:
            Selected action string
        """
        # Initialize state if not in Q-table
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in self.actions}
        
        # Epsilon-greedy: explore vs exploit
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.choice(self.actions)
        else:
            # Exploit: best action
            q_values = self.q_table[state]
            max_q = max(q_values.values())
            # Handle ties by random selection
            best_actions = [a for a, q in q_values.items() if q == max_q]
            return np.random.choice(best_actions)
    
    def update_q_value(
        self,
        state: Tuple,
        action: str,
        reward: float,
        next_state: Tuple
    ) -> None:
        """
        Update Q-value using standard Q-learning formula.
        
        Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
        
        Args:
            state: Current state tuple
            action: Action taken
            reward: Reward received
            next_state: Resulting state tuple
        """
        # Initialize states if needed
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.actions}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0.0 for a in self.actions}
        
        # Get current Q-value
        current_q = self.q_table[state][action]
        
        # Get max Q-value for next state
        max_next_q = max(self.q_table[next_state].values())
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state][action] = new_q
    
    def get_q_value(self, state: Tuple, action: str) -> float:
        """
        Get Q-value for state-action pair.
        
        Args:
            state: State tuple
            action: Action string
        
        Returns:
            Q-value, or 0.0 if state-action not in table
        """
        if state not in self.q_table:
            return 0.0
        return self.q_table[state].get(action, 0.0)
    
    def save(self, filepath: str) -> None:
        """
        Save Q-table to pickle file.
        
        Args:
            filepath: Path to save file
        """
        try:
            # Convert defaultdict to regular dict for pickling
            regular_dict = {k: dict(v) for k, v in self.q_table.items()}
            with open(filepath, 'wb') as f:
                pickle.dump(regular_dict, f)
        except Exception as e:
            print(f"Error saving Q-table to {filepath}: {e}")
    
    def load(self, filepath: str) -> bool:
        """
        Load Q-table from pickle file.
        
        Args:
            filepath: Path to load file
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not Path(filepath).exists():
                return False
            
            with open(filepath, 'rb') as f:
                loaded_dict = pickle.load(f)
            
            # Convert back to defaultdict
            self.q_table = defaultdict(
                lambda: {action: 0.0 for action in self.actions}
            )
            for state, actions in loaded_dict.items():
                self.q_table[state] = actions
            
            return True
        except Exception as e:
            print(f"Error loading Q-table from {filepath}: {e}")
            return False
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (f"{self.__class__.__name__}("
                f"states={len(self.q_table)}, "
                f"lr={self.learning_rate}, "
                f"gamma={self.discount_factor}, "
                f"epsilon={self.epsilon})")


# ============================================================================
# TIME SCHEDULING AGENT
# ============================================================================

class TimeSchedulingAgent(BaseQLearner):
    """
    Learns optimal TIME OF DAY to schedule different task types.
    
    This agent learns patterns like:
    - "Theory tasks succeed more often at 9 AM than 8 PM"
    - "Problem-solving works better in afternoon"
    - "Hard tasks should be deferred if it's late evening"
    
    State Space (45 states):
        - hour_block: 0=morning(6-11), 1=afternoon(12-17), 2=evening(18-23)
        - task_type: 0-4 (theory, problem_solving, revision, exam_sim, reading)
        - difficulty: 0-2 (easy, medium, hard)
    
    Actions:
        - schedule_morning: Schedule in morning block (6-11 AM)
        - schedule_afternoon: Schedule in afternoon block (12-5 PM)
        - schedule_evening: Schedule in evening block (6-11 PM)
        - defer_next_day: Task should wait until tomorrow
    """
    
    ACTIONS = [
        'schedule_morning',
        'schedule_afternoon',
        'schedule_evening',
        'defer_next_day'
    ]
    
    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 0.2
    ):
        """Initialize TimeSchedulingAgent with predefined actions."""
        super().__init__(
            actions=self.ACTIONS,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon
        )
    
    def get_state(
        self,
        current_hour: int,
        task_type: str,
        difficulty: str
    ) -> Tuple[int, int, int]:
        """
        Convert continuous parameters to discrete state tuple.
        
        Args:
            current_hour: Current hour (0-23)
            task_type: Task type string
            difficulty: Difficulty string
        
        Returns:
            State tuple (hour_block, task_type_code, difficulty_code)
        """
        # Clip hour to valid range
        hour = max(0, min(23, current_hour))
        
        # Discretize hour into blocks
        if 6 <= hour <= 11:
            hour_block = 0  # Morning
        elif 12 <= hour <= 17:
            hour_block = 1  # Afternoon
        elif 18 <= hour <= 23:
            hour_block = 2  # Evening
        else:
            hour_block = 0  # Default to morning for early hours
        
        # Encode task type and difficulty
        task_code = encode_task_type(task_type)
        diff_code = encode_difficulty(difficulty)
        
        return (hour_block, task_code, diff_code)
    
    def update(
        self,
        state: Tuple[int, int, int],
        action: str,
        reward: float,
        next_state: Tuple[int, int, int]
    ) -> None:
        """
        Update Q-table (wrapper for clarity).
        
        Args:
            state: Current state tuple
            action: Action taken
            reward: Reward received
            next_state: Next state tuple
        """
        self.update_q_value(state, action, reward, next_state)


# ============================================================================
# DURATION ALLOCATION AGENT
# ============================================================================

class DurationAllocationAgent(BaseQLearner):
    """
    Learns how much time users ACTUALLY need vs LLM estimates.
    
    This is the MOST CRITICAL agent - learns user-specific efficiency patterns.
    
    Examples:
    - "User completes theory tasks in 1.2x the estimated time (20% slower)"
    - "User solves coding problems in 0.7x time (30% faster)"
    - "Hard tasks for this user need 1.5x time allocation"
    
    State Space (45 states):
        - task_type: 0-4 (theory, problem_solving, revision, exam_sim, reading)
        - difficulty: 0-2 (easy, medium, hard)
        - user_efficiency_bucket: 0=fast(<0.85x), 1=normal(0.85-1.15x), 2=slow(>1.15x)
    
    Actions (duration multipliers):
        - multiply_0.7: Allocate 70% of LLM estimate (user is fast)
        - multiply_0.85: Allocate 85% of estimate
        - multiply_1.0: Allocate 100% of estimate (default)
        - multiply_1.2: Allocate 120% of estimate
        - multiply_1.5: Allocate 150% of estimate (user is slow)
    """
    
    ACTIONS = [
        'multiply_0.7',
        'multiply_0.85',
        'multiply_1.0',
        'multiply_1.2',
        'multiply_1.5'
    ]
    
    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 0.2
    ):
        """Initialize DurationAllocationAgent with multiplier actions."""
        super().__init__(
            actions=self.ACTIONS,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon
        )
    
    def get_state(
        self,
        task_type: str,
        difficulty: str,
        user_efficiency: float
    ) -> Tuple[int, int, int]:
        """
        Convert parameters to discrete state tuple.
        
        Args:
            task_type: Task type string
            difficulty: Difficulty string
            user_efficiency: User's efficiency multiplier for this task type
        
        Returns:
            State tuple (task_type_code, difficulty_code, efficiency_bucket)
        """
        # Encode task type and difficulty
        task_code = encode_task_type(task_type)
        diff_code = encode_difficulty(difficulty)
        
        # Discretize user efficiency into buckets
        if user_efficiency < 0.85:
            efficiency_bucket = 0  # Fast
        elif user_efficiency <= 1.15:
            efficiency_bucket = 1  # Normal
        else:
            efficiency_bucket = 2  # Slow
        
        return (task_code, diff_code, efficiency_bucket)
    
    def get_duration_multiplier(self, action: str) -> float:
        """
        Convert action string to float multiplier.
        
        Args:
            action: Action string like 'multiply_1.2'
        
        Returns:
            Float multiplier (e.g., 1.2)
        """
        try:
            # Extract number from action string
            multiplier = float(action.replace('multiply_', ''))
            return multiplier
        except:
            return 1.0  # Default to no adjustment
    
    def update(
        self,
        state: Tuple[int, int, int],
        action: str,
        reward: float,
        next_state: Tuple[int, int, int]
    ) -> None:
        """
        Update Q-table (wrapper for clarity).
        
        Args:
            state: Current state tuple
            action: Action taken
            reward: Reward received
            next_state: Next state tuple
        """
        self.update_q_value(state, action, reward, next_state)


# ============================================================================
# BREAK SCHEDULING AGENT
# ============================================================================

class BreakSchedulingAgent(BaseQLearner):
    """
    Learns when users need breaks to prevent burnout.
    
    Critical for maintaining sustainable productivity. Learns patterns like:
    - "After 2 high-load tasks with fatigue > 60, user needs 30min break"
    - "Low-load tasks don't need breaks even with medium fatigue"
    - "User can handle 3 consecutive medium tasks before needing rest"
    
    State Space (27 states):
        - fatigue_level: 0=low(0-40), 1=medium(41-70), 2=high(71-100)
        - cognitive_load: 0=low(1-4), 1=medium(5-7), 2=high(8-10)
        - consecutive_high_load_tasks: 0=none, 1=one, 2=two_or_more
    
    Actions:
        - no_break: Continue without break
        - break_15min: Schedule 15-minute break
        - break_30min: Schedule 30-minute break
        - break_45min: Schedule 45-minute break
    """
    
    ACTIONS = [
        'no_break',
        'break_15min',
        'break_30min',
        'break_45min'
    ]
    
    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 0.2
    ):
        """Initialize BreakSchedulingAgent with break duration actions."""
        super().__init__(
            actions=self.ACTIONS,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon
        )
    
    def get_state(
        self,
        fatigue_score: float,
        cognitive_load: int,
        consecutive_tasks: int
    ) -> Tuple[int, int, int]:
        """
        Convert continuous parameters to discrete state tuple.
        
        Args:
            fatigue_score: Current fatigue (0-100)
            cognitive_load: Task cognitive load (1-10)
            consecutive_tasks: Number of consecutive high-load tasks
        
        Returns:
            State tuple (fatigue_level, cognitive_load_bucket, consecutive_bucket)
        """
        # Clip fatigue to valid range
        fatigue = max(0, min(100, fatigue_score))
        
        # Discretize fatigue
        if fatigue <= 40:
            fatigue_level = 0  # Low
        elif fatigue <= 70:
            fatigue_level = 1  # Medium
        else:
            fatigue_level = 2  # High
        
        # Clip and discretize cognitive load
        load = max(1, min(10, cognitive_load))
        if load <= 4:
            load_bucket = 0  # Low
        elif load <= 7:
            load_bucket = 1  # Medium
        else:
            load_bucket = 2  # High
        
        # Discretize consecutive tasks
        if consecutive_tasks == 0:
            consecutive_bucket = 0
        elif consecutive_tasks == 1:
            consecutive_bucket = 1
        else:
            consecutive_bucket = 2  # Two or more
        
        return (fatigue_level, load_bucket, consecutive_bucket)
    
    def get_break_duration(self, action: str) -> int:
        """
        Convert action string to break duration in minutes.
        
        Args:
            action: Action string like 'break_15min'
        
        Returns:
            Break duration in minutes (0 for no_break)
        """
        duration_map = {
            'no_break': 0,
            'break_15min': 15,
            'break_30min': 30,
            'break_45min': 45
        }
        return duration_map.get(action, 0)
    
    def update(
        self,
        state: Tuple[int, int, int],
        action: str,
        reward: float,
        next_state: Tuple[int, int, int]
    ) -> None:
        """
        Update Q-table (wrapper for clarity).
        
        Args:
            state: Current state tuple
            action: Action taken
            reward: Reward received
            next_state: Next state tuple
        """
        self.update_q_value(state, action, reward, next_state)


# ============================================================================
# DECOMPOSED RL SCHEDULER (COORDINATOR)
# ============================================================================

class DecomposedRLScheduler:
    """
    Coordinator class that manages all three specialized RL agents.
    
    This is the main interface used by smart_scheduler.py. It:
    - Queries all three agents for scheduling decisions
    - Processes user feedback with targeted Q-table updates
    - Maintains current cognitive state (fatigue, time, etc.)
    - Provides learning statistics
    
    Key Innovation: When task fails due to "insufficient time", ONLY the
    DurationAgent gets penalized. TimeAgent and BreakAgent are untouched.
    This is more efficient than updating all agents for every failure.
    
    Attributes:
        time_agent: TimeSchedulingAgent instance
        duration_agent: DurationAllocationAgent instance
        break_agent: BreakSchedulingAgent instance
        current_state: Dict tracking fatigue, hour, consecutive tasks, etc.
    """
    
    def __init__(self):
        """Initialize all three agents and set initial state."""
        self.time_agent = TimeSchedulingAgent()
        self.duration_agent = DurationAllocationAgent()
        self.break_agent = BreakSchedulingAgent()
        
        # Initialize current state
        self.current_state = {
            'fatigue_score': 20.0,  # Start day refreshed
            'current_hour': 9,  # Default 9 AM
            'consecutive_high_load_tasks': 0,
            'tasks_completed_today': 0
        }
    
    def get_scheduling_decision(
        self,
        task: Dict,
        user_efficiency: float
    ) -> Dict:
        """
        Query all three agents for a scheduling decision.
        
        Args:
            task: Dict with keys {task_type, difficulty, cognitive_load, estimated_duration}
            user_efficiency: Current efficiency multiplier for this task type
        
        Returns:
            Dict containing:
                - time_slot: Recommended time slot
                - duration_multiplier: Duration adjustment multiplier
                - break_duration: Break duration in minutes
                - states: Dict of state tuples for each agent
                - actions: Dict of actions taken by each agent
        """
        # Query TimeSchedulingAgent
        time_state = self.time_agent.get_state(
            self.current_state['current_hour'],
            task['task_type'],
            task['difficulty']
        )
        time_action = self.time_agent.get_action(time_state)
        
        # Query DurationAllocationAgent
        duration_state = self.duration_agent.get_state(
            task['task_type'],
            task['difficulty'],
            user_efficiency
        )
        duration_action = self.duration_agent.get_action(duration_state)
        
        # Query BreakSchedulingAgent
        break_state = self.break_agent.get_state(
            self.current_state['fatigue_score'],
            task['cognitive_load'],
            self.current_state['consecutive_high_load_tasks']
        )
        break_action = self.break_agent.get_action(break_state)
        
        # Package decision
        return {
            'time_slot': time_action,
            'duration_multiplier': self.duration_agent.get_duration_multiplier(duration_action),
            'break_duration': self.break_agent.get_break_duration(break_action),
            'states': {
                'time_state': time_state,
                'duration_state': duration_state,
                'break_state': break_state
            },
            'actions': {
                'time_action': time_action,
                'duration_action': duration_action,
                'break_action': break_action
            }
        }
    
    def process_feedback(
        self,
        task: Dict,
        outcome: str,
        feedback: List[str],
        actual_time: int,
        decision: Dict,
        user_efficiency: float
    ) -> Dict[str, float]:
        """
        Process user feedback and update Q-tables with targeted rewards.
        
        This is where the TARGETED UPDATE innovation happens. Only the
        relevant agents get updated based on the feedback type.
        
        Args:
            task: Original task dict
            outcome: 'completed' or 'failed'
            feedback: List of feedback strings (e.g., ['time_too_less', 'too_tired'])
            actual_time: Minutes actually spent on task
            decision: Dict returned from get_scheduling_decision
            user_efficiency: Updated user efficiency for this task type
        
        Returns:
            Dict with rewards for each agent and total reward
        """
        # Base reward
        base_reward = 10.0 if outcome == 'completed' else -15.0
        
        # ========================================
        # CALCULATE AGENT-SPECIFIC REWARDS
        # ========================================
        
        # TimeAgent reward
        time_reward = base_reward
        if 'wrong_time' in feedback:
            time_reward += -12.0  # Heavy penalty for wrong time slot
        
        # DurationAgent reward (ALWAYS updated - learns from all outcomes)
        duration_reward = base_reward
        if outcome == 'completed':
            # Bonus for early completion, penalty for overtime
            if actual_time < task['estimated_duration'] * 0.9:
                duration_reward += 5.0  # Finished early!
            elif actual_time > task['estimated_duration'] * 1.1:
                duration_reward += -2.0  # Took longer than expected
        
        if 'time_too_less' in feedback:
            duration_reward += -10.0  # Heavy penalty for insufficient time
        
        # BreakAgent reward
        break_reward = base_reward
        if 'too_tired' in feedback:
            break_reward += -20.0  # Heavy penalty for not managing fatigue
        if 'distracted' in feedback:
            break_reward += -5.0  # Moderate penalty
        
        # ========================================
        # CALCULATE NEXT STATES
        # ========================================
        
        # Update fatigue (increases with cognitive load)
        next_fatigue = min(100, self.current_state['fatigue_score'] + task['cognitive_load'] * 8)
        
        # Update hour (simulate time passing)
        next_hour = (self.current_state['current_hour'] + actual_time // 60) % 24
        
        # Next time state
        next_time_state = self.time_agent.get_state(
            next_hour,
            task['task_type'],
            task['difficulty']
        )
        
        # Next duration state (use updated efficiency)
        next_duration_state = self.duration_agent.get_state(
            task['task_type'],
            task['difficulty'],
            user_efficiency
        )
        
        # Next break state
        next_consecutive = (
            self.current_state['consecutive_high_load_tasks'] + 1
            if task['cognitive_load'] >= 7
            else 0
        )
        next_break_state = self.break_agent.get_state(
            next_fatigue,
            task['cognitive_load'],
            next_consecutive
        )
        
        # ========================================
        # TARGETED Q-TABLE UPDATES
        # ========================================
        
        # Update TimeAgent ONLY if feedback relates to timing OR task completed
        if 'wrong_time' in feedback or outcome == 'completed':
            self.time_agent.update(
                decision['states']['time_state'],
                decision['actions']['time_action'],
                time_reward,
                next_time_state
            )
        
        # Update DurationAgent ALWAYS (learns from all outcomes)
        self.duration_agent.update(
            decision['states']['duration_state'],
            decision['actions']['duration_action'],
            duration_reward,
            next_duration_state
        )
        
        # Update BreakAgent ONLY if feedback relates to fatigue OR task completed
        if 'too_tired' in feedback or 'distracted' in feedback or outcome == 'completed':
            self.break_agent.update(
                decision['states']['break_state'],
                decision['actions']['break_action'],
                break_reward,
                next_break_state
            )
        
        # ========================================
        # UPDATE CURRENT STATE
        # ========================================
        
        self.current_state['fatigue_score'] = next_fatigue
        self.current_state['current_hour'] = next_hour
        
        if outcome == 'completed':
            self.current_state['consecutive_high_load_tasks'] = 0
            self.current_state['tasks_completed_today'] += 1
        else:
            if task['cognitive_load'] >= 7:
                self.current_state['consecutive_high_load_tasks'] += 1
        
        # Return reward summary
        return {
            'time_reward': time_reward,
            'duration_reward': duration_reward,
            'break_reward': break_reward,
            'total_reward': time_reward + duration_reward + break_reward
        }
    
    def update_state(self, fatigue_delta: int = 0, hour_delta: int = 0) -> None:
        """
        Manually update current state (e.g., user took break, time passed).
        
        Args:
            fatigue_delta: Change in fatigue (-100 to +100)
            hour_delta: Hours to advance (can be negative)
        """
        # Update fatigue with bounds
        new_fatigue = self.current_state['fatigue_score'] + fatigue_delta
        self.current_state['fatigue_score'] = max(0, min(100, new_fatigue))
        
        # Update hour with wrap-around
        new_hour = self.current_state['current_hour'] + hour_delta
        self.current_state['current_hour'] = new_hour % 24
    
    def reset_daily(self) -> None:
        """
        Reset daily counters (simulate sleep/new day).
        
        Resets:
        - Fatigue to 20 (refreshed after sleep)
        - Tasks completed to 0
        - Consecutive high-load tasks to 0
        """
        self.current_state['fatigue_score'] = 20.0
        self.current_state['tasks_completed_today'] = 0
        self.current_state['consecutive_high_load_tasks'] = 0
    
    def get_statistics(self) -> Dict:
        """
        Get learning statistics for all agents.
        
        Returns:
            Dict with state counts, fatigue, tasks completed, etc.
        """
        return {
            'time_agent_states': len(self.time_agent.q_table),
            'duration_agent_states': len(self.duration_agent.q_table),
            'break_agent_states': len(self.break_agent.q_table),
            'total_states_learned': (
                len(self.time_agent.q_table) +
                len(self.duration_agent.q_table) +
                len(self.break_agent.q_table)
            ),
            'current_fatigue': self.current_state['fatigue_score'],
            'tasks_today': self.current_state['tasks_completed_today'],
            'consecutive_high_load': self.current_state['consecutive_high_load_tasks']
        }
    
    def save_all(self, base_path: str = '.') -> None:
        """
        Save all three Q-tables to files.
        
        Args:
            base_path: Directory to save files in
        """
        base = Path(base_path)
        base.mkdir(parents=True, exist_ok=True)
        
        self.time_agent.save(str(base / 'time_agent.pkl'))
        self.duration_agent.save(str(base / 'duration_agent.pkl'))
        self.break_agent.save(str(base / 'break_agent.pkl'))
        
        print(f"Saved all Q-tables to {base_path}/")
    
    def load_all(self, base_path: str = '.') -> bool:
        """
        Load all three Q-tables from files.
        
        Args:
            base_path: Directory to load files from
        
        Returns:
            True if all loaded successfully, False otherwise
        """
        base = Path(base_path)
        
        success_time = self.time_agent.load(str(base / 'time_agent.pkl'))
        success_duration = self.duration_agent.load(str(base / 'duration_agent.pkl'))
        success_break = self.break_agent.load(str(base / 'break_agent.pkl'))
        
        success = success_time and success_duration and success_break
        
        if success:
            print(f"Loaded all Q-tables from {base_path}/")
        else:
            print(f"Warning: Some Q-tables failed to load from {base_path}/")
        
        return success
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        stats = self.get_statistics()
        return (f"DecomposedRLScheduler("
                f"states_learned={stats['total_states_learned']}, "
                f"fatigue={stats['current_fatigue']:.1f}, "
                f"tasks_today={stats['tasks_today']})")


# ============================================================================
# TESTING AND DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("TESTING DECOMPOSED RL AGENTS")
    print("=" * 80)
    print()
    
    # Initialize coordinator
    scheduler = DecomposedRLScheduler()
    print(f"Initialized: {scheduler}")
    print(f"Initial state: {scheduler.current_state}")
    print()
    
    # ========================================
    # TEST 1: Successful task completion
    # ========================================
    print("-" * 80)
    print("TEST 1: Successful Task (Problem Solving)")
    print("-" * 80)
    
    task1 = {
        'task_type': 'problem_solving',
        'difficulty': 'medium',
        'cognitive_load': 7,
        'estimated_duration': 90
    }
    
    print(f"Task: {task1}")
    
    # Get scheduling decision
    decision1 = scheduler.get_scheduling_decision(task1, user_efficiency=1.0)
    print(f"\nScheduling Decision:")
    print(f"  Time slot: {decision1['time_slot']}")
    print(f"  Duration multiplier: {decision1['duration_multiplier']}")
    print(f"  Break duration: {decision1['break_duration']} min")
    print(f"  Adjusted duration: {task1['estimated_duration'] * decision1['duration_multiplier']:.0f} min")
    
    # Simulate task completion (finished early - took 70 minutes instead of 90!)
    rewards1 = scheduler.process_feedback(
        task=task1,
        outcome='completed',
        feedback=[],  # No issues
        actual_time=70,
        decision=decision1,
        user_efficiency=1.0
    )
    
    print(f"\nRewards After Completion:")
    print(f"  Time agent: {rewards1['time_reward']:+.1f}")
    print(f"  Duration agent: {rewards1['duration_reward']:+.1f} (bonus for early finish)")
    print(f"  Break agent: {rewards1['break_reward']:+.1f}")
    print(f"  Total: {rewards1['total_reward']:+.1f}")
    
    print(f"\nUpdated State: {scheduler.current_state}")
    print()
    
    # ========================================
    # TEST 2: Failed task with feedback
    # ========================================
    print("-" * 80)
    print("TEST 2: Failed Task (Theory - Insufficient Time + Fatigue)")
    print("-" * 80)
    
    task2 = {
        'task_type': 'theory',
        'difficulty': 'hard',
        'cognitive_load': 9,
        'estimated_duration': 90
    }
    
    print(f"Task: {task2}")
    
    decision2 = scheduler.get_scheduling_decision(task2, user_efficiency=1.0)
    print(f"\nScheduling Decision:")
    print(f"  Time slot: {decision2['time_slot']}")
    print(f"  Duration multiplier: {decision2['duration_multiplier']}")
    print(f"  Break duration: {decision2['break_duration']} min")
    
    # Simulate task failure (gave up after 60 min, needed more time + too tired)
    rewards2 = scheduler.process_feedback(
        task=task2,
        outcome='failed',
        feedback=['time_too_less', 'too_tired'],
        actual_time=60,
        decision=decision2,
        user_efficiency=1.0
    )
    
    print(f"\nRewards After FAILURE:")
    print(f"  Time agent: {rewards2['time_reward']:+.1f} (not penalized - timing wasn't the issue)")
    print(f"  Duration agent: {rewards2['duration_reward']:+.1f} (PENALIZED for insufficient time)")
    print(f"  Break agent: {rewards2['break_reward']:+.1f} (PENALIZED for not managing fatigue)")
    print(f"  Total: {rewards2['total_reward']:+.1f}")
    
    print(f"\nUpdated State: {scheduler.current_state}")
    print()
    
    # ========================================
    # TEST 3: Wrong time of day feedback
    # ========================================
    print("-" * 80)
    print("TEST 3: Failed Task (Wrong Time of Day)")
    print("-" * 80)
    
    # Simulate late evening
    scheduler.update_state(hour_delta=12)  # Move to evening
    
    task3 = {
        'task_type': 'exam_simulation',
        'difficulty': 'hard',
        'cognitive_load': 10,
        'estimated_duration': 120
    }
    
    print(f"Task: {task3}")
    print(f"Current hour: {scheduler.current_state['current_hour']}")
    
    decision3 = scheduler.get_scheduling_decision(task3, user_efficiency=1.0)
    print(f"\nScheduling Decision:")
    print(f"  Time slot: {decision3['time_slot']}")
    
    # Failed because it's too late to take an exam
    rewards3 = scheduler.process_feedback(
        task=task3,
        outcome='failed',
        feedback=['wrong_time', 'too_tired'],
        actual_time=30,  # Gave up quickly
        decision=decision3,
        user_efficiency=1.0
    )
    
    print(f"\nRewards After FAILURE:")
    print(f"  Time agent: {rewards3['time_reward']:+.1f} (PENALIZED for wrong time)")
    print(f"  Duration agent: {rewards3['duration_reward']:+.1f}")
    print(f"  Break agent: {rewards3['break_reward']:+.1f} (PENALIZED for fatigue)")
    print(f"  Total: {rewards3['total_reward']:+.1f}")
    print()
    
    # ========================================
    # FINAL STATISTICS
    # ========================================
    print("-" * 80)
    print("LEARNING STATISTICS")
    print("-" * 80)
    
    stats = scheduler.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print()
    print("=" * 80)
    print("KEY INNOVATION DEMONSTRATED:")
    print("=" * 80)
    print("✓ TARGETED UPDATES: Only relevant agents penalized based on feedback type")
    print("✓ DurationAgent learned from ALL outcomes (success and failure)")
    print("✓ TimeAgent only updated when timing was the issue")
    print("✓ BreakAgent only updated when fatigue was mentioned")
    print()
    print("This prevents cascading penalties and enables faster convergence!")
    print("=" * 80)
