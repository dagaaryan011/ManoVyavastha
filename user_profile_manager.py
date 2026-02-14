"""
================================================================================
FILE: user_profile_manager.py
COMPONENT: User Profile Manager - Personalized Learning Tracker
================================================================================

This module manages user-specific cognitive performance data that RL agents use
to make scheduling decisions. Tracks how fast/slow users work on different task
types and maintains running history of performance patterns.

Key Innovation: Task-Type Specific Efficiency Tracking
-------------------------------------------------------
Instead of treating all tasks equally, maintains SEPARATE efficiency multipliers
for each task type (theory, problem_solving, revision, exam_simulation, reading).

Bidirectional Learning:
- Task completed EARLY → Reduce multiplier (user is faster)
- Task took LONGER → Increase multiplier (user needs more time)
- Task FAILED with "time_too_less" → Aggressive 10% increase

Uses Exponential Moving Average (EMA) for stability:
new_value = β * old_value + (1-β) * observed_value (β = 0.8)
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import os


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_bucket_label(multiplier: float) -> str:
    """
    Convert multiplier to human-readable label.
    
    Args:
        multiplier: Efficiency multiplier value
    
    Returns:
        'fast' | 'normal' | 'slow'
    """
    if multiplier < 0.85:
        return 'fast'
    elif multiplier <= 1.15:
        return 'normal'
    else:
        return 'slow'


def calculate_efficiency_change(old: float, new: float) -> str:
    """
    Describe change in efficiency.
    
    Args:
        old: Previous multiplier value
        new: New multiplier value
    
    Returns:
        Descriptive string of the change
    """
    change = new - old
    abs_change = abs(change)
    
    if abs_change < 0.01:
        return 'unchanged'
    
    if change > 0:  # Increased (slower)
        if abs_change > 0.1:
            return 'significantly increased'
        elif abs_change > 0.05:
            return 'increased'
        else:
            return 'slightly increased'
    else:  # Decreased (faster)
        if abs_change > 0.1:
            return 'significantly decreased'
        elif abs_change > 0.05:
            return 'decreased'
        else:
            return 'slightly decreased'


# ============================================================================
# USER PROFILE MANAGER
# ============================================================================

class UserProfileManager:
    """
    Manages user-specific cognitive performance data and learning patterns.
    
    Tracks:
    - Task-type specific efficiency multipliers
    - Success rates per task type
    - Current cognitive state (fatigue, tasks completed, failures)
    - Learned peak hours for different task types
    - Complete task history for analysis
    
    Attributes:
        profile: Dict containing all user metrics and preferences
        task_history: List of all attempted tasks with outcomes
        ema_beta: Smoothing factor for exponential moving average (0.8)
    """
    
    VALID_TASK_TYPES = [
        'theory',
        'problem_solving',
        'revision',
        'exam_simulation',
        'reading'
    ]
    
    def __init__(self, ema_beta: float = 0.8):
        """
        Initialize profile manager with default values.
        
        Args:
            ema_beta: Smoothing factor for EMA updates (default 0.8)
                     Higher = more weight to history, lower = faster adaptation
        """
        self.ema_beta = ema_beta
        
        # Initialize profile with default values
        self.profile: Dict = {
            # Task-type specific time multipliers
            'theory_multiplier': 1.0,
            'problem_solving_multiplier': 1.0,
            'revision_multiplier': 1.0,
            'exam_simulation_multiplier': 1.0,
            'reading_multiplier': 1.0,
            
            # Success rates per task type (start neutral)
            'theory_success_rate': 0.5,
            'problem_solving_success_rate': 0.5,
            'revision_success_rate': 0.5,
            'exam_simulation_success_rate': 0.5,
            'reading_success_rate': 0.5,
            
            # Current cognitive state
            'fatigue_score': 20.0,
            'tasks_completed_today': 0,
            'consecutive_failures': 0,
            
            # Learned peak hours (default patterns)
            'theory_peak_hours': [9, 10, 11],
            'problem_solving_peak_hours': [14, 15, 16, 20, 21],
            'revision_peak_hours': [8, 9, 19, 20, 21],
            'exam_simulation_peak_hours': [10, 11, 14, 15],
            'reading_peak_hours': [7, 8, 20, 21, 22],
            
            # Fatigue management
            'break_threshold_fatigue': 70.0,
            'fatigue_recovery_rate': 5.0,
            'max_tasks_per_day': 8
        }
        
        # Task history
        self.task_history: List[Dict] = []
    
    def get_adjusted_duration(self, task_type: str, llm_estimate: int) -> int:
        """
        Calculate adjusted task duration based on user's learned efficiency.
        
        Args:
            task_type: One of VALID_TASK_TYPES
            llm_estimate: Duration estimated by LLM in minutes
        
        Returns:
            Adjusted duration in minutes (rounded to nearest integer)
        
        Example:
            llm_estimate = 90 minutes
            user's problem_solving_multiplier = 0.7 (fast coder)
            return: 90 * 0.7 = 63 minutes
        """
        if task_type not in self.VALID_TASK_TYPES:
            return llm_estimate
        
        multiplier_key = f"{task_type}_multiplier"
        multiplier = self.profile.get(multiplier_key, 1.0)
        
        return int(llm_estimate * multiplier)
    
    def update_on_completion(
        self,
        task: Dict,
        actual_time: int,
        current_hour: Optional[int] = None
    ) -> Dict:
        """
        Update profile when task is completed successfully.
        
        Args:
            task: Dict with keys {task_type, difficulty, cognitive_load, estimated_duration}
            actual_time: Minutes actually spent (user input)
            current_hour: Hour when task was completed (0-23), optional
        
        Returns:
            Dict with update details including old/new multipliers
        """
        task_type = task['task_type']
        estimated = task['estimated_duration']
        
        # Validate task type
        if task_type not in self.VALID_TASK_TYPES:
            return {'error': f'Invalid task_type: {task_type}'}
        
        # Calculate observed efficiency
        observed_efficiency = actual_time / estimated if estimated > 0 else 1.0
        
        # Get current values
        multiplier_key = f"{task_type}_multiplier"
        success_rate_key = f"{task_type}_success_rate"
        
        old_multiplier = self.profile[multiplier_key]
        old_success_rate = self.profile[success_rate_key]
        
        # Update multiplier using Exponential Moving Average
        new_multiplier = (
            self.ema_beta * old_multiplier +
            (1 - self.ema_beta) * observed_efficiency
        )
        
        # Clip to safety bounds
        new_multiplier = max(0.3, min(2.0, new_multiplier))
        self.profile[multiplier_key] = new_multiplier
        
        # Update success rate (EMA with 1.0 since succeeded)
        new_success_rate = (
            self.ema_beta * old_success_rate +
            (1 - self.ema_beta) * 1.0
        )
        self.profile[success_rate_key] = new_success_rate
        
        # Increase fatigue based on cognitive load
        fatigue_before = self.profile['fatigue_score']
        fatigue_increase = task['cognitive_load'] * 3.5
        new_fatigue = min(100, fatigue_before + fatigue_increase)
        self.profile['fatigue_score'] = new_fatigue
        
        # Update peak hours if provided
        if current_hour is not None:
            peak_hours_key = f"{task_type}_peak_hours"
            peak_hours = self.profile[peak_hours_key]
            
            # Add current hour if not already in peak hours and limit to 5
            if current_hour not in peak_hours:
                peak_hours.append(current_hour)
                # Keep only most recent 5 peak hours
                if len(peak_hours) > 5:
                    peak_hours.pop(0)
        
        # Reset consecutive failures
        self.profile['consecutive_failures'] = 0
        
        # Increment tasks completed
        self.profile['tasks_completed_today'] += 1
        
        # Add to history
        self.task_history.append({
            'timestamp': datetime.now(),
            'task_type': task_type,
            'difficulty': task['difficulty'],
            'estimated_duration': estimated,
            'actual_duration': actual_time,
            'outcome': 'completed',
            'feedback': [],
            'fatigue_before': fatigue_before,
            'fatigue_after': new_fatigue
        })
        
        # Determine change direction
        change = calculate_efficiency_change(old_multiplier, new_multiplier)
        
        return {
            'old_multiplier': old_multiplier,
            'new_multiplier': new_multiplier,
            'change': change,
            'success_rate': new_success_rate
        }
    
    def update_on_failure(
        self,
        task: Dict,
        feedback: List[str],
        actual_time: int,
        current_hour: Optional[int] = None
    ) -> Dict:
        """
        Update profile when task fails.
        
        Args:
            task: Dict with keys {task_type, difficulty, cognitive_load, estimated_duration}
            feedback: List of feedback strings from user
            actual_time: Minutes user spent before giving up
            current_hour: Hour when task was attempted (0-23), optional
        
        Returns:
            Dict with update details including multiplier changes
        """
        task_type = task['task_type']
        estimated = task['estimated_duration']
        
        # Validate task type
        if task_type not in self.VALID_TASK_TYPES:
            return {'error': f'Invalid task_type: {task_type}'}
        
        # Get current values
        multiplier_key = f"{task_type}_multiplier"
        success_rate_key = f"{task_type}_success_rate"
        
        old_multiplier = self.profile[multiplier_key]
        old_success_rate = self.profile[success_rate_key]
        
        # Update multiplier based on feedback
        new_multiplier = old_multiplier
        
        if 'time_too_less' in feedback:
            # Aggressive increase: 10% bump
            new_multiplier = old_multiplier * 1.1
            
        elif actual_time > estimated * 0.5:
            # User tried but couldn't finish
            # Conservative estimate: assume 30% more time needed
            observed = actual_time / estimated if estimated > 0 else 1.0
            conservative_estimate = observed * 1.3
            
            # Update using EMA
            new_multiplier = (
                self.ema_beta * old_multiplier +
                (1 - self.ema_beta) * conservative_estimate
            )
        
        # Clip to safety bounds
        new_multiplier = max(0.3, min(2.0, new_multiplier))
        self.profile[multiplier_key] = new_multiplier
        
        # Update success rate (EMA with 0.0 since failed)
        new_success_rate = (
            self.ema_beta * old_success_rate +
            (1 - self.ema_beta) * 0.0
        )
        self.profile[success_rate_key] = new_success_rate
        
        # Apply fatigue PENALTY (failures are more draining)
        fatigue_before = self.profile['fatigue_score']
        fatigue_penalty = task['cognitive_load'] * 6 # 2.5x normal
        new_fatigue = min(100, fatigue_before + fatigue_penalty)
        self.profile['fatigue_score'] = new_fatigue
        
        # Lower break threshold if too tired
        if 'too_tired' in feedback:
            self.profile['break_threshold_fatigue'] = max(
                40,  # Don't go below 40
                self.profile['break_threshold_fatigue'] - 5
            )
        
        # Remove current hour from peak hours if wrong time
        if 'wrong_time' in feedback and current_hour is not None:
            peak_hours_key = f"{task_type}_peak_hours"
            peak_hours = self.profile[peak_hours_key]
            if current_hour in peak_hours:
                peak_hours.remove(current_hour)
        
        # Increment consecutive failures
        self.profile['consecutive_failures'] += 1
        
        # Add to history
        self.task_history.append({
            'timestamp': datetime.now(),
            'task_type': task_type,
            'difficulty': task['difficulty'],
            'estimated_duration': estimated,
            'actual_duration': actual_time,
            'outcome': 'failed',
            'feedback': feedback,
            'fatigue_before': fatigue_before,
            'fatigue_after': new_fatigue
        })
        
        return {
            'old_multiplier': old_multiplier,
            'new_multiplier': new_multiplier,
            'multiplier_increased_by': (new_multiplier - old_multiplier) / old_multiplier,
            'success_rate': new_success_rate,
            'fatigue_penalty': fatigue_penalty
        }
    
    def get_efficiency_bucket(self, task_type: str) -> int:
        """
        Convert continuous multiplier to discrete bucket for RL state.
        
        Args:
            task_type: Task type to check efficiency for
        
        Returns:
            0 = fast (multiplier < 0.85)
            1 = normal (0.85 <= multiplier <= 1.15)
            2 = slow (multiplier > 1.15)
        """
        if task_type not in self.VALID_TASK_TYPES:
            return 1  # Default to normal
        
        multiplier_key = f"{task_type}_multiplier"
        multiplier = self.profile.get(multiplier_key, 1.0)
        
        if multiplier < 0.85:
            return 0  # Fast
        elif multiplier <= 1.15:
            return 1  # Normal
        else:
            return 2  # Slow
    
    def is_peak_hour(self, task_type: str, hour: int) -> bool:
        """
        Check if given hour is in peak hours for this task type.
        
        Args:
            task_type: Task type to check
            hour: Hour of day (0-23)
        
        Returns:
            True if hour in peak_hours list for this task_type
        """
        if task_type not in self.VALID_TASK_TYPES:
            return False
        
        peak_hours_key = f"{task_type}_peak_hours"
        peak_hours = self.profile.get(peak_hours_key, [])
        
        return hour in peak_hours
    
    def should_force_break(self) -> bool:
        """
        Determine if user MUST take a break based on fatigue or failure count.
        
        Returns:
            True if fatigue >= threshold OR consecutive_failures >= 3
        """
        fatigue_check = (
            self.profile['fatigue_score'] >= 
            self.profile['break_threshold_fatigue']
        )
        
        failure_check = self.profile['consecutive_failures'] >= 3
        
        return fatigue_check or failure_check
    
    def apply_break(self, duration_minutes: int) -> Dict:
        """
        Apply break and reduce fatigue accordingly.
        
        Args:
            duration_minutes: Length of break taken
        
        Returns:
            Dict with fatigue before/after and reduction amount
        """
        fatigue_before = self.profile['fatigue_score']
        
        # Calculate fatigue reduction (1 point per 3 minutes)
        fatigue_reduction = duration_minutes / 3
        
        # Apply reduction
        fatigue_after = max(0, fatigue_before - fatigue_reduction)
        self.profile['fatigue_score'] = fatigue_after
        
        return {
            'fatigue_before': fatigue_before,
            'fatigue_after': fatigue_after,
            'fatigue_reduced': fatigue_reduction
        }
    
    def simulate_daily_reset(self) -> None:
        """
        Reset daily counters to simulate overnight recovery (sleep).
        
        Called when user clicks "Simulate Sleep" or new day detected.
        
        Logic:
        - Reset fatigue to 20 (fresh but not 0)
        - Reset daily counters
        - Slightly improve efficiency (recovery effect)
        """
        # Reset fatigue and counters
        self.profile['fatigue_score'] = 20.0
        self.profile['tasks_completed_today'] = 0
        self.profile['consecutive_failures'] = 0
        
        # Slight efficiency improvement (move toward 1.0)
        for task_type in self.VALID_TASK_TYPES:
            multiplier_key = f"{task_type}_multiplier"
            old_multiplier = self.profile[multiplier_key]
            
            # Move 5% closer to 1.0, but cap at 1.0
            if old_multiplier > 1.0:
                new_multiplier = max(1.0, old_multiplier - 0.05)
            else:
                new_multiplier = min(1.0, old_multiplier + 0.05)
            
            self.profile[multiplier_key] = new_multiplier
    
    def get_summary_stats(self) -> Dict:
        """
        Get comprehensive statistics for display in UI.
        
        Returns:
            Dict with efficiency profile, current state, peak hours, and history
        """
        # Build efficiency profile
        efficiency_profile = {}
        for task_type in self.VALID_TASK_TYPES:
            multiplier_key = f"{task_type}_multiplier"
            success_rate_key = f"{task_type}_success_rate"
            
            multiplier = self.profile[multiplier_key]
            
            efficiency_profile[task_type] = {
                'multiplier': multiplier,
                'bucket': get_bucket_label(multiplier),
                'success_rate': self.profile[success_rate_key]
            }
        
        # Current state
        current_state = {
            'fatigue': self.profile['fatigue_score'],
            'tasks_today': self.profile['tasks_completed_today'],
            'consecutive_failures': self.profile['consecutive_failures'],
            'needs_break': self.should_force_break()
        }
        
        # Peak hours
        peak_hours = {}
        for task_type in self.VALID_TASK_TYPES:
            peak_hours_key = f"{task_type}_peak_hours"
            peak_hours[task_type] = self.profile[peak_hours_key]
        
        # Task history summary
        total_tasks = len(self.task_history)
        completed = sum(1 for t in self.task_history if t['outcome'] == 'completed')
        failed = total_tasks - completed
        completion_rate = completed / total_tasks if total_tasks > 0 else 0
        
        # Calculate average efficiency
        if completed > 0:
            efficiency_sum = sum(
                t['actual_duration'] / t['estimated_duration']
                for t in self.task_history
                if t['outcome'] == 'completed' and t['estimated_duration'] > 0
            )
            avg_efficiency = efficiency_sum / completed
        else:
            avg_efficiency = 1.0
        
        task_history_summary = {
            'total_tasks': total_tasks,
            'completed': completed,
            'failed': failed,
            'completion_rate': completion_rate,
            'avg_efficiency': avg_efficiency
        }
        
        return {
            'efficiency_profile': efficiency_profile,
            'current_state': current_state,
            'peak_hours': peak_hours,
            'task_history_summary': task_history_summary
        }
    
    def save_profile(self, filepath: str = 'user_profile.json') -> None:
        """
        Save profile to JSON file.
        
        Args:
            filepath: Path to save file
        """
        # Prepare data for JSON serialization
        data = {
            'profile': self.profile,
            'ema_beta': self.ema_beta,
            'task_history': []
        }
        
        # Convert datetime objects to ISO strings
        for task in self.task_history:
            task_copy = task.copy()
            task_copy['timestamp'] = task['timestamp'].isoformat()
            data['task_history'].append(task_copy)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving profile to {filepath}: {e}")
    
    def load_profile(self, filepath: str = 'user_profile.json') -> bool:
        """
        Load profile from JSON file.
        
        Args:
            filepath: Path to load file
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(filepath):
                return False
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Load profile
            self.profile = data['profile']
            self.ema_beta = data.get('ema_beta', 0.8)
            
            # Load task history and convert timestamps back
            self.task_history = []
            for task in data.get('task_history', []):
                task_copy = task.copy()
                task_copy['timestamp'] = datetime.fromisoformat(task['timestamp'])
                self.task_history.append(task_copy)
            
            return True
            
        except Exception as e:
            print(f"Error loading profile from {filepath}: {e}")
            return False
    
    def export_history_csv(self, filepath: str = 'task_history.csv') -> None:
        """
        Export task history to CSV for external analysis.
        
        Args:
            filepath: Path to save CSV file
        """
        try:
            import csv
            
            if not self.task_history:
                print("No task history to export")
                return
            
            # Get all keys from first entry
            fieldnames = list(self.task_history[0].keys())
            
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for task in self.task_history:
                    row = task.copy()
                    # Convert datetime to string
                    row['timestamp'] = task['timestamp'].isoformat()
                    # Convert list to string
                    row['feedback'] = '; '.join(task['feedback'])
                    writer.writerow(row)
            
            print(f"✓ Exported {len(self.task_history)} tasks to {filepath}")
            
        except Exception as e:
            print(f"Error exporting history to {filepath}: {e}")
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        stats = self.get_summary_stats()
        return (f"UserProfileManager("
                f"fatigue={stats['current_state']['fatigue']:.1f}, "
                f"tasks_today={stats['current_state']['tasks_today']}, "
                f"tasks_total={stats['task_history_summary']['total_tasks']})")


# ============================================================================
# TESTING AND DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("TESTING USER PROFILE MANAGER")
    print("=" * 80)
    
    # Initialize profile
    manager = UserProfileManager()
    print(f"\nInitialized: {manager}\n")
    
    # ========================================
    # TEST 1: Successful task (early completion)
    # ========================================
    print("-" * 80)
    print("TEST 1: Successful Problem-Solving Task (Early Completion)")
    print("-" * 80)
    
    task1 = {
        'task_type': 'problem_solving',
        'difficulty': 'medium',
        'cognitive_load': 7,
        'estimated_duration': 90
    }
    
    print(f"Task: {task1}")
    print(f"Estimated: 90 min, Actual: 70 min (faster!)")
    
    # User completed in 70 minutes (faster than 90 min estimate)
    result1 = manager.update_on_completion(task1, actual_time=70, current_hour=14)
    
    print(f"\nUpdate Results:")
    print(f"  Old multiplier: {result1['old_multiplier']:.3f}")
    print(f"  New multiplier: {result1['new_multiplier']:.3f} ({result1['change']})")
    print(f"  Success rate: {result1['success_rate']:.2%}")
    print(f"  New fatigue: {manager.profile['fatigue_score']:.1f}")
    
    # Get adjusted duration for next similar task
    adjusted = manager.get_adjusted_duration('problem_solving', 90)
    print(f"\nNext time, 90 min task will be scheduled as: {adjusted} min")
    
    # ========================================
    # TEST 2: Failed task with insufficient time
    # ========================================
    print("\n" + "-" * 80)
    print("TEST 2: Failed Theory Task (Insufficient Time)")
    print("-" * 80)
    
    task2 = {
        'task_type': 'theory',
        'difficulty': 'hard',
        'cognitive_load': 9,
        'estimated_duration': 90
    }
    
    print(f"Task: {task2}")
    print(f"User gave up after 60 min, feedback: time_too_less, too_tired")
    
    result2 = manager.update_on_failure(
        task2,
        feedback=['time_too_less', 'too_tired'],
        actual_time=60,
        current_hour=21
    )
    
    print(f"\nUpdate Results:")
    print(f"  Old multiplier: {result2['old_multiplier']:.3f}")
    print(f"  New multiplier: {result2['new_multiplier']:.3f}")
    print(f"  Increased by: {result2['multiplier_increased_by']:.1%}")
    print(f"  Success rate: {result2['success_rate']:.2%}")
    print(f"  New fatigue: {manager.profile['fatigue_score']:.1f}")
    print(f"  Break threshold lowered to: {manager.profile['break_threshold_fatigue']:.1f}")
    
    # Check if break needed
    if manager.should_force_break():
        print("\n⚠️  MANDATORY BREAK REQUIRED!")
    
    # ========================================
    # TEST 3: Apply break
    # ========================================
    print("\n" + "-" * 80)
    print("TEST 3: Taking a Break")
    print("-" * 80)
    
    break_result = manager.apply_break(30)
    print(f"30-minute break taken:")
    print(f"  Fatigue before: {break_result['fatigue_before']:.1f}")
    print(f"  Fatigue after: {break_result['fatigue_after']:.1f}")
    print(f"  Reduced by: {break_result['fatigue_reduced']:.1f}")
    
    # ========================================
    # TEST 4: Another successful task
    # ========================================
    print("\n" + "-" * 80)
    print("TEST 4: Another Problem-Solving Task (Verify Learning)")
    print("-" * 80)
    
    task3 = {
        'task_type': 'problem_solving',
        'difficulty': 'hard',
        'cognitive_load': 8,
        'estimated_duration': 120
    }
    
    print(f"Task: {task3}")
    # User completed even faster this time (efficiency improving!)
    result3 = manager.update_on_completion(task3, actual_time=80, current_hour=15)
    
    print(f"\nUpdate Results:")
    print(f"  Old multiplier: {result3['old_multiplier']:.3f}")
    print(f"  New multiplier: {result3['new_multiplier']:.3f} ({result3['change']})")
    print(f"  Success rate: {result3['success_rate']:.2%}")
    
    adjusted2 = manager.get_adjusted_duration('problem_solving', 90)
    print(f"\nNow, 90 min task will be scheduled as: {adjusted2} min")
    print(f"  Improvement from initial: {90 - adjusted2} min saved!")
    
    # ========================================
    # TEST 5: Efficiency profile summary
    # ========================================
    print("\n" + "-" * 80)
    print("TEST 5: Efficiency Profile Summary")
    print("-" * 80)
    
    stats = manager.get_summary_stats()
    
    print("\nLearned Efficiency (Multipliers):")
    for task_type, data in stats['efficiency_profile'].items():
        bucket_emoji = "🚀" if data['bucket'] == 'fast' else "🐢" if data['bucket'] == 'slow' else "➡️"
        print(f"  {bucket_emoji} {task_type:20s}: {data['multiplier']:.2f}x ({data['bucket']:8s}) | Success: {data['success_rate']:.0%}")
    
    print("\nCurrent State:")
    for key, value in stats['current_state'].items():
        print(f"  {key}: {value}")
    
    print("\nPeak Hours by Task Type:")
    for task_type, hours in stats['peak_hours'].items():
        print(f"  {task_type:20s}: {hours}")
    
    print("\nTask History Summary:")
    for key, value in stats['task_history_summary'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2%}" if 0 <= value <= 1 else f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    # ========================================
    # TEST 6: Save and load
    # ========================================
    print("\n" + "-" * 80)
    print("TEST 6: Save and Load Profile")
    print("-" * 80)
    
    manager.save_profile('test_profile.json')
    print("✓ Profile saved to test_profile.json")
    
    # Create new manager and load
    manager2 = UserProfileManager()
    success = manager2.load_profile('test_profile.json')
    print(f"✓ Profile loaded successfully: {success}")
    print(f"  Loaded fatigue: {manager2.profile['fatigue_score']:.1f}")
    print(f"  Loaded problem_solving multiplier: {manager2.profile['problem_solving_multiplier']:.3f}")
    print(f"  Loaded task history entries: {len(manager2.task_history)}")
    
    # Export CSV
    manager.export_history_csv('test_history.csv')
    
    # Cleanup
    if os.path.exists('test_profile.json'):
        os.remove('test_profile.json')
    if os.path.exists('test_history.csv'):
        os.remove('test_history.csv')
    
    print("\n" + "=" * 80)
    print("KEY FEATURES DEMONSTRATED:")
    print("=" * 80)
    print("✓ Bidirectional learning: Updates from BOTH early and late completion")
    print("✓ Task-type specific efficiency: Different multipliers for theory vs coding")
    print("✓ Exponential moving average: Stable learning with 80/20 weighting")
    print("✓ Aggressive updates on failure: 10% bump when 'time_too_less' feedback")
    print("✓ Fatigue management: Penalty for failures, recovery from breaks")
    print("✓ Peak hours learning: Tracks when user performs best per task type")
    print("✓ Persistent storage: Save/load profile with complete history")
    print("=" * 80)
