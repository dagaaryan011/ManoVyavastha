"""
================================================================================
FILE: llm_task_decomposer.py
COMPONENT: LLM Task Decomposer - Intelligent Goal Breakdown
================================================================================

Takes user's high-level goal (e.g., "Prepare for Data Structures exam in 3 days")
and breaks it down into 5-7 actionable sub-tasks with detailed parameters.

Uses Gemini API (with rule-based fallback) to:
1. Identify key topics/concepts from user goal
2. Generate logical learning sequence with dependencies
3. Assign cognitive parameters (load, difficulty, duration)
4. Prioritize based on urgency and importance

Key Innovation: Domain-Aware Decomposition
-------------------------------------------
The LLM understands academic domains and creates realistic learning paths:
- Theory concepts before problem-solving practice
- Easy topics before hard ones
- Revision after initial learning
- Mock exams at the end
"""

import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import re
from collections import Counter


# ============================================================================
# CONFIGURATION
# ============================================================================
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
DEFAULT_GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Groq Configuration (uses Llama 3.1 70B - very fast and capable)
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def assign_task_ids(tasks: List[Dict], subject_prefix: str = "TASK") -> List[Dict]:
    """
    Assign unique task_ids if not already present.
    
    Args:
        tasks: List of task dicts
        subject_prefix: Prefix for IDs (e.g., "DS", "OS")
    
    Returns:
        Updated task list with task_id field
    """
    for i, task in enumerate(tasks, 1):
        if 'task_id' not in task or not task['task_id']:
            task['task_id'] = f"{subject_prefix}_{i:03d}"
    return tasks


def infer_subject_prefix(query: str) -> str:
    """
    Infer subject prefix from query for task ID generation.
    
    Args:
        query: User's goal text
    
    Returns:
        2-4 letter prefix (e.g., "DS", "OS", "ML")
    """
    query_lower = query.lower()
    
    # Subject keyword mapping
    subject_map = {
        'ds': ['data structures', 'dsa', ' ds '],
        'algo': ['algorithms', 'algo'],
        'os': ['operating systems', 'operating system', ' os '],
        'db': ['database', 'dbms', 'sql'],
        'ml': ['machine learning', 'deep learning', ' ml '],
        'cn': ['networks', 'networking', 'computer networks', ' cn '],
        'web': ['web dev', 'html', 'css', 'javascript', 'react'],
        'ai': ['artificial intelligence', ' ai '],
        'se': ['software engineering', 'software dev'],
        'py': ['python'],
        'java': ['java']
    }
    
    for prefix, keywords in subject_map.items():
        if any(kw in query_lower for kw in keywords):
            return prefix.upper()
    
    return "TASK"  # Generic prefix


def prioritize_by_deadline(tasks: List[Dict], deadline_days: int) -> List[Dict]:
    """
    Adjust task priorities based on deadline urgency.
    
    Args:
        tasks: List of task dicts
        deadline_days: Days until deadline
    
    Returns:
        Updated tasks with adjusted priorities
    """
    if deadline_days <= 1:
        # Everything is urgent
        for task in tasks:
            task['priority'] = 'urgent'
    
    elif deadline_days <= 2:
        # First half urgent, rest high
        midpoint = len(tasks) // 2
        for i, task in enumerate(tasks):
            task['priority'] = 'urgent' if i < midpoint else 'high'
    
    elif deadline_days <= 5:
        # Theory/practice high, revision medium
        for task in tasks:
            if task['task_type'] in ['theory', 'problem_solving', 'exam_simulation']:
                task['priority'] = 'high'
            else:
                task['priority'] = 'medium'
    
    # Otherwise, keep original priorities
    return tasks


# ============================================================================
# LLM TASK DECOMPOSER
# ============================================================================

class LLMTaskDecomposer:
    """
    Decomposes high-level user goals into structured sub-tasks.
    
    Uses Gemini API for intelligent decomposition, with comprehensive
    rule-based fallback when API is unavailable.
    
    Attributes:
        api_key: Gemini API key (uses hardcoded default if None)
        gemini_endpoint: Gemini API endpoint URL
        fallback_templates: Pre-defined task templates for common subjects
        timeout: API request timeout in seconds
    """
    
    VALID_TASK_TYPES = ['theory', 'problem_solving', 'revision', 'exam_simulation', 'reading']
    VALID_DIFFICULTIES = ['easy', 'medium', 'hard']
    VALID_PRIORITIES = ['low', 'medium', 'high', 'urgent']
    VALID_FLEXIBILITY = ['atomic', 'splittable']
    
    def __init__(self, api_key: Optional[str] = None, timeout: int = 30):
        """
        Initialize task decomposer with Groq API.
        
        Args:
            api_key: Groq API key (uses DEFAULT_GROQ_API_KEY if None)
            timeout: API request timeout in seconds
        """
        self.api_key = api_key or DEFAULT_GROQ_API_KEY
        self.groq_endpoint = GROQ_ENDPOINT
        self.model = GROQ_MODEL
        self.timeout = timeout
        self.fallback_templates = self._create_fallback_templates()
    
    def decompose_goal(
        self,
        user_query: str,
        deadline_days: int = 3,
        current_time: Optional[datetime] = None
    ) -> Dict:
        """
        Main method: Decompose user goal into sub-tasks.
        
        Args:
            user_query: User's goal description
            deadline_days: Days until deadline
            current_time: Current datetime (default: now)
        
        Returns:
            Dict with goal, deadline, sub_tasks, total hours, and generation method
        """
        if current_time is None:
            current_time = datetime.now()
        
        deadline_date = current_time + timedelta(days=deadline_days)
        
        # Try Groq API first if key is available
        result = None
        generation_method = 'fallback'
        
        if self.api_key:
            try:
                result = self._call_groq_api(user_query, deadline_days, current_time)
                if result:
                    generation_method = 'groq_api'
            except Exception as e:
                print(f"Groq API call failed: {e}")
                print("Falling back to rule-based generation...")
        
        # Fallback if API failed or no API key
        if not result:
            result = self._generate_fallback_tasks(user_query, deadline_days, current_time)
            generation_method = 'fallback'
        
        # Validate and post-process
        if not self._validate_tasks(result.get('sub_tasks', [])):
            print("Warning: Some tasks failed validation")
        
        # Calculate total time
        total_hours = self.estimate_total_time(result['sub_tasks'])
        
        return {
            'goal': result.get('goal', user_query),
            'deadline_date': deadline_date.strftime('%Y-%m-%d'),
            'sub_tasks': result['sub_tasks'],
            'total_estimated_hours': total_hours,
            'generation_method': generation_method
        }
    
    def _call_groq_api(
        self,
        user_query: str,
        deadline_days: int,
        current_time: datetime
    ) -> Optional[Dict]:
        """
        Call Groq API to generate task decomposition.
        
        Args:
            user_query: User's goal
            deadline_days: Days until deadline
            current_time: Current datetime
        
        Returns:
            Dict with sub_tasks or None if API call fails
        """
        prompt = self._build_groq_prompt(user_query, deadline_days, current_time)
        
        # Groq uses OpenAI-compatible API format
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert study planner. Generate task breakdowns in valid JSON format only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 2048
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            response = requests.post(
                self.groq_endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                print(f"Groq API returned status {response.status_code}: {response.text}")
                return None
            
            data = response.json()
            
            # Extract text from response
            if 'choices' not in data or not data['choices']:
                return None
            
            text = data['choices'][0]['message']['content']
            
            # Parse JSON from text
            result = self._extract_json_from_text(text)
            
            if result and 'sub_tasks' in result:
                return result
            
            return None
            
        except Exception as e:
            print(f"Error calling Groq API: {e}")
            return None
    
    def _build_groq_prompt(self, user_query: str, deadline_days: int, current_time: datetime) -> str:
        today = current_time.strftime('%Y-%m-%d')
        deadline_date = (current_time + timedelta(days=deadline_days)).strftime('%Y-%m-%d')
        
        prompt = f"""
You are an intelligent Task & Study Planner. Your goal is to break down a User Goal into actionable sub-tasks.
You must adapt your granularity based on the complexity of the request.

USER GOAL: "{user_query}"
CURRENT CONTEXT: Today is {today}. Deadline is {deadline_days} days ({deadline_date}).

---
### 🧠 ANALYSIS INSTRUCTIONS

**1. Determine Granularity (CRITICAL):**
   * **TYPE A: Simple Event / Errands** (e.g., "Gym at 5", "Buy groceries", "Call Mom")
       * Generate **1-2 atomic tasks**.
       * Set `task_type` to "other".
       * Set `flexibility` to "atomic".
   * **TYPE B: Complex Project / Study Goal** (e.g., "Study for Data Structures", "Build a React App")
       * Generate **5-7 structured tasks**.
       * Follow a learning path: Theory → Practice → Revision.
       * Set `flexibility` to "splittable".

**2. Time & Urgency Extraction:**
   * If the user specifies a time (e.g., "at 5 PM", "in the morning"), **APPEND** it to the `task_name`.
   * If a specific time is mentioned, set `priority` to "urgent".

**3. Parameter Guidelines:**
   * **Cognitive Load (1-10):**
       * 1-3: Chores, errands, simple reading.
       * 4-7: Standard study, coding practice, workouts.
       * 8-10: Deep focus, exams, complex system design.
   * **Duration:** Be realistic. Gym = 60-90m, Study Session = 45-120m.

---
### 📋 JSON OUTPUT FORMAT
Return **ONLY valid JSON** with no markdown or explanations.

{{
    "goal": "{user_query}",
    "deadline_date": "{deadline_date}",
    "sub_tasks": [
        {{
            "task_id": "WILL_BE_GENERATED_BY_SYSTEM",
            "task_name": "Clear description (e.g., 'Leg Workout (5 PM)')",
            "task_type": "theory" | "problem_solving" | "revision" | "exam_simulation" | "reading" | "other",
            "difficulty": "easy" | "medium" | "hard",
            "cognitive_load": 1-10,
            "estimated_duration": 60,
            "prerequisites": [], 
            "priority": "low" | "medium" | "high" | "urgent",
            "flexibility": "atomic" | "splittable",
            "notes": "Brief context or motivation"
        }}
    ]
}}
"""
        return prompt
    
    def _generate_fallback_tasks(
        self,
        user_query: str,
        deadline_days: int,
        current_time: datetime
    ) -> Dict:
        """
        Rule-based task generation when API is unavailable.
        
        Args:
            user_query: User's goal
            deadline_days: Days until deadline
            current_time: Current datetime
        
        Returns:
            Dict with same structure as API response
        """
        # Extract subjects from query
        subjects = self._extract_subjects(user_query)
        
        # Get subject prefix for task IDs
        subject_prefix = infer_subject_prefix(user_query)
        
        tasks = []
        
        if subjects:
            # Generate tasks from templates for detected subjects
            for subject in subjects[:2]:  # Limit to 2 subjects max
                subject_tasks = self.fallback_templates.get(subject, [])
                
                for template in subject_tasks[:3]:  # Max 3 tasks per subject
                    task = template.copy()
                    task['task_name'] = task['task_name'].format(subject=subject.title())
                    tasks.append(task)
        
        else:
            # Generic tasks for vague queries
            tasks = [
                {
                    'task_name': 'Study Session 1 - Core Concepts',
                    'task_type': 'theory',
                    'difficulty': 'medium',
                    'cognitive_load': 6,
                    'estimated_duration': 90,
                    'prerequisites': [],
                    'priority': 'high',
                    'flexibility': 'splittable',
                    'notes': 'Focus on fundamental concepts'
                },
                {
                    'task_name': 'Practice Problems - Basic Level',
                    'task_type': 'problem_solving',
                    'difficulty': 'easy',
                    'cognitive_load': 5,
                    'estimated_duration': 60,
                    'prerequisites': [],
                    'priority': 'medium',
                    'flexibility': 'splittable',
                    'notes': 'Build confidence with easier problems'
                },
                {
                    'task_name': 'Study Session 2 - Advanced Topics',
                    'task_type': 'theory',
                    'difficulty': 'hard',
                    'cognitive_load': 8,
                    'estimated_duration': 120,
                    'prerequisites': [],
                    'priority': 'high',
                    'flexibility': 'splittable',
                    'notes': 'Deep dive into complex topics'
                },
                {
                    'task_name': 'Practice Problems - Intermediate Level',
                    'task_type': 'problem_solving',
                    'difficulty': 'medium',
                    'cognitive_load': 7,
                    'estimated_duration': 90,
                    'prerequisites': [],
                    'priority': 'medium',
                    'flexibility': 'splittable',
                    'notes': 'Apply learned concepts'
                },
                {
                    'task_name': 'Comprehensive Revision',
                    'task_type': 'revision',
                    'difficulty': 'medium',
                    'cognitive_load': 5,
                    'estimated_duration': 60,
                    'prerequisites': [],
                    'priority': 'medium',
                    'flexibility': 'splittable',
                    'notes': 'Review all covered material'
                },
                {
                    'task_name': 'Full Mock Test',
                    'task_type': 'exam_simulation',
                    'difficulty': 'hard',
                    'cognitive_load': 10,
                    'estimated_duration': 120,
                    'prerequisites': [],
                    'priority': 'high',
                    'flexibility': 'atomic',
                    'notes': 'Simulate real exam conditions'
                }
            ]
        
        # Limit to 5-7 tasks
        tasks = tasks[:7]
        
        # Assign task IDs
        tasks = assign_task_ids(tasks, subject_prefix)
        
        # Adjust priorities based on deadline
        tasks = prioritize_by_deadline(tasks, deadline_days)
        
        # Extract goal from query
        goal = user_query.strip()
        
        return {
            'goal': goal,
            'sub_tasks': tasks
        }
    
    def _extract_subjects(self, query: str) -> List[str]:
        """
        Extract subject names from user query using keyword matching.
        
        Args:
            query: User's goal text
        
        Returns:
            List of detected subjects
        """
        query_lower = query.lower()
        
        # Keyword dictionary for subject detection
        subject_keywords = {
            'data structures': ['data structures', 'data structure', 'dsa', ' ds '],
            'algorithms': ['algorithms', 'algorithm', 'algo'],
            'operating systems': ['operating systems', 'operating system', ' os '],
            'database': ['database', 'dbms', 'sql', 'mysql', 'postgresql'],
            'machine learning': ['machine learning', 'ml', 'deep learning', 'neural network'],
            'networks': ['networks', 'networking', 'computer networks', ' cn '],
            'web development': ['web dev', 'html', 'css', 'javascript', 'react', 'node'],
            'artificial intelligence': ['artificial intelligence', ' ai ', 'expert systems'],
            'software engineering': ['software engineering', 'sdlc', 'agile'],
            'python': ['python'],
            'java': ['java'],
            'c++': ['c++', 'cpp']
        }
        
        detected = []
        for subject, keywords in subject_keywords.items():
            if any(kw in query_lower for kw in keywords):
                detected.append(subject)
        
        return detected
    
    def _create_fallback_templates(self) -> Dict[str, List[Dict]]:
        """
        Create pre-defined task templates for common subjects.
        
        Returns:
            Dict mapping subjects to task templates
        """
        templates = {
            'data structures': [
                {
                    'task_name': 'Learn {subject} - Arrays and Linked Lists',
                    'task_type': 'theory',
                    'difficulty': 'medium',
                    'cognitive_load': 6,
                    'estimated_duration': 90,
                    'prerequisites': [],
                    'priority': 'high',
                    'flexibility': 'splittable',
                    'notes': 'Foundation data structures'
                },
                {
                    'task_name': 'Practice {subject} - Array and List Problems',
                    'task_type': 'problem_solving',
                    'difficulty': 'medium',
                    'cognitive_load': 7,
                    'estimated_duration': 120,
                    'prerequisites': [],
                    'priority': 'high',
                    'flexibility': 'splittable',
                    'notes': 'Solve 15-20 problems'
                },
                {
                    'task_name': 'Learn {subject} - Trees and Graphs',
                    'task_type': 'theory',
                    'difficulty': 'hard',
                    'cognitive_load': 8,
                    'estimated_duration': 120,
                    'prerequisites': [],
                    'priority': 'high',
                    'flexibility': 'splittable',
                    'notes': 'Complex but important topics'
                },
                {
                    'task_name': 'Practice {subject} - Tree and Graph Problems',
                    'task_type': 'problem_solving',
                    'difficulty': 'hard',
                    'cognitive_load': 8,
                    'estimated_duration': 150,
                    'prerequisites': [],
                    'priority': 'high',
                    'flexibility': 'splittable',
                    'notes': 'Apply BFS, DFS, traversals'
                },
                {
                    'task_name': '{subject} Mock Test',
                    'task_type': 'exam_simulation',
                    'difficulty': 'hard',
                    'cognitive_load': 10,
                    'estimated_duration': 120,
                    'prerequisites': [],
                    'priority': 'high',
                    'flexibility': 'atomic',
                    'notes': 'Full-length practice exam'
                }
            ],
            'operating systems': [
                {
                    'task_name': 'Learn {subject} - Process Management',
                    'task_type': 'theory',
                    'difficulty': 'medium',
                    'cognitive_load': 7,
                    'estimated_duration': 90,
                    'prerequisites': [],
                    'priority': 'high',
                    'flexibility': 'splittable',
                    'notes': 'Processes, threads, scheduling'
                },
                {
                    'task_name': 'Learn {subject} - Memory Management',
                    'task_type': 'theory',
                    'difficulty': 'hard',
                    'cognitive_load': 8,
                    'estimated_duration': 90,
                    'prerequisites': [],
                    'priority': 'high',
                    'flexibility': 'splittable',
                    'notes': 'Paging, segmentation, virtual memory'
                },
                {
                    'task_name': 'Practice {subject} - Numerical Problems',
                    'task_type': 'problem_solving',
                    'difficulty': 'medium',
                    'cognitive_load': 7,
                    'estimated_duration': 90,
                    'prerequisites': [],
                    'priority': 'medium',
                    'flexibility': 'splittable',
                    'notes': 'Scheduling algorithms, page replacement'
                }
            ],
            'database': [
                {
                    'task_name': 'Learn {subject} - SQL Fundamentals',
                    'task_type': 'theory',
                    'difficulty': 'easy',
                    'cognitive_load': 5,
                    'estimated_duration': 60,
                    'prerequisites': [],
                    'priority': 'high',
                    'flexibility': 'splittable',
                    'notes': 'SELECT, JOIN, aggregations'
                },
                {
                    'task_name': 'Practice {subject} - SQL Queries',
                    'task_type': 'problem_solving',
                    'difficulty': 'medium',
                    'cognitive_load': 6,
                    'estimated_duration': 90,
                    'prerequisites': [],
                    'priority': 'high',
                    'flexibility': 'splittable',
                    'notes': 'Write complex queries'
                },
                {
                    'task_name': 'Learn {subject} - Normalization and Transactions',
                    'task_type': 'theory',
                    'difficulty': 'hard',
                    'cognitive_load': 8,
                    'estimated_duration': 90,
                    'prerequisites': [],
                    'priority': 'medium',
                    'flexibility': 'splittable',
                    'notes': 'ACID, normal forms, concurrency'
                }
            ],
            'machine learning': [
                {
                    'task_name': 'Learn {subject} - Supervised Learning',
                    'task_type': 'theory',
                    'difficulty': 'medium',
                    'cognitive_load': 7,
                    'estimated_duration': 120,
                    'prerequisites': [],
                    'priority': 'high',
                    'flexibility': 'splittable',
                    'notes': 'Regression, classification algorithms'
                },
                {
                    'task_name': 'Practice {subject} - Model Building',
                    'task_type': 'problem_solving',
                    'difficulty': 'hard',
                    'cognitive_load': 8,
                    'estimated_duration': 150,
                    'prerequisites': [],
                    'priority': 'high',
                    'flexibility': 'splittable',
                    'notes': 'Build and evaluate models'
                },
                {
                    'task_name': 'Learn {subject} - Neural Networks',
                    'task_type': 'theory',
                    'difficulty': 'hard',
                    'cognitive_load': 9,
                    'estimated_duration': 120,
                    'prerequisites': [],
                    'priority': 'medium',
                    'flexibility': 'splittable',
                    'notes': 'Deep learning fundamentals'
                }
            ]
        }
        
        return templates
    
    def _validate_tasks(self, tasks: List[Dict]) -> bool:
        """
        Validate task structure and parameters.
        
        Args:
            tasks: List of task dicts
        
        Returns:
            True if all tasks valid
        """
        required_fields = [
            'task_id', 'task_name', 'task_type', 'difficulty',
            'cognitive_load', 'estimated_duration', 'prerequisites',
            'priority', 'flexibility', 'notes'
        ]
        
        for task in tasks:
            # Check required fields
            for field in required_fields:
                if field not in task:
                    print(f"Missing field '{field}' in task: {task.get('task_name', 'unknown')}")
                    return False
            
            # Validate enums
            if task['task_type'] not in self.VALID_TASK_TYPES:
                print(f"Invalid task_type: {task['task_type']}")
                return False
            
            if task['difficulty'] not in self.VALID_DIFFICULTIES:
                print(f"Invalid difficulty: {task['difficulty']}")
                return False
            
            if task['priority'] not in self.VALID_PRIORITIES:
                print(f"Invalid priority: {task['priority']}")
                return False
            
            if task['flexibility'] not in self.VALID_FLEXIBILITY:
                print(f"Invalid flexibility: {task['flexibility']}")
                return False
            
            # Validate ranges
            if not (1 <= task['cognitive_load'] <= 10):
                print(f"cognitive_load out of range: {task['cognitive_load']}")
                return False
            
            if task['estimated_duration'] <= 0:
                print(f"Invalid duration: {task['estimated_duration']}")
                return False
        
        return True
    
    def _extract_json_from_text(self, text: str) -> Optional[Dict]:
        """
        Extract JSON from LLM response that might have markdown formatting.
        
        Args:
            text: Raw LLM response text
        
        Returns:
            Parsed dict or None
        """
        # Try to find JSON in markdown code blocks
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try generic code block
            code_match = re.search(r'```\s*(\{.*?\})\s*```', text, re.DOTALL)
            if code_match:
                json_str = code_match.group(1)
            else:
                # Use entire text
                json_str = text.strip()
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            print(f"Text: {json_str[:200]}...")
            return None
    
    def estimate_total_time(self, tasks: List[Dict]) -> float:
        """
        Calculate total estimated hours for all tasks.
        
        Args:
            tasks: List of task dicts
        
        Returns:
            Total hours (float)
        """
        total_minutes = sum(task['estimated_duration'] for task in tasks)
        return total_minutes / 60.0


# ============================================================================
# TESTING AND DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("TESTING LLM TASK DECOMPOSER")
    print("=" * 80)
    
    # ========================================
    # TEST 1: Fallback mode (no API key)
    # ========================================
    print("\n" + "-" * 80)
    print("TEST 1: Fallback Mode (Rule-Based Generation)")
    print("-" * 80)
    
    decomposer = LLMTaskDecomposer(api_key=None)  # Force fallback
    
    query1 = "I need to prepare for Data Structures exam in 3 days. Focus on Graphs and Dynamic Programming."
    result1 = decomposer.decompose_goal(query1, deadline_days=3)
    
    print(f"\nQuery: {query1}")
    print(f"Generation method: {result1['generation_method']}")
    print(f"Total tasks: {len(result1['sub_tasks'])}")
    print(f"Total estimated time: {result1['total_estimated_hours']:.1f} hours")
    
    print("\nGenerated Sub-Tasks:")
    for i, task in enumerate(result1['sub_tasks'], 1):
        prereq_str = f" [requires: {', '.join(task['prerequisites'])}]" if task['prerequisites'] else ""
        print(f"\n{i}. {task['task_id']}: {task['task_name']}{prereq_str}")
        print(f"   Type: {task['task_type']:15s} | Difficulty: {task['difficulty']:6s} | Load: {task['cognitive_load']}/10")
        print(f"   Duration: {task['estimated_duration']} min | Priority: {task['priority']:6s} | Flexibility: {task['flexibility']}")
        print(f"   Notes: {task['notes']}")
    
    # ========================================
    # TEST 2: Multiple subjects
    # ========================================
    print("\n\n" + "-" * 80)
    print("TEST 2: Multiple Subjects")
    print("-" * 80)
    
    query2 = "Help me study for OS and DBMS exams happening next week"
    result2 = decomposer.decompose_goal(query2, deadline_days=7)
    
    print(f"\nQuery: {query2}")
    print(f"Detected subjects: {decomposer._extract_subjects(query2)}")
    print(f"Total tasks: {len(result2['sub_tasks'])}")
    
    print("\nTask Distribution by Type:")
    type_counts = Counter(t['task_type'] for t in result2['sub_tasks'])
    for task_type, count in type_counts.items():
        print(f"  {task_type:20s}: {count}")
    
    # ========================================
    # TEST 3: Urgent deadline (1 day)
    # ========================================
    print("\n\n" + "-" * 80)
    print("TEST 3: Urgent Deadline (1 Day)")
    print("-" * 80)
    
    query3 = "Cram for Machine Learning exam tomorrow"
    result3 = decomposer.decompose_goal(query3, deadline_days=1)
    
    print(f"\nQuery: {query3}")
    urgent_tasks = [t for t in result3['sub_tasks'] if t['priority'] == 'urgent']
    print(f"Urgent tasks: {len(urgent_tasks)} / {len(result3['sub_tasks'])}")
    print(f"All priorities: {[t['priority'] for t in result3['sub_tasks']]}")
    
    # ========================================
    # TEST 4: Generic query
    # ========================================
    print("\n\n" + "-" * 80)
    print("TEST 4: Generic Query (No Specific Subject)")
    print("-" * 80)
    
    query4 = "I want to study for my upcoming exams"
    result4 = decomposer.decompose_goal(query4, deadline_days=5)
    
    print(f"\nQuery: {query4}")
    print(f"Generated {len(result4['sub_tasks'])} generic tasks")
    print("\nTask names:")
    for task in result4['sub_tasks']:
        print(f"  - {task['task_name']}")
    
    # ========================================
    # TEST 5: Subject prefix inference
    # ========================================
    print("\n\n" + "-" * 80)
    print("TEST 5: Subject Prefix Inference")
    print("-" * 80)
    
    test_queries = [
        "Data Structures exam",
        "Operating Systems project",
        "Machine Learning assignment",
        "Study for exams"
    ]
    
    print("\nInferred prefixes:")
    for query in test_queries:
        prefix = infer_subject_prefix(query)
        print(f"  '{query}' → {prefix}")
    
    print("\n" + "=" * 80)
    print("KEY FEATURES DEMONSTRATED:")
    print("=" * 80)
    print("✓ Groq API integration: Fast LLM-powered task generation")
    print("✓ Fallback mode: Works without API key using rule-based templates")
    print("✓ Multi-subject detection: Identifies multiple topics from query")
    print("✓ Priority adjustment: Urgent deadlines (1 day) → all tasks urgent")
    print("✓ Task templates: Pre-defined structures for common subjects")
    print("✓ Generic handling: Creates reasonable tasks for vague queries")
    print("✓ Subject inference: Extracts topic keywords and assigns prefixes")
    print("=" * 80)
    print("\nNOTE: Update DEFAULT_GROQ_API_KEY at top of file with your key")
    print("Get free key at: https://console.groq.com")
    print("=" * 80)
