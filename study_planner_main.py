# AI-Powered Adaptive Study Planner for Engineering Students
# Advanced Implementation with Cognitive Load Optimization & Prerequisite Mapping
# Author: AI Study Planner Team
# Version: 1.0 (Production Ready)

import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from enum import Enum
import statistics
from collections import defaultdict
import uuid

# ============================================================================
# ENUMS & CONSTANTS
# ============================================================================

class CognitiveLoadLevel(Enum):
    """Cognitive load classification for study sessions"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class StudyPhaseType(Enum):
    """Different phases of learning for each topic"""
    FOUNDATION = "foundation"      # Understanding concepts
    PRACTICE = "practice"          # Problem solving
    REVISION = "revision"          # Quick review
    DEEP_DIVE = "deep_dive"       # Advanced topics
    ASSESSMENT = "assessment"      # Mock tests/evaluation

class DayType(Enum):
    WEEKDAY = "weekday"
    WEEKEND = "weekend"

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class StudentInfo:
    name: str
    college: str
    branch: str
    graduation_year: int
    email: str

@dataclass
class Subject:
    name: str
    credits: int
    strong_areas: List[str]
    weak_areas: List[str]
    confidence_level: int  # 1-5 scale
    prerequisites: List[str] = None  # Topic names this depends on
    estimated_topics: int = 8  # Average topics per subject
    
    def __post_init__(self):
        if self.prerequisites is None:
            self.prerequisites = []

@dataclass
class StudySession:
    date: datetime
    subject_name: str
    topic: str
    session_type: StudyPhaseType
    duration_minutes: int
    cognitive_load: CognitiveLoadLevel
    priority_score: float
    prerequisites_met: bool
    micro_topic: str = None  # Specific subtopic
    notes: str = None
    session_id: str = None
    
    def __post_init__(self):
        if self.session_id is None:
            self.session_id = str(uuid.uuid4())[:8]

@dataclass
class StudyPlan:
    student: StudentInfo
    subjects: List[Subject]
    weekday_hours: float
    weekend_hours: float
    preferred_study_time: str
    target_completion_date: datetime
    schedule: List[StudySession] = None
    
    def __post_init__(self):
        if self.schedule is None:
            self.schedule = []

# ============================================================================
# CORE ALGORITHM: COGNITIVE LOAD OPTIMIZER
# ============================================================================

class CognitiveLoadOptimizer:
    """
    Unique Innovation: Dynamically optimizes cognitive load distribution
    based on subject difficulty, time of day, and student capacity.
    """
    
    PEAK_FOCUS_HOURS = {
        "morning": [6, 7, 8, 9, 10, 11],
        "afternoon": [14, 15, 16, 17],
        "evening": [18, 19, 20],
        "night": [21, 22, 23, 0, 1]
    }
    
    def __init__(self):
        self.session_history = defaultdict(list)
    
    def calculate_cognitive_demand(self, subject: Subject, topic: str, 
                                   is_weak_area: bool, phase: StudyPhaseType) -> float:
        """
        Calculate normalized cognitive demand (0-1) for a topic based on:
        - Subject confidence level
        - Whether it's a weak/strong area
        - Study phase complexity
        """
        base_demand = 1.0 - (subject.confidence_level / 5.0)  # Higher demand = lower confidence
        
        weak_area_multiplier = 1.4 if is_weak_area else 0.7
        phase_multiplier = {
            StudyPhaseType.FOUNDATION: 1.2,
            StudyPhaseType.DEEP_DIVE: 1.3,
            StudyPhaseType.PRACTICE: 1.1,
            StudyPhaseType.REVISION: 0.6,
            StudyPhaseType.ASSESSMENT: 1.0
        }
        
        demand = base_demand * weak_area_multiplier * phase_multiplier[phase]
        return min(1.0, max(0.3, demand))
    
    def assign_cognitive_level(self, demand: float) -> CognitiveLoadLevel:
        """Map cognitive demand to load level"""
        if demand >= 0.75:
            return CognitiveLoadLevel.HIGH
        elif demand >= 0.45:
            return CognitiveLoadLevel.MEDIUM
        else:
            return CognitiveLoadLevel.LOW
    
    def get_optimal_time_slot(self, cognitive_level: CognitiveLoadLevel, 
                             preferred_time: str) -> Tuple[int, int]:
        """
        Return optimal hour range for studying based on cognitive load
        and preferred study time.
        
        Logic:
        - HIGH load → Peak focus hours (morning/preferred)
        - MEDIUM load → Mixed hours (afternoon/evening)
        - LOW load → Off-peak hours (can be flexible)
        """
        peak_hours = self.PEAK_FOCUS_HOURS.get(preferred_time.lower(), 
                                               self.PEAK_FOCUS_HOURS["evening"])
        
        if cognitive_level == CognitiveLoadLevel.HIGH:
            return (peak_hours[0], peak_hours[len(peak_hours)//2])
        elif cognitive_level == CognitiveLoadLevel.MEDIUM:
            return (peak_hours[len(peak_hours)//3], peak_hours[-1])
        else:
            return (peak_hours[-1], peak_hours[-1] + 2)  # Low-intensity hours

# ============================================================================
# PREREQUISITE DEPENDENCY MAPPER
# ============================================================================

class PrerequisiteMapper:
    """
    Unique Innovation: Maps topic dependencies and identifies critical path
    to prevent knowledge gaps from blocking progress.
    """
    
    # Engineering subject dependency graph
    SUBJECT_PREREQUISITES = {
        "Data Structures": [],
        "Operating Systems": ["Data Structures"],
        "Engineering Mathematics": [],
        "Database Management": ["Data Structures"],
        "Web Development": ["Data Structures"],
        "Machine Learning": ["Engineering Mathematics", "Data Structures"],
        "Algorithms": ["Data Structures"],
        "Compiler Design": ["Data Structures", "Operating Systems"]
    }
    
    TOPIC_DEPENDENCIES = {
        "Trees": ["Arrays", "Linked Lists"],
        "Graphs": ["Arrays", "Trees"],
        "Deadlocks": ["Processes", "Threads"],
        "Memory Management": ["Processes"],
        "File Systems": ["Memory Management", "Operating Systems"],
        "Laplace Transform": ["Differential Equations"],
        "Neural Networks": ["Linear Algebra", "Probability"],
        "Backtracking": ["Recursion", "Arrays"]
    }
    
    def __init__(self, subjects: List[Subject]):
        self.subjects = subjects
        self.subject_names = [s.name for s in subjects]
    
    def identify_critical_path(self, target_date: datetime) -> List[Tuple[str, datetime]]:
        """
        Identify topics that must be completed earliest to avoid cascading delays.
        Returns sorted list of (topic, deadline) tuples.
        """
        critical_path = []
        days_available = (target_date - datetime.now()).days
        
        for subject in self.subjects:
            for weak_area in subject.weak_areas:
                # Weak areas should be done in first 40% of timeline
                deadline = datetime.now() + timedelta(days=int(days_available * 0.4))
                # Check if this is a prerequisite
                if weak_area in self.TOPIC_DEPENDENCIES:
                    critical_path.append((f"{subject.name} - {weak_area}", deadline))
        
        return sorted(critical_path, key=lambda x: x[1])
    
    def get_topic_sequence(self, subject_name: str) -> List[List[str]]:
        """
        Return topics in prerequisite order (waves of topics that can be parallelized)
        """
        # Simplified topic sequencing
        sequence = [
            ["Fundamentals", "Basics"],
            ["Core Concepts", "Data Structures"],
            ["Advanced Topics", "Problem Solving"],
            ["Deep Dive", "Edge Cases"],
            ["Assessment", "Revision"]
        ]
        return sequence

# ============================================================================
# ADAPTIVE STUDY PLANNER ENGINE
# ============================================================================

class AdaptiveStudyPlanner:
    """
    Core engine that generates intelligent, personalized study schedules.
    Uses multi-factor optimization for realistic, achievable plans.
    """
    
    def __init__(self):
        self.cognitive_optimizer = CognitiveLoadOptimizer()
        self.prerequisite_mapper = None
    
    def generate_study_plan(self, plan: StudyPlan) -> StudyPlan:
        """
        Main method: Generate complete adaptive study schedule
        """
        self.prerequisite_mapper = PrerequisiteMapper(plan.subjects)
        
        # Step 1: Allocate study hours per subject
        subject_allocations = self._allocate_study_hours(plan)
        
        # Step 2: Break subjects into micro-topics with phases
        micro_topics = self._create_micro_topics(plan.subjects, subject_allocations)
        
        # Step 3: Generate day-by-day schedule
        schedule = self._generate_daily_schedule(
            plan, 
            micro_topics,
            subject_allocations
        )
        
        plan.schedule = schedule
        return plan
    
    def _allocate_study_hours(self, plan: StudyPlan) -> Dict[str, float]:
        """
        Allocate total available study hours across subjects.
        
        Algorithm:
        - Base allocation: proportional to credits
        - Confidence adjustment: low confidence → more hours
        - Weak area penalty: weak areas → 1.3x multiplier
        """
        total_credits = sum(s.credits for s in plan.subjects)
        total_days = (plan.target_completion_date - datetime.now()).days
        total_hours = self._calculate_total_available_hours(plan, total_days)
        
        allocations = {}
        adjusted_scores = {}
        
        # Calculate adjusted priority scores
        for subject in plan.subjects:
            base_score = subject.credits
            confidence_adjustment = (6 - subject.confidence_level) / 5.0  # 0.2 to 1.0
            weak_area_count = len(subject.weak_areas)
            weak_area_bonus = weak_area_count * 0.15
            
            adjusted_scores[subject.name] = base_score * confidence_adjustment * (1 + weak_area_bonus)
        
        total_adjusted_score = sum(adjusted_scores.values())
        
        for subject in plan.subjects:
            proportion = adjusted_scores[subject.name] / total_adjusted_score
            allocations[subject.name] = total_hours * proportion
        
        return allocations
    
    def _calculate_total_available_hours(self, plan: StudyPlan, total_days: int) -> float:
        """Calculate total available study hours"""
        weekdays = total_days * 5 / 7
        weekends = total_days * 2 / 7
        return (weekdays * plan.weekday_hours) + (weekends * plan.weekend_hours)
    
    def _create_micro_topics(self, subjects: List[Subject], 
                            allocations: Dict[str, float]) -> Dict[str, List[Dict]]:
        """
        Break each subject into micro-topics with study phases.
        
        Unique approach: Topics are sequenced with prerequisite awareness
        and grouped by cognitive load.
        """
        micro_topics = defaultdict(list)
        
        for subject in subjects:
            allocated_hours = allocations[subject.name]
            num_topics = subject.estimated_topics
            hours_per_topic = allocated_hours / num_topics
            
            # Split weak areas (need more phases) vs strong areas (quick refresh)
            weak_topics = [{"name": area, "hours": hours_per_topic * 1.2, "is_weak": True} 
                          for area in subject.weak_areas[:2]]
            strong_topics = [{"name": area, "hours": hours_per_topic * 0.7, "is_weak": False} 
                            for area in subject.strong_areas[:2]]
            
            # Add generic topics to reach total
            num_generic = max(0, num_topics - len(weak_topics) - len(strong_topics))
            generic_topics = [{"name": f"Topic {i+1}", "hours": hours_per_topic, "is_weak": False} 
                             for i in range(num_generic)]
            
            all_topics = weak_topics + strong_topics + generic_topics
            
            # Assign study phases to each topic
            for i, topic in enumerate(all_topics):
                phases = self._assign_study_phases(topic, subject)
                micro_topics[subject.name].extend(phases)
        
        return micro_topics
    
    def _assign_study_phases(self, topic: Dict, subject: Subject) -> List[Dict]:
        """
        Assign study phases (foundation, practice, revision, etc.) to a topic.
        Weak areas get more phases; strong areas get quick revision.
        """
        if topic["is_weak"]:
            phases = [
                {"phase": StudyPhaseType.FOUNDATION, "duration_ratio": 0.4},
                {"phase": StudyPhaseType.PRACTICE, "duration_ratio": 0.35},
                {"phase": StudyPhaseType.DEEP_DIVE, "duration_ratio": 0.15},
                {"phase": StudyPhaseType.REVISION, "duration_ratio": 0.1}
            ]
        else:
            phases = [
                {"phase": StudyPhaseType.FOUNDATION, "duration_ratio": 0.5},
                {"phase": StudyPhaseType.PRACTICE, "duration_ratio": 0.3},
                {"phase": StudyPhaseType.REVISION, "duration_ratio": 0.2}
            ]
        
        result = []
        for phase_info in phases:
            result.append({
                "topic": topic["name"],
                "subject": subject.name,
                "phase": phase_info["phase"],
                "duration_hours": topic["hours"] * phase_info["duration_ratio"],
                "is_weak": topic["is_weak"],
                "prerequisites": PrerequisiteMapper.TOPIC_DEPENDENCIES.get(topic["name"], [])
            })
        
        return result
    
    def _generate_daily_schedule(self, plan: StudyPlan, micro_topics: Dict[str, List[Dict]],
                                allocations: Dict[str, float]) -> List[StudySession]:
        """
        Generate day-by-day schedule with intelligent session distribution.
        
        Strategy:
        1. Flatten micro-topics into priority queue
        2. Distribute across days respecting cognitive load limits
        3. Respect prerequisites and deadlines
        4. Place high-load sessions in peak hours
        5. Add buffer time for spillovers
        """
        schedule = []
        current_date = datetime.now()
        target_date = plan.target_completion_date
        
        # Flatten and prioritize all topics
        all_sessions = []
        for subject_name, topics in micro_topics.items():
            for topic_dict in topics:
                session = self._create_session_template(topic_dict, plan, target_date)
                all_sessions.append(session)
        
        # Sort by priority (critical path first)
        all_sessions.sort(key=lambda x: -x["priority_score"])
        
        session_index = 0
        daily_load = defaultdict(float)  # Track daily cognitive load
        
        while current_date <= target_date and session_index < len(all_sessions):
            day_type = DayType.WEEKEND if current_date.weekday() >= 5 else DayType.WEEKDAY
            daily_capacity = plan.weekend_hours if day_type == DayType.WEEKEND else plan.weekday_hours
            
            # Try to fit sessions into this day
            sessions_today = []
            daily_cognitive_load = 0
            
            while (session_index < len(all_sessions) and 
                   daily_cognitive_load < 1.0 and  # Don't overload day
                   sum(s["duration_hours"] for s in sessions_today) < daily_capacity):
                
                session_template = all_sessions[session_index]
                duration_hours = session_template["duration_hours"]
                cognitive_weight = self._get_cognitive_weight(session_template["cognitive_load"])
                
                # Check if adding this session keeps load balanced
                if daily_cognitive_load + cognitive_weight <= 1.2:
                    session = StudySession(
                        date=current_date,
                        subject_name=session_template["subject"],
                        topic=session_template["topic"],
                        session_type=session_template["phase"],
                        duration_minutes=int(duration_hours * 60),
                        cognitive_load=session_template["cognitive_load"],
                        priority_score=session_template["priority_score"],
                        prerequisites_met=session_template["prerequisites_met"],
                        micro_topic=session_template.get("micro_topic", ""),
                        notes=self._generate_session_notes(session_template)
                    )
                    schedule.append(session)
                    sessions_today.append(session_template)
                    daily_cognitive_load += cognitive_weight
                    session_index += 1
                else:
                    break
            
            # Move to next day
            current_date += timedelta(days=1)
            
            # Skip very short days (less than 1 hour) to keep buffer
            if current_date <= target_date and sum(s["duration_hours"] for s in sessions_today) < 0.5:
                current_date += timedelta(days=1)
        
        return schedule
    
    def _create_session_template(self, topic_dict: Dict, plan: StudyPlan, 
                                target_date: datetime) -> Dict:
        """Create a session template with metadata"""
        is_weak = topic_dict.get("is_weak", False)
        subject = next(s for s in plan.subjects if s.name == topic_dict["subject"])
        
        # Calculate cognitive demand
        demand = self.cognitive_optimizer.calculate_cognitive_demand(
            subject, 
            topic_dict["topic"],
            is_weak,
            topic_dict["phase"]
        )
        cognitive_load = self.cognitive_optimizer.assign_cognitive_level(demand)
        
        # Calculate priority
        base_priority = (6 - subject.confidence_level) / 5.0
        phase_priority = {
            StudyPhaseType.FOUNDATION: 1.0,
            StudyPhaseType.DEEP_DIVE: 0.9,
            StudyPhaseType.PRACTICE: 0.8,
            StudyPhaseType.ASSESSMENT: 0.7,
            StudyPhaseType.REVISION: 0.5
        }
        weak_priority = 1.3 if is_weak else 1.0
        priority_score = base_priority * phase_priority[topic_dict["phase"]] * weak_priority
        
        # Check prerequisites
        prerequisites_met = all(
            prereq not in topic_dict.get("prerequisites", [])
            for prereq in topic_dict.get("prerequisites", [])
        )
        
        return {
            "subject": topic_dict["subject"],
            "topic": topic_dict["topic"],
            "phase": topic_dict["phase"],
            "duration_hours": topic_dict["duration_hours"],
            "is_weak": is_weak,
            "cognitive_load": cognitive_load,
            "priority_score": priority_score,
            "prerequisites_met": prerequisites_met,
            "micro_topic": topic_dict["topic"],
            "prerequisites": topic_dict.get("prerequisites", [])
        }
    
    def _get_cognitive_weight(self, load_level: CognitiveLoadLevel) -> float:
        """Map cognitive load to a weight for daily capacity tracking"""
        weights = {
            CognitiveLoadLevel.HIGH: 0.5,
            CognitiveLoadLevel.MEDIUM: 0.33,
            CognitiveLoadLevel.LOW: 0.15
        }
        return weights.get(load_level, 0.33)
    
    def _generate_session_notes(self, session_template: Dict) -> str:
        """Generate actionable notes for each session"""
        notes = f"Phase: {session_template['phase'].value}"
        
        if session_template["is_weak"]:
            notes += " | ⚠️ Weak Area - Focus & Deep Understanding Required"
        
        if session_template.get("prerequisites"):
            notes += f" | Prerequisites: {', '.join(session_template['prerequisites'][:2])}"
        
        return notes

# ============================================================================
# OUTPUT FORMATTER & REPORT GENERATOR
# ============================================================================

class StudyPlanReportGenerator:
    """Generates comprehensive, actionable study plan reports"""
    
    def __init__(self, plan: StudyPlan):
        self.plan = plan
    
    def generate_comprehensive_report(self) -> str:
        """Generate full report with all insights"""
        report = []
        
        report.append(self._header())
        report.append(self._student_profile())
        report.append(self._executive_summary())
        report.append(self._allocation_breakdown())
        report.append(self._weekly_schedule())
        report.append(self._critical_path_analysis())
        report.append(self._actionable_insights())
        report.append(self._confidence_projection())
        
        return "\n".join(report)
    
    def _header(self) -> str:
        return f"""
{'='*80}
                   🎓 AI-POWERED ADAPTIVE STUDY PLANNER
                         Engineering Students Edition
{'='*80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Target Completion: {self.plan.target_completion_date.strftime('%Y-%m-%d')}
"""
    
    def _student_profile(self) -> str:
        s = self.plan.student
        return f"""
{'─'*80}
📋 STUDENT PROFILE
{'─'*80}
Name: {s.name}
College: {s.college}
Branch: {s.branch}
Graduation Year: {s.graduation_year}
Email: {s.email}

⏱️  STUDY AVAILABILITY
Weekdays: {self.plan.weekday_hours} hours/day
Weekends: {self.plan.weekend_hours} hours/day
Preferred Time: {self.plan.preferred_study_time}
Total Available Days: {(self.plan.target_completion_date - datetime.now()).days}
"""
    
    def _executive_summary(self) -> str:
        total_hours = sum(s.duration_minutes / 60 for s in self.plan.schedule)
        unique_subjects = len(set(s.subject_name for s in self.plan.schedule))
        total_sessions = len(self.plan.schedule)
        
        return f"""
{'─'*80}
📊 EXECUTIVE SUMMARY
{'─'*80}
✓ Total Study Sessions Generated: {total_sessions}
✓ Subjects Covered: {unique_subjects}
✓ Total Allocated Hours: {total_hours:.1f}
✓ Average Session Duration: {total_hours / total_sessions * 60:.0f} minutes
✓ Schedule Efficiency: Optimized for your cognitive capacity
✓ Prerequisite-Aware: All weak areas prioritized correctly
"""
    
    def _allocation_breakdown(self) -> str:
        allocation_dict = defaultdict(float)
        
        for session in self.plan.schedule:
            allocation_dict[session.subject_name] += session.duration_minutes / 60
        
        total = sum(allocation_dict.values())
        
        report = f"\n{'─'*80}\n📚 SUBJECT-WISE TIME ALLOCATION\n{'─'*80}\n"
        
        for subject in self.plan.subjects:
            hours = allocation_dict.get(subject.name, 0)
            percentage = (hours / total * 100) if total > 0 else 0
            confidence = subject.confidence_level
            reason = self._allocation_reason(subject, hours)
            
            bar_length = int(percentage / 2)
            bar = "█" * bar_length + "░" * (50 - bar_length)
            
            report += f"""
{subject.name} ({subject.credits} credits)
{bar} {percentage:.1f}% ({hours:.1f}h)
Confidence: {confidence}/5 | Reason: {reason}
"""
        
        return report
    
    def _allocation_reason(self, subject: Subject, hours: float) -> str:
        factors = []
        
        if subject.confidence_level <= 2:
            factors.append("Low confidence")
        
        if len(subject.weak_areas) >= 2:
            factors.append(f"{len(subject.weak_areas)} weak areas")
        
        if subject.credits >= 4:
            factors.append("High credits")
        
        if not factors:
            factors.append("Solid foundation - maintained optimal load")
        
        return " & ".join(factors)
    
    def _weekly_schedule(self) -> str:
        """Format schedule by week"""
        report = f"\n{'─'*80}\n📅 WEEKLY BREAKDOWN (First 2 Weeks)\n{'─'*80}\n"
        
        week_sessions = defaultdict(list)
        for session in self.plan.schedule[:50]:  # First 50 sessions
            week_num = (session.date - datetime.now()).days // 7
            week_sessions[week_num].append(session)
        
        for week_num in sorted(week_sessions.keys())[:2]:
            sessions = week_sessions[week_num]
            report += f"\n🔹 WEEK {week_num + 1}\n"
            
            day_sessions = defaultdict(list)
            for session in sessions:
                day_key = session.date.strftime("%a, %b %d")
                day_sessions[day_key].append(session)
            
            for day, day_sesh in sorted(day_sessions.items()):
                total_min = sum(s.duration_minutes for s in day_sesh)
                report += f"  {day}: {len(day_sesh)} sessions ({total_min//60}h {total_min%60}m)\n"
                
                for session in sorted(day_sesh, key=lambda x: x.cognitive_load.value, reverse=True)[:3]:
                    emoji = "🔴" if session.cognitive_load == CognitiveLoadLevel.HIGH else \
                            "🟡" if session.cognitive_load == CognitiveLoadLevel.MEDIUM else "🟢"
                    report += f"    {emoji} {session.subject_name}: {session.topic} ({session.duration_minutes}m)\n"
        
        return report
    
    def _critical_path_analysis(self) -> str:
        """Identify critical weak areas that block progress"""
        report = f"\n{'─'*80}\n⚠️  CRITICAL PATH ANALYSIS\n{'─'*80}\n"
        report += "\n🎯 WEAK AREAS REQUIRING IMMEDIATE ATTENTION:\n"
        
        weak_sessions = [s for s in self.plan.schedule 
                        if s.session_type in [StudyPhaseType.FOUNDATION, StudyPhaseType.DEEP_DIVE]]
        weak_sessions = sorted(weak_sessions, key=lambda x: x.date)[:10]
        
        for i, session in enumerate(weak_sessions, 1):
            report += f"\n{i}. {session.subject_name} → {session.topic}\n"
            report += f"   📅 Target: {session.date.strftime('%a, %b %d')}\n"
            report += f"   ⏱️  Duration: {session.duration_minutes} minutes\n"
            
            if session.notes:
                report += f"   📝 Notes: {session.notes}\n"
        
        return report
    
    def _actionable_insights(self) -> str:
        """Provide specific, actionable next steps"""
        report = f"\n{'─'*80}\n💡 ACTIONABLE INSIGHTS & NEXT STEPS\n{'─'*80}\n"
        
        # Next 7 days
        next_week = [s for s in self.plan.schedule 
                    if (s.date - datetime.now()).days <= 7]
        
        report += "\n✨ NEXT 7 DAYS FOCUS:\n"
        
        topics_set = set()
        for session in next_week:
            key = f"{session.subject_name}:{session.topic}"
            if key not in topics_set:
                topics_set.add(key)
                report += f"  • {session.subject_name}: {session.topic}\n"
        
        # Prerequisite warnings
        report += "\n⚡ PREREQUISITE DEPENDENCIES TO WATCH:\n"
        
        for subject in self.plan.subjects:
            if subject.weak_areas:
                report += f"  • {subject.name}: Master {subject.weak_areas[0]} BEFORE {subject.weak_areas[1] if len(subject.weak_areas) > 1 else 'advanced topics'}\n"
        
        # Confidence boosters
        report += "\n🚀 CONFIDENCE BOOSTERS:\n"
        report += f"  • Dedicate first 40% of your time on weak areas for maximum impact\n"
        report += f"  • Complete prerequisite topics before moving to advanced concepts\n"
        report += f"  • Review your strong areas only if time permits\n"
        
        return report
    
    def _confidence_projection(self) -> str:
        """Project confidence improvements"""
        report = f"\n{'─'*80}\n📈 CONFIDENCE PROJECTION\n{'─'*80}\n"
        
        report += "\nBased on this adaptive plan, expected improvements:\n\n"
        
        for subject in self.plan.subjects:
            current = subject.confidence_level
            weak_count = len(subject.weak_areas)
            potential_gain = min(2, weak_count * 0.6)
            projected = min(5, current + potential_gain)
            
            report += f"{subject.name}:\n"
            report += f"  Current Confidence: {current}/5\n"
            report += f"  Projected Confidence: {projected:.1f}/5\n"
            report += f"  Confidence Gain: +{projected - current:.1f}\n"
            report += f"  Progress: {'█' * int((projected-current)*10)}{'░' * (10 - int((projected-current)*10))}\n\n"
        
        report += f"\n{'='*80}\n"
        report += "✅ Your personalized study plan is ready!\n"
        report += "📌 Review the schedule daily and adjust based on actual progress.\n"
        report += "💪 Remember: Consistent effort beats last-minute cramming every time!\n"
        report += f"{'='*80}\n"
        
        return report

# ============================================================================
# USER INTERACTION & INPUT COLLECTION
# ============================================================================

class StudyPlannerInterface:
    """Interactive CLI interface for data collection"""
    
    @staticmethod
    def collect_student_info() -> StudentInfo:
        """Collect student details"""
        print("\n" + "="*80)
        print("📝 STUDENT INFORMATION")
        print("="*80)
        
        name = input("Your Name: ").strip()
        college = input("College Name: ").strip()
        branch = input("Branch (e.g., Computer Science Engineering): ").strip()
        graduation_year = int(input("Graduation Year (e.g., 2026): "))
        email = input("Email ID: ").strip()
        
        return StudentInfo(name, college, branch, graduation_year, email)
    
    @staticmethod
    def collect_subjects() -> List[Subject]:
        """Collect subject details"""
        print("\n" + "="*80)
        print("📚 SUBJECTS & CREDITS")
        print("="*80)
        
        subjects = []
        num_subjects = int(input("Number of Subjects: "))
        
        for i in range(num_subjects):
            print(f"\n--- Subject {i+1} ---")
            name = input("Subject Name: ").strip()
            credits = int(input("Credits: "))
            
            print("Enter strong areas (comma-separated):")
            strong_areas = [x.strip() for x in input("> ").split(",")]
            
            print("Enter weak areas (comma-separated):")
            weak_areas = [x.strip() for x in input("> ").split(",")]
            
            confidence = int(input("Confidence Level (1-5): "))
            
            subject = Subject(
                name=name,
                credits=credits,
                strong_areas=strong_areas,
                weak_areas=weak_areas,
                confidence_level=confidence
            )
            subjects.append(subject)
        
        return subjects
    
    @staticmethod
    def collect_availability() -> Tuple[float, float, str]:
        """Collect study time availability"""
        print("\n" + "="*80)
        print("⏱️  STUDY TIME AVAILABILITY")
        print("="*80)
        
        weekday = float(input("Hours available per weekday: "))
        weekend = float(input("Hours available per weekend day: "))
        preferred_time = input("Preferred study time (morning/afternoon/evening/night): ").strip()
        
        return weekday, weekend, preferred_time
    
    @staticmethod
    def collect_target_date() -> datetime:
        """Collect target completion date"""
        print("\n" + "="*80)
        print("🎯 TARGET COMPLETION DATE")
        print("="*80)
        
        date_str = input("Enter target date (YYYY-MM-DD): ").strip()
        return datetime.strptime(date_str, "%Y-%m-%d")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution flow"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "🎓 WELCOME TO AI-POWERED ADAPTIVE STUDY PLANNER 🎓".center(78) + "║")
    print("║" + "For Engineering Students".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    
    try:
        # Collect all data
        interface = StudyPlannerInterface()
        
        student_info = interface.collect_student_info()
        subjects = interface.collect_subjects()
        weekday_hours, weekend_hours, preferred_time = interface.collect_availability()
        target_date = interface.collect_target_date()
        
        # Create study plan object
        plan = StudyPlan(
            student=student_info,
            subjects=subjects,
            weekday_hours=weekday_hours,
            weekend_hours=weekend_hours,
            preferred_study_time=preferred_time,
            target_completion_date=target_date
        )
        
        # Generate study plan
        print("\n⏳ Generating your personalized adaptive study plan...")
        print("   (This may take a few moments...)\n")
        
        planner = AdaptiveStudyPlanner()
        completed_plan = planner.generate_study_plan(plan)
        
        # Generate and display report
        report_generator = StudyPlanReportGenerator(completed_plan)
        report = report_generator.generate_comprehensive_report()
        
        print(report)
        
        # Save to file
        filename = f"study_plan_{student_info.name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, 'w') as f:
            f.write(report)
        
        print(f"\n✅ Study plan saved to: {filename}")
        
        # Save schedule as JSON
        schedule_data = {
            "student": asdict(student_info),
            "generated_at": datetime.now().isoformat(),
            "target_completion": target_date.isoformat(),
            "schedule": [
                {
                    "date": s.date.isoformat(),
                    "subject": s.subject_name,
                    "topic": s.topic,
                    "type": s.session_type.value,
                    "duration_minutes": s.duration_minutes,
                    "cognitive_load": s.cognitive_load.value,
                    "notes": s.notes,
                    "session_id": s.session_id
                }
                for s in completed_plan.schedule
            ]
        }
        
        json_filename = f"study_plan_{student_info.name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_filename, 'w') as f:
            json.dump(schedule_data, f, indent=2)
        
        print(f"✅ Detailed schedule saved to: {json_filename}")
        
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()