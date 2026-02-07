"""
Comprehensive Test Suite for AI Study Planner
Run with: pytest test_study_planner.py -v
"""

import pytest
from datetime import datetime, timedelta
from study_planner_main import (
    AdaptiveStudyPlanner, StudyPlan, StudentInfo, Subject,
    CognitiveLoadOptimizer, PrerequisiteMapper,
    StudyPhaseType, CognitiveLoadLevel, StudyPlanReportGenerator
)

# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_student():
    return StudentInfo(
        name="Aman Kumar",
        college="XYZ Institute of Technology",
        branch="Computer Science Engineering",
        graduation_year=2026,
        email="aman@example.com"
    )

@pytest.fixture
def sample_subjects():
    return [
        Subject(
            name="Data Structures",
            credits=4,
            strong_areas=["Arrays", "Linked Lists"],
            weak_areas=["Trees", "Graphs"],
            confidence_level=3
        ),
        Subject(
            name="Operating Systems",
            credits=3,
            strong_areas=["Processes", "Threads"],
            weak_areas=["Deadlocks", "Memory Management"],
            confidence_level=2
        ),
        Subject(
            name="Engineering Mathematics",
            credits=4,
            strong_areas=["Differential Equations"],
            weak_areas=["Laplace Transform"],
            confidence_level=3
        )
    ]

@pytest.fixture
def sample_study_plan(sample_student, sample_subjects):
    return StudyPlan(
        student=sample_student,
        subjects=sample_subjects,
        weekday_hours=3.0,
        weekend_hours=6.0,
        preferred_study_time="night",
        target_completion_date=datetime.now() + timedelta(days=36)
    )

# ============================================================================
# TEST COGNITIVE LOAD OPTIMIZER
# ============================================================================

class TestCognitiveLoadOptimizer:
    
    def test_cognitive_demand_calculation(self):
        """Test cognitive demand calculation"""
        optimizer = CognitiveLoadOptimizer()
        
        # Create test subject
        subject = Subject(
            name="Test Subject",
            credits=4,
            strong_areas=["Topic A"],
            weak_areas=["Topic B"],
            confidence_level=2
        )
        
        # Test high demand (weak area, low confidence)
        demand = optimizer.calculate_cognitive_demand(
            subject, "Topic B", True, StudyPhaseType.FOUNDATION
        )
        assert 0.3 <= demand <= 1.0
        assert demand > 0.5  # Should be relatively high
        
        # Test low demand (strong area)
        demand = optimizer.calculate_cognitive_demand(
            subject, "Topic A", False, StudyPhaseType.REVISION
        )
        assert demand < 0.5  # Should be relatively low
    
    def test_cognitive_level_assignment(self):
        """Test cognitive level classification"""
        optimizer = CognitiveLoadOptimizer()
        
        # Test HIGH level
        level = optimizer.assign_cognitive_level(0.85)
        assert level == CognitiveLoadLevel.HIGH
        
        # Test MEDIUM level
        level = optimizer.assign_cognitive_level(0.6)
        assert level == CognitiveLoadLevel.MEDIUM
        
        # Test LOW level
        level = optimizer.assign_cognitive_level(0.35)
        assert level == CognitiveLoadLevel.LOW
    
    def test_optimal_time_slot_assignment(self):
        """Test time slot assignment based on cognitive load"""
        optimizer = CognitiveLoadOptimizer()
        
        # Test HIGH load in night preference
        slot = optimizer.get_optimal_time_slot(CognitiveLoadLevel.HIGH, "night")
        assert isinstance(slot, tuple)
        assert len(slot) == 2
        
        # Test LOW load gets flexible hours
        slot = optimizer.get_optimal_time_slot(CognitiveLoadLevel.LOW, "night")
        assert isinstance(slot, tuple)

# ============================================================================
# TEST PREREQUISITE MAPPER
# ============================================================================

class TestPrerequisiteMapper:
    
    def test_prerequisite_mapper_initialization(self, sample_subjects):
        mapper = PrerequisiteMapper(sample_subjects)
        assert len(mapper.subjects) == 3
        assert mapper.subject_names == ["Data Structures", "Operating Systems", "Engineering Mathematics"]
    
    def test_critical_path_identification(self, sample_subjects):
        """Test identification of critical path"""
        mapper = PrerequisiteMapper(sample_subjects)
        target_date = datetime.now() + timedelta(days=36)
        
        critical_path = mapper.identify_critical_path(target_date)
        
        # Should return list of tuples
        assert isinstance(critical_path, list)
        for item in critical_path:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[1], datetime)
    
    def test_topic_sequence_generation(self, sample_subjects):
        """Test topic sequence generation"""
        mapper = PrerequisiteMapper(sample_subjects)
        
        sequence = mapper.get_topic_sequence("Data Structures")
        
        # Should return list of lists
        assert isinstance(sequence, list)
        assert all(isinstance(s, list) for s in sequence)

# ============================================================================
# TEST STUDY PLANNER
# ============================================================================

class TestAdaptiveStudyPlanner:
    
    def test_planner_initialization(self):
        """Test planner initialization"""
        planner = AdaptiveStudyPlanner()
        assert planner.cognitive_optimizer is not None
        assert planner.prerequisite_mapper is None
    
    def test_study_hours_allocation(self, sample_study_plan):
        """Test study hours allocation algorithm"""
        planner = AdaptiveStudyPlanner()
        allocations = planner._allocate_study_hours(sample_study_plan)
        
        # Should allocate to all subjects
        assert len(allocations) == 3
        
        # All allocations should be positive
        for hours in allocations.values():
            assert hours > 0
        
        # Total should match available hours
        total_available = sample_study_plan._calculate_total_available_hours(
            sample_study_plan,
            (sample_study_plan.target_completion_date - datetime.now()).days
        )
        total_allocated = sum(allocations.values())
        assert abs(total_allocated - total_available) < 1  # Allow 1 hour tolerance
    
    def test_plan_generation(self, sample_study_plan):
        """Test complete plan generation"""
        planner = AdaptiveStudyPlanner()
        completed_plan = planner.generate_study_plan(sample_study_plan)
        
        # Should have schedule
        assert completed_plan.schedule is not None
        assert len(completed_plan.schedule) > 0
        
        # All sessions should be valid
        for session in completed_plan.schedule:
            assert session.subject_name
            assert session.topic
            assert session.session_type in StudyPhaseType
            assert session.duration_minutes > 0
            assert session.cognitive_load in CognitiveLoadLevel
    
    def test_schedule_ordering(self, sample_study_plan):
        """Test that sessions are in chronological order"""
        planner = AdaptiveStudyPlanner()
        completed_plan = planner.generate_study_plan(sample_study_plan)
        
        for i in range(len(completed_plan.schedule) - 1):
            assert completed_plan.schedule[i].date <= completed_plan.schedule[i+1].date
    
    def test_cognitive_load_balancing(self, sample_study_plan):
        """Test that daily cognitive load is balanced"""
        planner = AdaptiveStudyPlanner()
        completed_plan = planner.generate_study_plan(sample_study_plan)
        
        from collections import defaultdict
        daily_load = defaultdict(float)
        
        for session in completed_plan.schedule:
            date_key = session.date.strftime("%Y-%m-%d")
            weight = planner._get_cognitive_weight(session.cognitive_load)
            daily_load[date_key] += weight
        
        # No day should be extremely overloaded
        for load in daily_load.values():
            assert load <= 2.0  # Reasonable upper bound

# ============================================================================
# TEST REPORT GENERATION
# ============================================================================

class TestReportGenerator:
    
    def test_report_generation(self, sample_study_plan):
        """Test comprehensive report generation"""
        planner = AdaptiveStudyPlanner()
        completed_plan = planner.generate_study_plan(sample_study_plan)
        
        generator = StudyPlanReportGenerator(completed_plan)
        report = generator.generate_comprehensive_report()
        
        # Report should be non-empty string
        assert isinstance(report, str)
        assert len(report) > 0
        
        # Report should contain key sections
        assert "STUDENT PROFILE" in report
        assert "ALLOCATION" in report
        assert "WEEKLY" in report
    
    def test_report_contains_student_info(self, sample_study_plan):
        """Test that report contains student information"""
        planner = AdaptiveStudyPlanner()
        completed_plan = planner.generate_study_plan(sample_study_plan)
        
        generator = StudyPlanReportGenerator(completed_plan)
        report = generator.generate_comprehensive_report()
        
        # Should contain student name and email
        assert "Aman Kumar" in report
        assert "aman@example.com" in report
    
    def test_report_contains_insights(self, sample_study_plan):
        """Test that report contains actionable insights"""
        planner = AdaptiveStudyPlanner()
        completed_plan = planner.generate_study_plan(sample_study_plan)
        
        generator = StudyPlanReportGenerator(completed_plan)
        report = generator.generate_comprehensive_report()
        
        # Should contain insights sections
        assert "NEXT 7 DAYS" in report or "FOCUS" in report

# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    
    def test_end_to_end_workflow(self, sample_study_plan):
        """Test complete workflow from input to output"""
        # 1. Generate plan
        planner = AdaptiveStudyPlanner()
        completed_plan = planner.generate_study_plan(sample_study_plan)
        
        # 2. Generate report
        generator = StudyPlanReportGenerator(completed_plan)
        report = generator.generate_comprehensive_report()
        
        # 3. Verify outputs
        assert len(completed_plan.schedule) > 0
        assert len(report) > 1000  # Should be detailed report
    
    def test_multiple_plan_generation(self, sample_study_plan):
        """Test generating multiple plans"""
        planner = AdaptiveStudyPlanner()
        
        # Generate plan twice
        plan1 = planner.generate_study_plan(sample_study_plan)
        plan2 = planner.generate_study_plan(sample_study_plan)
        
        # Both should be valid
        assert len(plan1.schedule) > 0
        assert len(plan2.schedule) > 0
    
    def test_scalability_with_more_subjects(self, sample_student):
        """Test scalability with more subjects"""
        # Create plan with 6 subjects
        subjects = [
            Subject("Subject " + str(i), 3, ["Area A"], ["Area B"], 3)
            for i in range(6)
        ]
        
        plan = StudyPlan(
            student=sample_student,
            subjects=subjects,
            weekday_hours=4.0,
            weekend_hours=8.0,
            preferred_study_time="evening",
            target_completion_date=datetime.now() + timedelta(days=60)
        )
        
        planner = AdaptiveStudyPlanner()
        completed_plan = planner.generate_study_plan(plan)
        
        # Should handle multiple subjects
        assert len(completed_plan.schedule) > 0
        
        # Should allocate to all subjects
        allocated_subjects = set(s.subject_name for s in completed_plan.schedule)
        assert len(allocated_subjects) == 6

# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    
    def test_single_subject_plan(self, sample_student):
        """Test plan with single subject"""
        subject = Subject(
            name="Only Subject",
            credits=6,
            strong_areas=["Topic A"],
            weak_areas=["Topic B"],
            confidence_level=3
        )
        
        plan = StudyPlan(
            student=sample_student,
            subjects=[subject],
            weekday_hours=3.0,
            weekend_hours=6.0,
            preferred_study_time="morning",
            target_completion_date=datetime.now() + timedelta(days=30)
        )
        
        planner = AdaptiveStudyPlanner()
        completed_plan = planner.generate_study_plan(plan)
        
        assert len(completed_plan.schedule) > 0
        assert all(s.subject_name == "Only Subject" for s in completed_plan.schedule)
    
    def test_very_high_confidence(self, sample_student):
        """Test with very high confidence subjects"""
        subject = Subject(
            name="Expert Subject",
            credits=4,
            strong_areas=["Area A", "Area B"],
            weak_areas=[],  # No weak areas
            confidence_level=5
        )
        
        plan = StudyPlan(
            student=sample_student,
            subjects=[subject],
            weekday_hours=2.0,
            weekend_hours=4.0,
            preferred_study_time="morning",
            target_completion_date=datetime.now() + timedelta(days=30)
        )
        
        planner = AdaptiveStudyPlanner()
        completed_plan = planner.generate_study_plan(plan)
        
        # Should still generate plan but with less time
        assert len(completed_plan.schedule) > 0
    
    def test_short_deadline(self, sample_study_plan):
        """Test with very short deadline"""
        sample_study_plan.target_completion_date = datetime.now() + timedelta(days=7)
        
        planner = AdaptiveStudyPlanner()
        completed_plan = planner.generate_study_plan(sample_study_plan)
        
        # Should still generate plan
        assert len(completed_plan.schedule) > 0

# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    
    def test_generation_speed(self, sample_study_plan):
        """Test that plan generation completes in reasonable time"""
        import time
        
        planner = AdaptiveStudyPlanner()
        
        start_time = time.time()
        completed_plan = planner.generate_study_plan(sample_study_plan)
        end_time = time.time()
        
        generation_time = end_time - start_time
        
        # Should complete in under 5 seconds
        assert generation_time < 5.0
    
    def test_report_generation_speed(self, sample_study_plan):
        """Test that report generation is fast"""
        import time
        
        planner = AdaptiveStudyPlanner()
        completed_plan = planner.generate_study_plan(sample_study_plan)
        
        generator = StudyPlanReportGenerator(completed_plan)
        
        start_time = time.time()
        report = generator.generate_comprehensive_report()
        end_time = time.time()
        
        generation_time = end_time - start_time
        
        # Should complete in under 2 seconds
        assert generation_time < 2.0

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])