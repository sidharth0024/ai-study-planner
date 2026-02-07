"""
Flask API Server for AI Study Planner
Provides REST API endpoints for study plan generation
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta
import json
import uuid
from study_planner_main import (
    AdaptiveStudyPlanner, StudyPlan, StudentInfo, Subject,
    StudyPlanReportGenerator, StudyPhaseType, CognitiveLoadLevel
)

app = Flask(__name__)
CORS(app)

# In-memory storage (use database in production)
generated_plans = {}

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0"
    }), 200

@app.route('/api/generate-plan', methods=['POST'])
def generate_plan():
    """
    Generate personalized study plan
    POST /api/generate-plan
    """
    try:
        data = request.json
        
        # Validate input
        if not all(key in data for key in ['student', 'subjects', 'weekday_hours', 
                                           'weekend_hours', 'preferred_time', 'target_date']):
            return jsonify({"error": "Missing required fields"}), 400
        
        # Create student info
        student_data = data['student']
        student = StudentInfo(
            name=student_data['name'],
            college=student_data['college'],
            branch=student_data['branch'],
            graduation_year=student_data['graduation_year'],
            email=student_data['email']
        )
        
        # Create subjects
        subjects = []
        for subject_data in data['subjects']:
            subject = Subject(
                name=subject_data['name'],
                credits=subject_data['credits'],
                strong_areas=subject_data['strong_areas'],
                weak_areas=subject_data['weak_areas'],
                confidence_level=subject_data['confidence_level']
            )
            subjects.append(subject)
        
        # Parse target date
        target_date = datetime.strptime(data['target_date'], '%Y-%m-%d')
        
        # Create study plan
        plan = StudyPlan(
            student=student,
            subjects=subjects,
            weekday_hours=data['weekday_hours'],
            weekend_hours=data['weekend_hours'],
            preferred_study_time=data['preferred_time'],
            target_completion_date=target_date
        )
        
        # Generate plan
        planner = AdaptiveStudyPlanner()
        completed_plan = planner.generate_study_plan(plan)
        
        # Generate report
        report_generator = StudyPlanReportGenerator(completed_plan)
        report = report_generator.generate_comprehensive_report()
        
        # Calculate allocations
        allocations = {}
        for session in completed_plan.schedule:
            if session.subject_name not in allocations:
                allocations[session.subject_name] = 0
            allocations[session.subject_name] += session.duration_minutes / 60
        
        # Calculate confidence projections
        confidence_projections = []
        for subject in subjects:
            current = subject.confidence_level
            weak_count = len(subject.weak_areas)
            potential_gain = min(2, weak_count * 0.6)
            projected = min(5, current + potential_gain)
            
            confidence_projections.append({
                "subject": subject.name,
                "current": current,
                "projected": round(projected, 1),
                "gain": round(projected - current, 1)
            })
        
        # Generate insights
        insights = generate_insights(completed_plan)
        
        # Create response
        plan_id = str(uuid.uuid4())
        response_data = {
            "status": "success",
            "plan_id": plan_id,
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_sessions": len(completed_plan.schedule),
                "total_hours": sum(allocations.values()),
                "subjects": len(subjects),
                "days_available": (target_date - datetime.now()).days
            },
            "allocations": allocations,
            "confidence_projections": confidence_projections,
            "insights": insights,
            "schedule_preview": generate_schedule_preview(completed_plan),
            "report": report
        }
        
        # Store plan
        generated_plans[plan_id] = {
            "plan": completed_plan,
            "data": response_data,
            "created_at": datetime.now()
        }
        
        return jsonify(response_data), 200
        
    except ValueError as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/plan/<plan_id>', methods=['GET'])
def get_plan(plan_id):
    """Get generated plan details"""
    if plan_id not in generated_plans:
        return jsonify({"error": "Plan not found"}), 404
    
    plan_data = generated_plans[plan_id]['data']
    return jsonify(plan_data), 200

@app.route('/api/plan/<plan_id>/export', methods=['GET'])
def export_plan(plan_id):
    """Export plan as JSON"""
    if plan_id not in generated_plans:
        return jsonify({"error": "Plan not found"}), 404
    
    completed_plan = generated_plans[plan_id]['plan']
    
    export_data = {
        "student": {
            "name": completed_plan.student.name,
            "email": completed_plan.student.email,
            "college": completed_plan.student.college,
            "branch": completed_plan.student.branch,
            "graduation_year": completed_plan.student.graduation_year
        },
        "generated_at": datetime.now().isoformat(),
        "target_completion": completed_plan.target_completion_date.isoformat(),
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
    
    return jsonify(export_data), 200

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    total_plans = len(generated_plans)
    total_sessions = sum(len(p['plan'].schedule) for p in generated_plans.values())
    
    return jsonify({
        "total_plans_generated": total_plans,
        "total_sessions_created": total_sessions,
        "average_sessions_per_plan": total_sessions / max(1, total_plans),
        "timestamp": datetime.now().isoformat()
    }), 200

def generate_insights(plan: StudyPlan):
    """Generate actionable insights"""
    insights = []
    
    # Next 7 days focus
    next_week = [s for s in plan.schedule 
                if (s.date - datetime.now()).days <= 7]
    
    topics_set = set()
    next_7_days = []
    for session in next_week:
        key = f"{session.subject_name}:{session.topic}"
        if key not in topics_set:
            topics_set.add(key)
            next_7_days.append(f"{session.subject_name}: {session.topic}")
    
    insights.append({
        "type": "focus_area",
        "title": "Next 7 Days Focus",
        "content": next_7_days[:5]
    })
    
    # Weak areas that need immediate attention
    weak_area_insight = []
    for subject in plan.subjects:
        if subject.weak_areas:
            weak_area_insight.append(f"{subject.name} - {subject.weak_areas[0]}")
    
    insights.append({
        "type": "weak_areas",
        "title": "Weak Areas Requiring Attention",
        "content": weak_area_insight[:5]
    })
    
    # Study time recommendations
    total_hours = sum(s.duration_minutes / 60 for s in plan.schedule)
    insights.append({
        "type": "recommendation",
        "title": "Study Time Recommendations",
        "content": [
            f"Total hours required: {total_hours:.1f}",
            f"Preferred study time: {plan.preferred_study_time}",
            f"Daily average: {total_hours / max(1, (plan.target_completion_date - datetime.now()).days):.1f} hours"
        ]
    })
    
    return insights

def generate_schedule_preview(plan: StudyPlan):
    """Generate preview of first 2 weeks"""
    preview = {}
    
    for session in plan.schedule[:50]:
        date_str = session.date.strftime("%Y-%m-%d")
        if date_str not in preview:
            preview[date_str] = []
        
        preview[date_str].append({
            "subject": session.subject_name,
            "topic": session.topic,
            "type": session.session_type.value,
            "duration": session.duration_minutes,
            "cognitive_load": session.cognitive_load.value
        })
    
    return preview

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)