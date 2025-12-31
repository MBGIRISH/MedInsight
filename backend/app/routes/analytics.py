"""
Analytics API endpoints for MedInsight dashboard.
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from app.models.db import MongoDB
from bson import ObjectId
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/analytics/audits")
async def get_audits(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    categories: Optional[str] = Query(None, description="Comma-separated list of categories"),
    symptoms: Optional[str] = Query(None, description="Comma-separated list of symptoms"),
    limit: int = Query(1000, description="Maximum number of audits to return")
):
    """Get all audits with optional filtering."""
    try:
        collection = MongoDB.get_collection("audits")
        query = {}
        
        # Date filtering
        if start_date or end_date:
            date_query = {}
            if start_date:
                date_query["$gte"] = datetime.fromisoformat(start_date)
            if end_date:
                date_query["$lte"] = datetime.fromisoformat(end_date) + timedelta(days=1)
            query["created_at"] = date_query
        
        # Severity filtering
        if severity:
            # Map severity to risk_score ranges
            severity_ranges = {
                "critical": {"$gte": 9.0},
                "high": {"$gte": 6.0, "$lt": 9.0},
                "moderate": {"$gte": 3.0, "$lt": 6.0},
                "low": {"$gte": 1.0, "$lt": 3.0},
                "ok": {"$eq": 0.0}
            }
            if severity.lower() in severity_ranges:
                query["risk_score"] = severity_ranges[severity.lower()]
        
        # Category filtering
        if categories:
            category_list = [c.strip() for c in categories.split(",")]
            category_keywords = {
                "Respiratory": ["breath", "cough", "sputum", "oxygen", "dyspnea", "sob", "wheezing"],
                "Cardiac": ["chest pain", "heart", "cardiac", "myocardial", "angina", "tachycardia"],
                "Infection": ["fever", "infection", "sepsis", "meningitis", "bacterial"],
                "Neurological": ["headache", "confusion", "stroke", "seizure", "weakness", "facial droop"],
                "Gastrointestinal": ["nausea", "vomiting", "diarrhea", "abdominal pain", "gi"],
                "Metabolic": ["glucose", "diabetes", "diabetic", "hyperglycemia", "hypoglycemia"]
            }
            category_regex = []
            for cat in category_list:
                if cat in category_keywords:
                    category_regex.extend([{"$regex": kw, "$options": "i"} for kw in category_keywords[cat]])
            if category_regex:
                query["$or"] = query.get("$or", [])
                query["$or"].append({"ner_result.normalized_entities.symptoms": {"$in": category_regex}})
        
        # Symptom filtering
        if symptoms:
            symptom_list = [s.strip() for s in symptoms.split(",")]
            query["ner_result.normalized_entities.symptoms"] = {"$in": symptom_list}
        
        audits = list(collection.find(query).sort("created_at", -1).limit(limit))
        
        # Convert ObjectId to string
        for audit in audits:
            audit["_id"] = str(audit["_id"])
            if "created_at" in audit and isinstance(audit["created_at"], datetime):
                audit["created_at"] = audit["created_at"].isoformat()
        
        return {
            "total": len(audits),
            "audits": audits
        }
    except Exception as e:
        logger.error(f"Error fetching audits: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/kpis")
async def get_kpis(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    severity: Optional[str] = Query(None, description="Filter by severity")
):
    """Get KPI metrics."""
    try:
        collection = MongoDB.get_collection("audits")
        query = {}
        
        if start_date or end_date:
            date_query = {}
            if start_date:
                date_query["$gte"] = datetime.fromisoformat(start_date)
            if end_date:
                date_query["$lte"] = datetime.fromisoformat(end_date) + timedelta(days=1)
            query["created_at"] = date_query
        
        # Severity filtering
        if severity:
            severity_ranges = {
                "critical": {"$gte": 9.0},
                "high": {"$gte": 6.0, "$lt": 9.0},
                "moderate": {"$gte": 3.0, "$lt": 6.0},
                "low": {"$gte": 1.0, "$lt": 3.0},
                "ok": {"$eq": 0.0}
            }
            if severity.lower() in severity_ranges:
                query["risk_score"] = severity_ranges[severity.lower()]
        
        total_audits = collection.count_documents(query)
        
        # Calculate severity distribution
        severity_counts = {
            "critical": collection.count_documents({**query, "risk_score": {"$gte": 9.0}}),
            "high": collection.count_documents({**query, "risk_score": {"$gte": 6.0, "$lt": 9.0}}),
            "moderate": collection.count_documents({**query, "risk_score": {"$gte": 3.0, "$lt": 6.0}}),
            "low": collection.count_documents({**query, "risk_score": {"$gte": 1.0, "$lt": 3.0}}),
            "ok": collection.count_documents({**query, "risk_score": {"$eq": 0.0}})
        }
        
        # Calculate average risk score
        pipeline = [
            {"$match": query},
            {"$group": {
                "_id": None,
                "avg_risk_score": {"$avg": "$risk_score"},
                "max_risk_score": {"$max": "$risk_score"},
                "min_risk_score": {"$min": "$risk_score"}
            }}
        ]
        stats = list(collection.aggregate(pipeline))
        avg_risk = stats[0]["avg_risk_score"] if stats else 0.0
        
        # Get most common symptoms
        pipeline = [
            {"$match": query},
            {"$unwind": {"path": "$ner_result.normalized_entities.symptoms", "preserveNullAndEmptyArrays": True}},
            {"$group": {"_id": "$ner_result.normalized_entities.symptoms", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 10}
        ]
        symptom_results = list(collection.aggregate(pipeline))
        top_symptoms = [{"symptom": r["_id"], "count": r["count"]} for r in symptom_results if r["_id"]]
        
        # Get most frequently triggered agent
        pipeline = [
            {"$match": query},
            {"$unwind": {"path": "$agent_outputs", "preserveNullAndEmptyArrays": True}},
            {"$match": {"agent_outputs.status": {"$ne": "ok"}}},
            {"$group": {"_id": "$agent_outputs.agent", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 1}
        ]
        agent_results = list(collection.aggregate(pipeline))
        top_agent = agent_results[0]["_id"] if agent_results else "None"
        
        return {
            "total_audits": total_audits,
            "severity_distribution": severity_counts,
            "average_risk_score": round(avg_risk, 2),
            "top_symptoms": top_symptoms,
            "most_triggered_agent": top_agent
        }
    except Exception as e:
        logger.error(f"Error calculating KPIs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/trends")
async def get_trends(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    group_by: str = Query("day", description="Group by: day, week, month")
):
    """Get risk score and severity trends over time."""
    try:
        collection = MongoDB.get_collection("audits")
        query = {}
        
        if start_date or end_date:
            date_query = {}
            if start_date:
                date_query["$gte"] = datetime.fromisoformat(start_date)
            if end_date:
                date_query["$lte"] = datetime.fromisoformat(end_date) + timedelta(days=1)
            query["created_at"] = date_query
        
        # Severity filtering
        if severity:
            severity_ranges = {
                "critical": {"$gte": 9.0},
                "high": {"$gte": 6.0, "$lt": 9.0},
                "moderate": {"$gte": 3.0, "$lt": 6.0},
                "low": {"$gte": 1.0, "$lt": 3.0},
                "ok": {"$eq": 0.0}
            }
            if severity.lower() in severity_ranges:
                query["risk_score"] = severity_ranges[severity.lower()]
        
        # Group by date
        date_format = {
            "day": {"$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"}},
            "week": {"$dateToString": {"format": "%Y-W%V", "date": "$created_at"}},
            "month": {"$dateToString": {"format": "%Y-%m", "date": "$created_at"}}
        }.get(group_by, date_format["day"])
        
        pipeline = [
            {"$match": query},
            {"$group": {
                "_id": date_format,
                "avg_risk_score": {"$avg": "$risk_score"},
                "count": {"$sum": 1},
                "critical": {"$sum": {"$cond": [{"$gte": ["$risk_score", 9.0]}, 1, 0]}},
                "high": {"$sum": {"$cond": [{"$and": [{"$gte": ["$risk_score", 6.0]}, {"$lt": ["$risk_score", 9.0]}]}, 1, 0]}},
                "moderate": {"$sum": {"$cond": [{"$and": [{"$gte": ["$risk_score", 3.0]}, {"$lt": ["$risk_score", 6.0]}]}, 1, 0]}},
                "low": {"$sum": {"$cond": [{"$and": [{"$gte": ["$risk_score", 1.0]}, {"$lt": ["$risk_score", 3.0]}]}, 1, 0]}},
                "ok": {"$sum": {"$cond": [{"$eq": ["$risk_score", 0.0]}, 1, 0]}}
            }},
            {"$sort": {"_id": 1}}
        ]
        
        results = list(collection.aggregate(pipeline))
        
        return {
            "trends": results
        }
    except Exception as e:
        logger.error(f"Error fetching trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/symptoms")
async def get_symptom_analytics(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    limit: int = Query(20, description="Number of top symptoms to return")
):
    """Get symptom frequency and correlation data."""
    try:
        collection = MongoDB.get_collection("audits")
        query = {}
        
        if start_date or end_date:
            date_query = {}
            if start_date:
                date_query["$gte"] = datetime.fromisoformat(start_date)
            if end_date:
                date_query["$lte"] = datetime.fromisoformat(end_date) + timedelta(days=1)
            query["created_at"] = date_query
        
        # Severity filtering
        if severity:
            severity_ranges = {
                "critical": {"$gte": 9.0},
                "high": {"$gte": 6.0, "$lt": 9.0},
                "moderate": {"$gte": 3.0, "$lt": 6.0},
                "low": {"$gte": 1.0, "$lt": 3.0},
                "ok": {"$eq": 0.0}
            }
            if severity.lower() in severity_ranges:
                query["risk_score"] = severity_ranges[severity.lower()]
        
        # Get symptom frequency
        pipeline = [
            {"$match": query},
            {"$unwind": {"path": "$ner_result.normalized_entities.symptoms", "preserveNullAndEmptyArrays": True}},
            {"$group": {"_id": "$ner_result.normalized_entities.symptoms", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": limit}
        ]
        symptom_results = list(collection.aggregate(pipeline))
        
        symptoms = [r["_id"] for r in symptom_results if r["_id"]]
        
        # Build correlation matrix (simplified - count co-occurrences)
        correlation_data = []
        for i, sym1 in enumerate(symptoms[:10]):
            for sym2 in symptoms[i+1:11]:
                co_occurrence = collection.count_documents({
                    **query,
                    "ner_result.normalized_entities.symptoms": {"$all": [sym1, sym2]}
                })
                if co_occurrence > 0:
                    correlation_data.append({
                        "symptom1": sym1,
                        "symptom2": sym2,
                        "co_occurrence": co_occurrence
                    })
        
        return {
            "symptom_frequency": [{"symptom": r["_id"], "count": r["count"]} for r in symptom_results if r["_id"]],
            "correlations": correlation_data
        }
    except Exception as e:
        logger.error(f"Error fetching symptom analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/agents")
async def get_agent_analytics(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    severity: Optional[str] = Query(None, description="Filter by severity")
):
    """Get agent performance and trigger statistics."""
    try:
        collection = MongoDB.get_collection("audits")
        query = {}
        
        if start_date or end_date:
            date_query = {}
            if start_date:
                date_query["$gte"] = datetime.fromisoformat(start_date)
            if end_date:
                date_query["$lte"] = datetime.fromisoformat(end_date) + timedelta(days=1)
            query["created_at"] = date_query
        
        # Severity filtering
        if severity:
            severity_ranges = {
                "critical": {"$gte": 9.0},
                "high": {"$gte": 6.0, "$lt": 9.0},
                "moderate": {"$gte": 3.0, "$lt": 6.0},
                "low": {"$gte": 1.0, "$lt": 3.0},
                "ok": {"$eq": 0.0}
            }
            if severity.lower() in severity_ranges:
                query["risk_score"] = severity_ranges[severity.lower()]
        
        pipeline = [
            {"$match": query},
            {"$unwind": {"path": "$agent_outputs", "preserveNullAndEmptyArrays": True}},
            {"$group": {
                "_id": "$agent_outputs.agent",
                "total_triggers": {"$sum": {"$cond": [{"$ne": ["$agent_outputs.status", "ok"]}, 1, 0]}},
                "avg_score": {"$avg": "$agent_outputs.score"},
                "critical_count": {"$sum": {"$cond": [{"$eq": ["$agent_outputs.status", "critical"]}, 1, 0]}},
                "high_count": {"$sum": {"$cond": [{"$eq": ["$agent_outputs.status", "high"]}, 1, 0]}},
                "moderate_count": {"$sum": {"$cond": [{"$eq": ["$agent_outputs.status", "moderate"]}, 1, 0]}},
                "low_count": {"$sum": {"$cond": [{"$eq": ["$agent_outputs.status", "low"]}, 1, 0]}}
            }},
            {"$sort": {"total_triggers": -1}}
        ]
        
        results = list(collection.aggregate(pipeline))
        
        return {
            "agent_performance": [
                {
                    "agent": r["_id"],
                    "total_triggers": r["total_triggers"],
                    "avg_score": round(r["avg_score"] or 0, 2),
                    "severity_breakdown": {
                        "critical": r["critical_count"],
                        "high": r["high_count"],
                        "moderate": r["moderate_count"],
                        "low": r["low_count"]
                    }
                }
                for r in results if r["_id"]
            ]
        }
    except Exception as e:
        logger.error(f"Error fetching agent analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/categories")
async def get_category_analytics(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    categories: Optional[str] = Query(None, description="Comma-separated list of categories to filter")
):
    """Get clinical category analytics."""
    try:
        collection = MongoDB.get_collection("audits")
        query = {}
        
        if start_date or end_date:
            date_query = {}
            if start_date:
                date_query["$gte"] = datetime.fromisoformat(start_date)
            if end_date:
                date_query["$lte"] = datetime.fromisoformat(end_date) + timedelta(days=1)
            query["created_at"] = date_query
        
        # Category keywords
        all_categories = {
            "Respiratory": ["breath", "cough", "sputum", "oxygen", "dyspnea", "sob", "wheezing"],
            "Cardiac": ["chest pain", "heart", "cardiac", "myocardial", "angina", "tachycardia"],
            "Infection": ["fever", "infection", "sepsis", "meningitis", "bacterial"],
            "Neurological": ["headache", "confusion", "stroke", "seizure", "weakness", "facial droop"],
            "Gastrointestinal": ["nausea", "vomiting", "diarrhea", "abdominal pain", "gi"],
            "Metabolic": ["glucose", "diabetes", "diabetic", "hyperglycemia", "hypoglycemia"]
        }
        
        # Filter categories if specified
        categories_to_process = all_categories
        if categories:
            category_list = [c.strip() for c in categories.split(",")]
            categories_to_process = {k: v for k, v in all_categories.items() if k in category_list}
        
        category_stats = {}
        for category, keywords in categories_to_process.items():
            # Count audits matching category keywords
            category_query = {
                **query,
                "$or": [
                    {"ner_result.normalized_entities.symptoms": {"$regex": keyword, "$options": "i"}}
                    for keyword in keywords
                ]
            }
            count = collection.count_documents(category_query)
            
            if count > 0:
                # Get severity breakdown
                pipeline = [
                    {"$match": category_query},
                    {"$group": {
                        "_id": None,
                        "critical": {"$sum": {"$cond": [{"$gte": ["$risk_score", 9.0]}, 1, 0]}},
                        "high": {"$sum": {"$cond": [{"$and": [{"$gte": ["$risk_score", 6.0]}, {"$lt": ["$risk_score", 9.0]}]}, 1, 0]}},
                        "moderate": {"$sum": {"$cond": [{"$and": [{"$gte": ["$risk_score", 3.0]}, {"$lt": ["$risk_score", 6.0]}]}, 1, 0]}},
                        "low": {"$sum": {"$cond": [{"$and": [{"$gte": ["$risk_score", 1.0]}, {"$lt": ["$risk_score", 3.0]}]}, 1, 0]}},
                        "ok": {"$sum": {"$cond": [{"$eq": ["$risk_score", 0.0]}, 1, 0]}},
                        "avg_risk": {"$avg": "$risk_score"}
                    }}
                ]
                severity_results = list(collection.aggregate(pipeline))
                
                if severity_results:
                    category_stats[category] = {
                        "count": count,
                        "severity_breakdown": {
                            "critical": severity_results[0]["critical"],
                            "high": severity_results[0]["high"],
                            "moderate": severity_results[0]["moderate"],
                            "low": severity_results[0]["low"],
                            "ok": severity_results[0]["ok"]
                        },
                        "avg_risk_score": round(severity_results[0]["avg_risk"] or 0, 2)
                    }
        
        return {
            "categories": category_stats
        }
    except Exception as e:
        logger.error(f"Error fetching category analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/medications")
async def get_medication_analytics(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    severity: Optional[str] = Query(None, description="Filter by severity")
):
    """Get medication safety analytics (dosage errors, interactions)."""
    try:
        collection = MongoDB.get_collection("audits")
        query = {}
        
        if start_date or end_date:
            date_query = {}
            if start_date:
                date_query["$gte"] = datetime.fromisoformat(start_date)
            if end_date:
                date_query["$lte"] = datetime.fromisoformat(end_date) + timedelta(days=1)
            query["created_at"] = date_query
        
        # Severity filtering
        if severity:
            severity_ranges = {
                "critical": {"$gte": 9.0},
                "high": {"$gte": 6.0, "$lt": 9.0},
                "moderate": {"$gte": 3.0, "$lt": 6.0},
                "low": {"$gte": 1.0, "$lt": 3.0},
                "ok": {"$eq": 0.0}
            }
            if severity.lower() in severity_ranges:
                query["risk_score"] = severity_ranges[severity.lower()]
        
        # Get dosage errors
        dosage_errors = collection.count_documents({
            **query,
            "agent_outputs": {
                "$elemMatch": {
                    "agent": "DosageChecker",
                    "status": {"$in": ["critical", "high", "moderate"]}
                }
            }
        })
        
        # Get interaction errors
        interaction_errors = collection.count_documents({
            **query,
            "agent_outputs": {
                "$elemMatch": {
                    "agent": "InteractionChecker",
                    "status": {"$in": ["critical", "high", "moderate"]}
                }
            }
        })
        
        # Get most common dosage issues
        pipeline = [
            {"$match": {
                **query,
                "agent_outputs": {
                    "$elemMatch": {
                        "agent": "DosageChecker",
                        "status": {"$ne": "ok"}
                    }
                }
            }},
            {"$unwind": "$agent_outputs"},
            {"$match": {"agent_outputs.agent": "DosageChecker", "agent_outputs.status": {"$ne": "ok"}}},
            {"$group": {
                "_id": "$agent_outputs.message",
                "count": {"$sum": 1},
                "avg_score": {"$avg": "$agent_outputs.score"}
            }},
            {"$sort": {"count": -1}},
            {"$limit": 10}
        ]
        dosage_issues = list(collection.aggregate(pipeline))
        
        # Get most common interactions
        pipeline = [
            {"$match": {
                **query,
                "agent_outputs": {
                    "$elemMatch": {
                        "agent": "InteractionChecker",
                        "status": {"$ne": "ok"}
                    }
                }
            }},
            {"$unwind": "$agent_outputs"},
            {"$match": {"agent_outputs.agent": "InteractionChecker", "agent_outputs.status": {"$ne": "ok"}}},
            {"$group": {
                "_id": "$agent_outputs.message",
                "count": {"$sum": 1},
                "avg_score": {"$avg": "$agent_outputs.score"}
            }},
            {"$sort": {"count": -1}},
            {"$limit": 10}
        ]
        interaction_issues = list(collection.aggregate(pipeline))
        
        return {
            "dosage_errors": dosage_errors,
            "interaction_errors": interaction_errors,
            "common_dosage_issues": [
                {"issue": r["_id"], "count": r["count"], "avg_score": round(r["avg_score"] or 0, 2)}
                for r in dosage_issues
            ],
            "common_interactions": [
                {"interaction": r["_id"], "count": r["count"], "avg_score": round(r["avg_score"] or 0, 2)}
                for r in interaction_issues
            ]
        }
    except Exception as e:
        logger.error(f"Error fetching medication analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

