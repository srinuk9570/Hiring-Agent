"""
Standardized API response format across all endpoints
"""

from datetime import datetime, timezone
from flask import jsonify


def success_response(data=None, message="Success", status_code=200):
    """Standard success response"""
    return jsonify({
        "success": True,
        "message": message,
        "data": data,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }), status_code


def error_response(message="Error", status_code=400, errors=None):
    """Standard error response"""
    return jsonify({
        "success": False,
        "message": message,
        "errors": errors or [],
        "timestamp": datetime.now(timezone.utc).isoformat()
    }), status_code


def paginated_response(items, total, page, per_page):
    """Standard paginated response"""
    return jsonify({
        "success": True,
        "data": items,
        "pagination": {
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": (total + per_page - 1) // per_page,
            "has_next": (page * per_page) < total,
            "has_prev": page > 1
        }
    }), 200