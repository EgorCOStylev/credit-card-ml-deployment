"""
Flask web service for credit card default prediction.

Endpoints:
  POST /predict          — predict default for a given client (uses model v1 by default)
  POST /predict?model=v2 — predict using model v2 (A/B testing)
  GET  /health           — service health check
  GET  /features         — list expected feature names
"""

import logging
import json
import os
import random
from datetime import datetime

from flask import Flask, request, jsonify
from model_handler import predict, get_feature_names

# ── Logging setup (JSON format for production log collectors) ──────────────────
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": datetime.now(datetime.timezone.utc).isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra"):
            log_obj.update(record.extra)
        return json.dumps(log_obj)

handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────────────
app = Flask(__name__)

# A/B traffic split ratio (probability of routing to v2)
AB_SPLIT = float(os.environ.get("AB_SPLIT", "0.5"))


def _resolve_model_version() -> str:
    """
    Determine which model version to use.
    Priority: explicit ?model=vX query param > A/B random split.
    """
    requested = request.args.get("model")
    if requested in ("v1", "v2"):
        return requested
    # Random 50/50 A/B split
    return "v2" if random.random() < AB_SPLIT else "v1"


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "timestamp": datetime.now(datetime.timezone.utc).isoformat() + "Z"}), 200


@app.route("/features", methods=["GET"])
def features():
    """Return list of expected feature names for /predict."""
    version = request.args.get("model", "v1")
    try:
        names = get_feature_names(version)
        return jsonify({"model_version": version, "features": names}), 200
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    """
    Predict credit card default.

    Request body (JSON):
      {
        "limit_bal": 20000,
        "sex": 2,
        "education": 2,
        "marriage": 1,
        "age": 24,
        "pay_0": 2, "pay_2": 2, "pay_3": -1, "pay_4": -1, "pay_5": -1, "pay_6": -1,
        "bill_amt1": 3913, ..., "bill_amt6": 0,
        "pay_amt1": 0, ..., "pay_amt6": 0
      }

    Optional query param:
      ?model=v1  (default) | ?model=v2

    Response:
      {
        "prediction": 0,          // 0 = no default, 1 = default
        "probability": 0.1234,    // probability of default
        "model_version": "v1"
      }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    version = _resolve_model_version()

    try:
        result = predict(data, version=version)
    except ValueError as e:
        logger.warning("Prediction error", extra={"error": str(e), "version": version})
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503

    logger.info(
        "Prediction made",
        extra={
            "model_version": version,
            "prediction": result["prediction"],
            "probability": result["probability"],
        },
    )
    return jsonify(result), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
