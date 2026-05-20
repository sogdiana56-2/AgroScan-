"""
Flask API-сервер системы диагностики растений
"""

import os
import sys
import traceback
import cv2
import numpy as np
import base64

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from knowledge_base import DISEASES, SYMPTOM_LIST
from diagnosis_engine import DiagnosisEngine
from image_analyzer import PlantImageAnalyzer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)
CORS(app)

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

engine   = DiagnosisEngine()
analyzer = PlantImageAnalyzer()

ALLOWED_EXT = {"png", "jpg", "jpeg", "webp", "bmp", "tiff"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def safe_opencv_result(cv_result):
    """
    Убирает numpy-объекты (ndarray, numpy int/float) из результата OpenCV
    перед сериализацией Flask в JSON.

    Проблема: _analyze_colors() кладёт numpy-маску в каждый цвет:
        color_analysis["yellow"]["mask"] = <ndarray>  ← не сериализуется!
    _analyze_texture() возвращает numpy int в spots и скалярах.
    """
    # --- color_analysis: убираем ключ "mask" из каждого цвета ---------------
    color_safe = {}
    for name, vals in cv_result.get("color_analysis", {}).items():
        color_safe[name] = {
            k: (int(v) if hasattr(v, "item") else v)
            for k, v in vals.items()
            if k != "mask"
        }

    # --- texture_analysis: конвертируем numpy скаляры, spots — list of dict --
    texture_safe = {}
    for k, v in cv_result.get("texture_analysis", {}).items():
        if k == "spots":
            texture_safe[k] = [
                {sk: (int(sv) if hasattr(sv, "item") else sv)
                 for sk, sv in spot.items()}
                for spot in (v or [])
            ]
        elif hasattr(v, "item"):
            texture_safe[k] = v.item()
        else:
            texture_safe[k] = v

    return {
        "detected_symptoms": cv_result.get("detected_symptoms", []),
        "health_score":      cv_result.get("health_score"),
        "color_analysis":    color_safe,
        "texture_analysis":  texture_safe,
        "debug_images":      cv_result.get("debug_images", {}),
    }


# ── UI ────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect")
def detect_page():
    return render_template("detect.html")


# ── HEALTH ────────────────────────────────────────────
@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


# ── SYMPTOMS / DISEASES ───────────────────────────────
@app.route("/api/symptoms")
def get_symptoms():
    return jsonify({
        "symptoms": [
            {"id": k, "label": v["label"], "icon": v["icon"]}
            for k, v in SYMPTOM_LIST.items()
        ]
    })


@app.route("/api/diseases")
def get_diseases():
    return jsonify({
        "diseases": [
            {"id": k, "name": v["name"], "name_en": v["name_en"],
             "severity": v["severity"], "description": v["description"]}
            for k, v in DISEASES.items()
            if k != "healthy"
        ]
    })


# ── DIAGNOSE (только симптомы, без фото) ──────────────
@app.route("/api/diagnose", methods=["POST"])
def diagnose():
    try:
        data     = request.get_json(force=True)
        symptoms = data.get("symptoms", [])
        if not isinstance(symptoms, list):
            return jsonify({"error": "symptoms должен быть списком"}), 400
        valid  = [s for s in symptoms if s in SYMPTOM_LIST]
        result = engine.diagnose(symptoms=valid)
        result["plant_name"] = data.get("plant_name", "Неизвестное растение")
        return jsonify(result)
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Server error"}), 500


# ── UPLOAD (multipart/form-data) ──────────────────────
@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "no file"}), 400
    file = request.files["image"]
    if not allowed_file(file.filename):
        return jsonify({"error": "bad format"}), 400

    npimg = np.frombuffer(file.read(), np.uint8)
    img   = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Не удалось декодировать изображение"}), 400

    _, buf     = cv2.imencode(".jpg", img)
    img_source = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

    cv_result = analyzer.analyze(img_source)
    if "error" in cv_result:
        return jsonify({"error": cv_result["error"]}), 500

    image_symptoms = cv_result.get("detected_symptoms", [])
    health_score   = cv_result.get("health_score")
    diag = engine.diagnose(symptoms=[], image_symptoms=image_symptoms, image_score=health_score)

    return jsonify({
        "diagnoses":          diag.get("diagnoses", []),
        "primary_diagnosis":  diag.get("primary_diagnosis"),
        "combined_symptoms":  diag.get("combined_symptoms", []),
        "image_health_score": health_score,
        "plant_name":         request.form.get("plant_name", "Неизвестное растение"),
        "opencv_analysis":    safe_opencv_result(cv_result),
    })


# ── ANALYZE IMAGE (base64 JSON) — основной эндпоинт UI
@app.route("/api/analyze-image", methods=["POST"])
def analyze_image():
    try:
        data       = request.get_json(force=True)
        img_source = data.get("image_base64")
        symptoms   = data.get("symptoms", [])
        plant_name = data.get("plant_name", "Неизвестное растение")

        if not img_source:
            return jsonify({"error": "Изображение не предоставлено"}), 400

        cv_result = analyzer.analyze(img_source)
        if "error" in cv_result:
            return jsonify({"error": cv_result["error"]}), 500

        image_symptoms = cv_result.get("detected_symptoms", [])
        health_score   = cv_result.get("health_score")
        valid_syms     = [s for s in symptoms if s in SYMPTOM_LIST]

        diag = engine.diagnose(
            symptoms=valid_syms,
            image_symptoms=image_symptoms,
            image_score=health_score
        )
        diag["plant_name"] = plant_name

        return jsonify({
            **diag,
            "opencv_analysis": safe_opencv_result(cv_result),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Ошибка сервера: {str(e)}"}), 500


if __name__ == "__main__":
    print("🌱 Plant Diagnostics → http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)