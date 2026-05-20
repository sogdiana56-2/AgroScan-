"""
Движок экспертной системы диагностики заболеваний растений.
Объединяет анализ симптомов и результаты OpenCV.
ИСПРАВЛЕНИЕ: убраны аннотации типов Python 3.10+ (list[str] | None)
             заменены на Optional[List[str]] для совместимости с 3.8/3.9
"""

from typing import Dict, List, Optional, Tuple
from knowledge_base import DISEASES, SYMPTOM_WEIGHTS


class DiagnosisEngine:
    """Экспертная система: симптомы → диагноз → рекомендации."""

    MIN_SCORE = 0.10
    TOP_N     = 3

    def diagnose(
        self,
        symptoms:        List[str],
        image_symptoms:  Optional[List[str]] = None,
        image_score:     Optional[int] = None,
    ) -> dict:
        """
        symptoms:       список симптомов из UI (выбраны пользователем)
        image_symptoms: список симптомов, обнаруженных OpenCV
        image_score:    скор здоровья 0-100 из OpenCV
        """
        combined: Dict[str, float] = dict.fromkeys(symptoms, 1.0)
        for s in (image_symptoms or []):
            if s in combined:
                combined[s] = min(combined[s] + 0.5, 1.5)
            else:
                combined[s] = 0.6

        if not combined:
            if image_score is not None and image_score >= 70:
                return self._build_result([("healthy", 0.85)], combined, image_score)
            return {"diagnoses": [], "combined_symptoms": [], "image_health_score": image_score}

        scores = self._score_diseases(combined)
        top    = self._select_top(scores)

        if not top and image_score is not None and image_score >= 70:
            top = [("healthy", 0.60)]

        return self._build_result(top, combined, image_score)

    # ─── Скоринг ──────────────────────────────────────

    def _score_diseases(self, symptom_weights: Dict[str, float]) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for symptom, user_weight in symptom_weights.items():
            disease_map = SYMPTOM_WEIGHTS.get(symptom, {})
            for disease, kb_weight in disease_map.items():
                scores[disease] = scores.get(disease, 0.0) + kb_weight * user_weight

        if not scores:
            return {}

        max_score = max(scores.values())
        if max_score > 0:
            scores = {k: round(v / max_score, 4) for k, v in scores.items()}
        return scores

    def _select_top(self, scores: Dict[str, float]) -> List[Tuple[str, float]]:
        filtered = [(k, v) for k, v in scores.items() if v >= self.MIN_SCORE]
        filtered.sort(key=lambda x: x[1], reverse=True)
        return filtered[:self.TOP_N]

    # ─── Результат ────────────────────────────────────

    def _build_result(
        self,
        top:           List[Tuple[str, float]],
        combined_syms: Dict[str, float],
        image_score:   Optional[int],
    ) -> dict:
        diagnoses = []
        for disease_id, confidence in top:
            disease = DISEASES.get(disease_id)
            if not disease:
                continue
            matching = [s for s in combined_syms if s in disease["symptoms"]]
            diagnoses.append({
                "id":                disease_id,
                "name":              disease["name"],
                "name_en":           disease["name_en"],
                "severity":          disease["severity"],
                "confidence":        round(confidence * 100),
                "description":       disease["description"],
                "matching_symptoms": matching,
                "recommendations":   disease["recommendations"],
                "prevention":        disease["prevention"],
            })

        return {
            "diagnoses":          diagnoses,
            "combined_symptoms":  list(combined_syms.keys()),
            "image_health_score": image_score,
            "primary_diagnosis":  diagnoses[0] if diagnoses else None,
        }
