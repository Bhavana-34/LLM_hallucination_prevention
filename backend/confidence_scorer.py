import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class ConfidenceScorer:
    """Calculate overall confidence scores for responses"""
    
    def __init__(self):
        # Confidence thresholds
        self.HIGH_CONFIDENCE_THRESHOLD = 0.8
        self.MEDIUM_CONFIDENCE_THRESHOLD = 0.5
        
    def score_response(self, verified_facts: List[Dict]) -> Dict:
        """
        Calculate overall confidence score for a response
        Returns scoring details with color coding
        """
        if not verified_facts:
            return {
                "overall_confidence": "unknown",
                "confidence_score": 0.0,
                "color": "gray",
                "emoji": "❓",
                "summary": "No verifiable facts found",
                "stats": {
                    "total_facts": 0,
                    "verified": 0,
                    "unverified": 0,
                    "unknown": 0
                }
            }
        
        # Count verification results
        stats = self._calculate_stats(verified_facts)
        
        # Calculate confidence score (0.0 to 1.0)
        confidence_score = self._calculate_confidence_score(stats, verified_facts)
        
        # Determine overall confidence level
        confidence_level, color, emoji = self._determine_confidence_level(confidence_score, stats)
        
        # Generate summary message
        summary = self._generate_summary(stats, confidence_level)
        
        return {
            "overall_confidence": confidence_level,
            "confidence_score": round(confidence_score, 2),
            "color": color,
            "emoji": emoji,
            "summary": summary,
            "stats": stats,
            "detailed_facts": self._categorize_facts(verified_facts)
        }
    
    def _calculate_stats(self, verified_facts: List[Dict]) -> Dict:
        """Calculate statistics from verified facts"""
        total = len(verified_facts)
        verified = sum(1 for f in verified_facts if f.get('verified') == True)
        unverified = sum(1 for f in verified_facts if f.get('verified') == False and f.get('confidence') != 'unknown')
        unknown = sum(1 for f in verified_facts if f.get('confidence') == 'unknown')
        
        # Calculate by confidence level
        high_conf = sum(1 for f in verified_facts if f.get('confidence') == 'high')
        medium_conf = sum(1 for f in verified_facts if f.get('confidence') == 'medium')
        low_conf = sum(1 for f in verified_facts if f.get('confidence') == 'low')
        
        return {
            "total_facts": total,
            "verified": verified,
            "unverified": unverified,
            "unknown": unknown,
            "high_confidence": high_conf,
            "medium_confidence": medium_conf,
            "low_confidence": low_conf
        }
    
    def _calculate_confidence_score(self, stats: Dict, verified_facts: List[Dict]) -> float:
        """
        Calculate weighted confidence score
        High confidence facts = 1.0 points
        Medium confidence facts = 0.6 points
        Low confidence facts = 0.2 points
        Unknown = 0.0 points
        """
        if stats["total_facts"] == 0:
            return 0.0
        
        total_points = 0.0
        
        for fact in verified_facts:
            confidence = fact.get('confidence', 'unknown')
            
            if confidence == 'high':
                total_points += 1.0
            elif confidence == 'medium':
                total_points += 0.6
            elif confidence == 'low':
                total_points += 0.2
            else:  # unknown
                total_points += 0.0
        
        # Average score
        return total_points / stats["total_facts"]
    
    def _determine_confidence_level(self, score: float, stats: Dict) -> tuple:
        """
        Determine overall confidence level, color, and emoji
        Returns: (level, color, emoji)
        """
        # If majority are low confidence or unverified, it's low
        if stats["low_confidence"] > stats["total_facts"] / 2:
            return ("low", "red", "🔴")
        
        # If high score, it's verified
        if score >= self.HIGH_CONFIDENCE_THRESHOLD:
            return ("high", "green", "🟢")
        
        # If medium score, it's uncertain
        elif score >= self.MEDIUM_CONFIDENCE_THRESHOLD:
            return ("medium", "yellow", "🟡")
        
        # Otherwise, it's low confidence
        else:
            return ("low", "red", "🔴")
    
    def _generate_summary(self, stats: Dict, confidence_level: str) -> str:
        """Generate human-readable summary"""
        total = stats["total_facts"]
        verified = stats["verified"]
        
        if confidence_level == "high":
            return f"Highly verified: {verified}/{total} facts confirmed by Wikipedia"
        elif confidence_level == "medium":
            return f"Partially verified: {verified}/{total} facts confirmed, some uncertain"
        elif confidence_level == "low":
            if stats["unverified"] > 0:
                return f"Low confidence: {stats['unverified']}/{total} facts could not be verified"
            else:
                return f"Uncertain: Unable to verify most claims"
        else:
            return "No verifiable facts found"
    
    def _categorize_facts(self, verified_facts: List[Dict]) -> Dict:
        """Categorize facts by verification status"""
        verified = [f for f in verified_facts if f.get('verified') == True and f.get('confidence') == 'high']
        uncertain = [f for f in verified_facts if f.get('confidence') in ['medium', 'unknown']]
        contradicted = [f for f in verified_facts if f.get('verified') == False and f.get('confidence') == 'low']
        
        return {
            "verified": verified,
            "uncertain": uncertain,
            "contradicted": contradicted
        }

    def compute_risk_score(self, confidence_report: Dict, contradictions: List[Dict], divergence: Dict) -> Dict:
        """
        Compute a composite Hallucination Risk Score (0–100).
        Higher = more likely the response contains hallucinations.

        Weights:
          40 pts  — low confidence (inverted confidence_score)
          40 pts  — contradictions (capped at 3 contradictions = max)
          20 pts  — response divergence
        """
        # Component 1: confidence (0-40)
        conf_score = confidence_report.get('confidence_score', 0.0)
        confidence_risk = (1.0 - conf_score) * 40

        # Component 2: contradictions (0-40), capped at 3
        n_contradictions = min(len(contradictions), 3)
        contradiction_risk = (n_contradictions / 3) * 40

        # Component 3: divergence (0-20)
        div_score = divergence.get('similarity_score', 1.0) if divergence else 1.0
        divergence_risk = (1.0 - div_score) * 20 if divergence.get('diverged') else 0

        total_risk = round(confidence_risk + contradiction_risk + divergence_risk, 1)

        if total_risk >= 70:
            level, emoji, color = "critical", "🚨", "red"
        elif total_risk >= 45:
            level, emoji, color = "high", "🔴", "orange"
        elif total_risk >= 25:
            level, emoji, color = "medium", "🟡", "yellow"
        else:
            level, emoji, color = "low", "🟢", "green"

        return {
            "risk_score": total_risk,
            "risk_level": level,
            "emoji": emoji,
            "color": color,
            "breakdown": {
                "confidence_risk": round(confidence_risk, 1),
                "contradiction_risk": round(contradiction_risk, 1),
                "divergence_risk": round(divergence_risk, 1),
            },
            "label": f"{emoji} {level.upper()} hallucination risk ({total_risk}/100)"
        }

# Create singleton instance
confidence_scorer = ConfidenceScorer()