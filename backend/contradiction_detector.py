import logging
from typing import List, Dict, Optional
import re

logger = logging.getLogger(__name__)

# Load spaCy model for semantic similarity
try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy loaded in contradiction detector")
except Exception:
    _nlp = None
    logger.warning("spaCy not available — falling back to word overlap for similarity")

class ContradictionDetector:
    """Detect contradictions in conversation history"""
    
    def __init__(self):
        # Store facts by session
        self.session_facts = {}  # {session_id: [facts]}
        
    def add_facts(self, session_id: str, facts: List[Dict]):
        """Add new facts to session history"""
        if session_id not in self.session_facts:
            self.session_facts[session_id] = []
        
        self.session_facts[session_id].extend(facts)
        logger.info(f"Session {session_id}: Now tracking {len(self.session_facts[session_id])} total facts")
    
    def detect_contradictions(self, session_id: str, new_facts: List[Dict]) -> List[Dict]:
        """
        Check if new facts contradict previous facts in the conversation
        Returns list of contradictions found
        """
        if session_id not in self.session_facts:
            return []
        
        previous_facts = self.session_facts[session_id]
        contradictions = []
        
        for new_fact in new_facts:
            # Check against all previous facts
            for old_fact in previous_facts:
                contradiction = self._check_fact_pair(new_fact, old_fact)
                if contradiction:
                    contradictions.append(contradiction)
        
        if contradictions:
            logger.warning(f"Session {session_id}: Found {len(contradictions)} contradiction(s)")
        
        return contradictions
    
    def _check_fact_pair(self, fact1: Dict, fact2: Dict) -> Optional[Dict]:
        """
        Check if two facts contradict each other
        Returns contradiction details if found, None otherwise
        """
        # Must be about same entity type
        if fact1['entity_type'] != fact2['entity_type']:
            return None
        
        entity_type = fact1['entity_type']
        
        # Check for numeric contradictions
        if entity_type in ['CARDINAL', 'QUANTITY', 'MONEY', 'MEASUREMENT', 'POPULATION']:
            return self._check_numeric_contradiction(fact1, fact2)
        
        # Check for date contradictions
        if entity_type == 'DATE':
            return self._check_date_contradiction(fact1, fact2)
        
        # Check for entity contradictions (same subject, different claims)
        if entity_type in ['GPE', 'LOC', 'PERSON', 'ORG']:
            return self._check_entity_contradiction(fact1, fact2)
        
        return None
    
    def _check_numeric_contradiction(self, fact1: Dict, fact2: Dict) -> Optional[Dict]:
        """Check if numeric facts contradict"""
        # Extract numbers from both facts
        num1 = self._extract_number(fact1['entity'])
        num2 = self._extract_number(fact2['entity'])
        
        if num1 is None or num2 is None:
            return None
        
        # Check if they're talking about the same thing
        if not self._same_subject(fact1['sentence'], fact2['sentence']):
            return None
        
        # Check if numbers differ significantly (more than 20% difference)
        diff_percent = abs(num1 - num2) / max(num1, num2) * 100
        
        if diff_percent > 20:
            return {
                "type": "numeric_contradiction",
                "severity": "high" if diff_percent > 50 else "medium",
                "previous_claim": fact2['sentence'],
                "current_claim": fact1['sentence'],
                "previous_value": fact2['entity'],
                "current_value": fact1['entity'],
                "difference": f"{diff_percent:.1f}% difference",
                "message": f"Contradiction detected: Different values for the same claim"
            }
        
        return None
    
    def _check_date_contradiction(self, fact1: Dict, fact2: Dict) -> Optional[Dict]:
        """Check if dates contradict"""
        # Extract years
        year1 = self._extract_year(fact1['entity'])
        year2 = self._extract_year(fact2['entity'])
        
        if year1 is None or year2 is None or year1 == year2:
            return None
        
        # Check if talking about same subject
        if not self._same_subject(fact1['sentence'], fact2['sentence']):
            return None
        
        return {
            "type": "date_contradiction",
            "severity": "high",
            "previous_claim": fact2['sentence'],
            "current_claim": fact1['sentence'],
            "previous_value": fact2['entity'],
            "current_value": fact1['entity'],
            "difference": f"{abs(year1 - year2)} years apart",
            "message": f"Contradiction: Different dates for the same event"
        }
    
    def _check_entity_contradiction(self, fact1: Dict, fact2: Dict) -> Optional[Dict]:
        """Check if entity facts contradict"""
        # Check if sentences make contradictory claims about same subject
        # Example: "Paris is in France" vs "Paris is in Germany"
        
        # Look for "is" statements
        pattern1 = self._extract_is_statement(fact1['sentence'])
        pattern2 = self._extract_is_statement(fact2['sentence'])
        
        if pattern1 and pattern2:
            subject1, claim1 = pattern1
            subject2, claim2 = pattern2
            
            # Same subject, different claims
            if self._similar_text(subject1, subject2) and not self._similar_text(claim1, claim2):
                return {
                    "type": "entity_contradiction",
                    "severity": "high",
                    "previous_claim": fact2['sentence'],
                    "current_claim": fact1['sentence'],
                    "message": f"Contradiction: Different information about {subject1}"
                }
        
        return None
    
    def _extract_number(self, text: str) -> Optional[float]:
        """Extract numeric value from text"""
        # Remove commas, extract number
        text_clean = text.replace(',', '')
        
        # Handle millions, billions
        multiplier = 1
        if 'billion' in text.lower():
            multiplier = 1_000_000_000
        elif 'million' in text.lower():
            multiplier = 1_000_000
        elif 'thousand' in text.lower():
            multiplier = 1_000
        
        # Extract number
        match = re.search(r'([\d.]+)', text_clean)
        if match:
            try:
                return float(match.group(1)) * multiplier
            except:
                return None
        
        return None
    
    def _extract_year(self, text: str) -> Optional[int]:
        """Extract year from text"""
        match = re.search(r'(\d{4})', text)
        if match:
            try:
                return int(match.group(1))
            except:
                return None
        return None
    
    def _same_subject(self, sentence1: str, sentence2: str) -> bool:
        """Check if two sentences are about the same subject using spaCy similarity"""
        if _nlp:
            try:
                doc1 = _nlp(sentence1[:200])
                doc2 = _nlp(sentence2[:200])
                return doc1.similarity(doc2) > 0.65
            except Exception:
                pass
        # Fallback: word overlap on first 5 words
        words1 = sentence1.lower().split()[:5]
        words2 = sentence2.lower().split()[:5]
        common = set(words1) & set(words2)
        return len(common) >= 2
    
    def _extract_is_statement(self, sentence: str) -> Optional[tuple]:
        """Extract 'X is Y' patterns from sentence"""
        # Pattern: "Paris is the capital"
        match = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+(.+)', sentence)
        if match:
            return (match.group(1), match.group(2))
        return None
    
    def _similar_text(self, text1: str, text2: str) -> bool:
        """Check if two texts are semantically similar using spaCy"""
        if _nlp:
            try:
                doc1 = _nlp(text1[:200])
                doc2 = _nlp(text2[:200])
                return doc1.similarity(doc2) > 0.75
            except Exception:
                pass
        # Fallback: word overlap
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return False
        overlap = len(words1 & words2) / min(len(words1), len(words2))
        return overlap > 0.5
    
    def get_session_summary(self, session_id: str) -> Dict:
        """Get summary of facts tracked in a session"""
        if session_id not in self.session_facts:
            return {"total_facts": 0, "facts": []}
        
        facts = self.session_facts[session_id]
        return {
            "total_facts": len(facts),
            "facts": facts
        }
    
    def clear_session(self, session_id: str):
        """Clear facts for a session"""
        if session_id in self.session_facts:
            del self.session_facts[session_id]
            logger.info(f"Cleared session {session_id}")

# Create singleton instance
contradiction_detector = ContradictionDetector()


class DivergenceChecker:
    """Detect when LLM responses diverge significantly from previous responses on the same topic."""

    def __init__(self, threshold: float = 0.25):
        self.session_responses: Dict[str, List[str]] = {}
        self.threshold = threshold  # cosine similarity below this → diverged

    def add_response(self, session_id: str, text: str):
        self.session_responses.setdefault(session_id, []).append(text)

    def check_divergence(self, session_id: str, new_text: str) -> Dict:
        history = self.session_responses.get(session_id)
        if not history:
            return {"diverged": False, "similarity_score": 1.0, "message": None}

        score = self._cosine_similarity(new_text, history[-1])
        diverged = score < self.threshold
        return {
            "diverged": diverged,
            "similarity_score": round(score, 3),
            "severity": "high" if score < 0.1 else "medium" if score < 0.2 else "low",
            "message": (
                f"Response diverged from previous turn (similarity: {score:.2f})"
                if diverged else None
            ),
        }

    def _cosine_similarity(self, text1: str, text2: str) -> float:
        stopwords = {'the','a','an','is','are','was','were','in','on','at','to','for','of','and','or','but','it','this','that'}
        def vectorize(t):
            words = [w for w in re.findall(r'\w+', t.lower()) if w not in stopwords]
            freq: Dict[str, int] = {}
            for w in words:
                freq[w] = freq.get(w, 0) + 1
            return freq

        v1, v2 = vectorize(text1), vectorize(text2)
        vocab = set(v1) | set(v2)
        if not vocab:
            return 1.0
        dot = sum(v1.get(w, 0) * v2.get(w, 0) for w in vocab)
        mag1 = sum(x ** 2 for x in v1.values()) ** 0.5
        mag2 = sum(x ** 2 for x in v2.values()) ** 0.5
        return dot / (mag1 * mag2) if mag1 and mag2 else 0.0

    def clear_session(self, session_id: str):
        self.session_responses.pop(session_id, None)


divergence_checker = DivergenceChecker()