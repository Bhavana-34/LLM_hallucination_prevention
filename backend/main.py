from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
import os
import logging
from typing import Optional, List, Dict
import uuid
from fact_extract import fact_extractor
from refdatabase import wikipedia_verifier
from confidence_scorer import confidence_scorer
from contradiction_detector import contradiction_detector, divergence_checker
from response_formatter import response_formatter


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="LLM Hallucination Prevention API")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")  # Override via .env if needed

# In-memory conversation storage
# Groq uses OpenAI-style messages: {"role": "user"/"assistant"/"system", "content": "..."}
conversations: Dict[str, List[Dict]] = {}

# Per-session analytics tracking
session_analytics: Dict[str, Dict] = {}

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    extracted_facts: List[Dict] = []
    confidence_report: Dict = {}
    contradictions: List[Dict] = []
    formatted_response: Dict = {}
    divergence: Dict = {}
    hallucination_risk: Dict = {}


# Health check endpoint
@app.get("/")
async def root():
    return {"status": "LLM Proxy is running", "model": GROQ_MODEL}


# Main chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Generate or use existing session ID
        session_id = request.session_id or str(uuid.uuid4())

        logger.info(f"Session {session_id}: Received message: {request.message}")

        # Get or create conversation history
        if session_id not in conversations:
            conversations[session_id] = []

        # Add user message to history
        conversations[session_id].append({
            "role": "user",
            "content": request.message
        })

        # Call Groq API with full conversation history
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=conversations[session_id],
            temperature=0.7,
            max_tokens=1024,
        )

        response_text = completion.choices[0].message.content

        # Add assistant response to history
        conversations[session_id].append({
            "role": "assistant",
            "content": response_text
        })

        # Check divergence before adding current response to history
        divergence = divergence_checker.check_divergence(session_id, response_text)
        divergence_checker.add_response(session_id, response_text)
        if divergence["diverged"]:
            logger.warning(
                f"Session {session_id}: Divergence detected (similarity={divergence['similarity_score']})"
            )

        # Extract facts from response
        extracted_facts = fact_extractor.extract_facts(response_text)
        logger.info(f"Session {session_id}: Extracted {len(extracted_facts)} facts")

        # Verify facts against Wikipedia
        verified_facts = wikipedia_verifier.verify_facts(extracted_facts)
        logger.info(f"Session {session_id}: Verified {len(verified_facts)} facts")

        # Check for contradictions with previous messages
        contradictions = contradiction_detector.detect_contradictions(session_id, verified_facts)
        if contradictions:
            logger.warning(f"Session {session_id}: Found {len(contradictions)} contradiction(s)!")

        # Add facts to session history (for future contradiction checks)
        contradiction_detector.add_facts(session_id, verified_facts)

        # Calculate confidence score
        confidence_report = confidence_scorer.score_response(verified_facts)

        # Adjust confidence if contradictions found
        if contradictions:
            confidence_report["overall_confidence"] = "low"
            confidence_report["color"] = "red"
            confidence_report["emoji"] = "🔴"
            confidence_report["summary"] = (
                f"⚠️  {len(contradictions)} contradiction(s) detected in conversation"
            )

        # Compute composite hallucination risk score
        hallucination_risk = confidence_scorer.compute_risk_score(
            confidence_report, contradictions, divergence
        )

        # Update session analytics
        if session_id not in session_analytics:
            session_analytics[session_id] = {
                "turns": 0,
                "risk_scores": [],
                "contradiction_counts": [],
                "divergence_events": 0,
            }
        sa = session_analytics[session_id]
        sa["turns"] += 1
        sa["risk_scores"].append(hallucination_risk["risk_score"])
        sa["contradiction_counts"].append(len(contradictions))
        if divergence.get("diverged"):
            sa["divergence_events"] += 1

        formatted_response = response_formatter.format_response(
            response_text, verified_facts, contradictions
        )

        logger.info(
            f"Session {session_id}: {hallucination_risk['emoji']} "
            f"Risk={hallucination_risk['risk_score']}/100 | "
            f"Confidence={confidence_report['overall_confidence']} ({confidence_report['confidence_score']})"
        )
        logger.info(f"Session {session_id}: Response generated successfully")

        return ChatResponse(
            response=response_text,
            session_id=session_id,
            extracted_facts=verified_facts,
            confidence_report=confidence_report,
            contradictions=contradictions,
            formatted_response=formatted_response,
            divergence=divergence,
            hallucination_risk=hallucination_risk,
        )

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Get conversation history
@app.get("/history/{session_id}")
async def get_history(session_id: str):
    if session_id not in conversations:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "history": conversations[session_id]}


@app.get("/analytics/{session_id}")
async def get_analytics(session_id: str):
    """Return session-level hallucination analytics"""
    sa = session_analytics.get(session_id)
    if not sa or sa["turns"] == 0:
        raise HTTPException(status_code=404, detail="No analytics for this session")
    risk_scores = sa["risk_scores"]
    return {
        "session_id": session_id,
        "turns": sa["turns"],
        "avg_risk_score": round(sum(risk_scores) / len(risk_scores), 1),
        "max_risk_score": max(risk_scores),
        "min_risk_score": min(risk_scores),
        "risk_trend": risk_scores,
        "total_contradictions": sum(sa["contradiction_counts"]),
        "divergence_events": sa["divergence_events"],
        "contradiction_rate": round(
            sum(1 for c in sa["contradiction_counts"] if c > 0) / sa["turns"] * 100, 1
        ),
    }


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation and fact history for a session"""
    if session_id in conversations:
        del conversations[session_id]
    session_analytics.pop(session_id, None)
    contradiction_detector.clear_session(session_id)
    divergence_checker.clear_session(session_id)
    return {"message": f"Session {session_id} cleared"}