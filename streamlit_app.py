import streamlit as st
import sys
import os

# Add backend to path
backend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend')
sys.path.insert(0, backend_path)

from fact_extract import fact_extractor
from refdatabase import wikipedia_verifier
from confidence_scorer import confidence_scorer
from contradiction_detector import contradiction_detector, divergence_checker
from response_formatter import response_formatter
from groq import Groq
from dotenv import load_dotenv
import uuid

# Load environment variables
env_path = os.path.join(backend_path, '.env')
load_dotenv(env_path)

# Configure Groq
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# Page config
st.set_page_config(
    page_title="LLM Hallucination Prevention",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: transparent;
    }
    .fact-green {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .fact-yellow {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .fact-red {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .fact-orange {
        background-color: #ffe5b4;
        border-left: 4px solid #ff8c00;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Header
st.title("🛡️ LLM Hallucination Prevention System")
st.markdown("### Real-time fact-checking for AI responses")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This system:
    - ✅ Extracts facts from AI responses
    - 🔍 Verifies them against Wikipedia
    - 🎯 Scores confidence levels
    - ⚠️ Detects contradictions
    - 📈 Checks response divergence
    - 🚨 Computes Hallucination Risk Score
    """)
    
    st.divider()
    
    st.header("📊 Session Info")
    st.write(f"Session ID: `{st.session_state.session_id[:8]}...`")
    st.write(f"Messages: {len(st.session_state.conversation_history) // 2}")
    
    if st.button("🔄 Reset Session", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.conversation_history = []
        st.rerun()

# Main input
query = st.text_input(
    "Ask a question:",
    placeholder="e.g., What is the population of Tokyo?",
    key="query_input"
)

col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    verify_button = st.button("✨ Verify", type="primary", use_container_width=True)

# Process query
if verify_button and query:
    with st.spinner("🔍 Analyzing response and verifying facts..."):
        try:
            # Add user message to history
            st.session_state.conversation_history.append({
                "role": "user",
                "content": query
            })
            
            # Get response from Groq
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=st.session_state.conversation_history
            )
            response_text = response.choices[0].message.content
            
            # Add assistant response to history
            st.session_state.conversation_history.append({
                "role": "assistant",
                "content": response_text
            })
            
            # Check divergence
            divergence = divergence_checker.check_divergence(st.session_state.session_id, response_text)
            divergence_checker.add_response(st.session_state.session_id, response_text)

            # Extract facts
            extracted_facts = fact_extractor.extract_facts(response_text)
            
            # Verify facts
            verified_facts = wikipedia_verifier.verify_facts(extracted_facts)
            
            # Check contradictions
            contradictions = contradiction_detector.detect_contradictions(
                st.session_state.session_id, 
                verified_facts
            )
            contradiction_detector.add_facts(st.session_state.session_id, verified_facts)
            
            # Calculate confidence
            confidence_report = confidence_scorer.score_response(verified_facts)
            
            # Adjust confidence if contradictions
            if contradictions:
                confidence_report['overall_confidence'] = 'low'
                confidence_report['color'] = 'red'
                confidence_report['emoji'] = '🔴'
                confidence_report['summary'] = f"⚠️ {len(contradictions)} contradiction(s) detected"

            # Compute hallucination risk score
            hallucination_risk = confidence_scorer.compute_risk_score(confidence_report, contradictions, divergence)
            
            # Format response
            formatted_response = response_formatter.format_response(
                response_text,
                verified_facts,
                contradictions
            )
            
            # Display results
            st.divider()

            # --- Hallucination Risk Score (prominent) ---
            risk = hallucination_risk
            st.markdown(f"## {risk['emoji']} Hallucination Risk: **{risk['risk_level'].upper()}** &nbsp; `{risk['risk_score']}/100`")
            risk_col1, risk_col2, risk_col3 = st.columns(3)
            risk_col1.metric("Confidence Risk", f"{risk['breakdown']['confidence_risk']}/40")
            risk_col2.metric("Contradiction Risk", f"{risk['breakdown']['contradiction_risk']}/40")
            risk_col3.metric("Divergence Risk", f"{risk['breakdown']['divergence_risk']}/20")
            st.progress(int(risk['risk_score']), text=risk['label'])
            st.divider()

            # Divergence warning
            if divergence.get('diverged'):
                st.warning(f"↗️ **Response Divergence Detected** — similarity to previous turn: `{divergence['similarity_score']}` (severity: {divergence.get('severity', 'unknown')})")

            # Contradictions warning
            if contradictions:
                st.error("⚠️ **Contradictions Detected!**")
                for cont in contradictions:
                    with st.expander(f"Contradiction: {cont['message']}", expanded=True):
                        st.write(f"**Previous:** {cont['previous_value']}")
                        st.write(f"**Current:** {cont['current_value']}")
                        st.write(f"**Difference:** {cont['difference']}")
            
            # Confidence banner
            conf_emoji = confidence_report['emoji']
            conf_level = confidence_report['overall_confidence'].upper()
            conf_summary = confidence_report['summary']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Confidence", f"{conf_emoji} {conf_level}")
            with col2:
                st.metric("Total Facts", confidence_report['stats']['total_facts'])
            with col3:
                st.metric("Verified", confidence_report['stats']['verified'])
            with col4:
                st.metric("Score", f"{confidence_report['confidence_score']:.2f}")
            
            st.info(conf_summary)
            
            # Response comparison
            st.divider()
            st.subheader("📄 Response Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Response**")
                st.write(response_text)
            
            with col2:
                st.markdown("**Verified Response (with highlights)**")
                st.markdown(formatted_response['markdown'])
            
            # Facts summary
            st.divider()
            st.subheader("📊 Fact Verification Details")
            
            if verified_facts:
                for i, fact in enumerate(verified_facts, 1):
                    # Determine color class
                    if fact.get('verified') and fact.get('confidence') == 'high':
                        color_class = 'fact-green'
                        emoji = '✅'
                    elif fact.get('verified') and fact.get('confidence') == 'medium':
                        color_class = 'fact-yellow'
                        emoji = '⚠️'
                    elif fact.get('verified') == False:
                        color_class = 'fact-red'
                        emoji = '❌'
                    else:
                        color_class = 'fact-orange'
                        emoji = '❓'

                    # Hallucination type badge
                    h_type = fact.get('hallucination_type', '')
                    type_badge_colors = {
                        'CONFIRMED': '#28a745',
                        'PARTIALLY_VERIFIED': '#ffc107',
                        'NUMERIC_MISMATCH': '#fd7e14',
                        'FABRICATED': '#dc3545',
                        'UNVERIFIABLE': '#6c757d',
                    }
                    badge_color = type_badge_colors.get(h_type, '#6c757d')
                    badge_html = f"<span style='background:{badge_color};color:white;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:bold'>{h_type}</span>" if h_type else ""

                    # Evidence snippet
                    evidence = fact.get('evidence_snippet')
                    evidence_html = f"<br><small>📄 <em>Wikipedia says:</em> \"{evidence}\"</small>" if evidence else ""

                    with st.container():
                        st.markdown(f"""
                        <div class="{color_class}">
                            <strong>{emoji} {fact['entity']}</strong> &nbsp; {badge_html}<br>
                            <small>Type: {fact['entity_type']} | Confidence: {fact.get('confidence', 'unknown')}</small><br>
                            <small>{fact.get('verification_note', 'No note')}</small>
                            {evidence_html}
                            {f"<br><a href='{fact.get('wikipedia_url')}' target='_blank'>📖 View on Wikipedia →</a>" if fact.get('wikipedia_url') else ''}
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.write("No verifiable facts found in this response.")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.exception(e)

elif verify_button and not query:
    st.warning("Please enter a question!")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: white; padding: 20px;'>
    <small>Built by Soumyashis Sarkar | Powered by Groq (LLaMA3) & Wikipedia (as database of reference)</small>
</div>
""", unsafe_allow_html=True)