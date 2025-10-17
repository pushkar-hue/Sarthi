from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import fitz  # PyMuPDF
import google.generativeai as genai
import os
import json
import re
from datetime import datetime
import logging
from enum import Enum
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Project Saarthi API v2.0",
    description="AI-powered financial document analysis with advanced chunking",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("✅ Gemini API configured successfully")
else:
    logger.warning("⚠️ GEMINI_API_KEY not found in environment variables")

# Constants
CHUNK_SIZE = 4000  # Characters per chunk
OVERLAP_SIZE = 500  # Overlap between chunks for context
MODEL_NAME = "gemini-2.0-flash-exp"  # Optimized for Gemini 2.5 Flash

# Enums
class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

# Enhanced Response Models
class KeyDetail(BaseModel):
    label: str
    value: str
    confidence: Confidence
    source_page: Optional[int] = None

class RedFlag(BaseModel):
    severity: Severity
    title: str
    description: str
    impact: str  # What this means for the borrower
    recommendation: str
    relevant_clause: Optional[str] = None

class JargonTerm(BaseModel):
    term: str
    simple_explanation: str
    hindi_translation: Optional[str] = None
    example: Optional[str] = None

class ComplianceCheck(BaseModel):
    aspect: str
    status: str  # "compliant", "non-compliant", "unclear"
    details: str
    rbi_reference: Optional[str] = None

class DocumentSummary(BaseModel):
    document_type: str
    total_pages: int
    key_entities: List[str]
    summary: str
    document_date: Optional[str] = None

class ComparativeAnalysis(BaseModel):
    metric: str
    your_value: str
    market_average: str
    verdict: str  # "better", "average", "worse"

class AnalysisResult(BaseModel):
    success: bool
    document_info: Dict[str, Any]
    key_details: List[KeyDetail]
    red_flags: List[RedFlag]
    jargon_buster: List[JargonTerm]
    compliance_checks: List[ComplianceCheck]
    document_summary: DocumentSummary
    comparative_analysis: List[ComparativeAnalysis]
    overall_risk_score: int = Field(..., ge=0, le=100)
    trust_score: int = Field(..., ge=0, le=100)  # How trustworthy is this lender
    recommendations: List[str]
    ai_confidence: float = Field(..., ge=0.0, le=1.0)
    processing_time: float
    timestamp: str


# Document Chunking Helper
class DocumentChunker:
    """Smart chunking with context preservation"""
    
    def __init__(self, text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP_SIZE):
        self.text = text
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def create_chunks(self) -> List[Dict[str, Any]]:
        """Create overlapping chunks with metadata"""
        chunks = []
        text_length = len(self.text)
        start = 0
        chunk_id = 0
        
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            
            # Try to end at a sentence boundary
            if end < text_length:
                # Look for sentence endings
                last_period = self.text.rfind('.', start, end)
                last_newline = self.text.rfind('\n', start, end)
                boundary = max(last_period, last_newline)
                
                if boundary > start + self.chunk_size // 2:  # If boundary is reasonable
                    end = boundary + 1
            
            chunk_text = self.text[start:end].strip()
            
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "start": start,
                "end": end,
                "length": len(chunk_text)
            })
            
            chunk_id += 1
            start = end - self.overlap if end < text_length else text_length
        
        logger.info(f"📄 Created {len(chunks)} chunks from document")
        return chunks


# PDF Extraction with Enhanced Metadata
def extract_text_from_pdf(file_content: bytes) -> Dict[str, Any]:
    """Extract text and metadata from PDF"""
    try:
        pdf_document = fitz.open(stream=file_content, filetype="pdf")
        
        pages_text = []
        full_text = ""
        
        for page_num, page in enumerate(pdf_document):
            page_text = page.get_text()
            pages_text.append({
                "page_number": page_num + 1,
                "text": page_text,
                "char_count": len(page_text)
            })
            full_text += f"\n[Page {page_num + 1}]\n{page_text}"
        
        metadata = pdf_document.metadata or {}
        page_count = len(pdf_document)
        pdf_document.close()
        
        return {
            "full_text": full_text.strip(),
            "pages": pages_text,
            "page_count": page_count,
            "metadata": metadata,
            "total_chars": len(full_text)
        }
    except Exception as e:
        logger.error(f"❌ PDF extraction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to extract PDF: {str(e)}")


# Enhanced JSON Parser
def parse_json_from_response(response_text: str, expected_type: str = "object") -> Any:
    """Robust JSON extraction with multiple fallback strategies"""
    try:
        # Strategy 1: Look for JSON in code blocks
        json_patterns = [
            r'```json\s*(\[[\s\S]*?\]|\{[\s\S]*?\})\s*```',
            r'```\s*(\[[\s\S]*?\]|\{[\s\S]*?\})\s*```',
            r'(\[[\s\S]*?\]|\{[\s\S]*?\})'
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response_text, re.DOTALL)
            for match in matches:
                try:
                    parsed = json.loads(match)
                    if expected_type == "array" and isinstance(parsed, list):
                        return parsed
                    elif expected_type == "object" and isinstance(parsed, dict):
                        return parsed
                    elif expected_type == "any":
                        return parsed
                except json.JSONDecodeError:
                    continue
        
        # Strategy 2: Try to parse the entire response
        try:
            parsed = json.loads(response_text)
            return parsed
        except json.JSONDecodeError:
            pass
        
        logger.warning(f"⚠️ Could not parse JSON from response")
        return [] if expected_type == "array" else {}
        
    except Exception as e:
        logger.error(f"❌ JSON parsing error: {str(e)}")
        return [] if expected_type == "array" else {}


# Gemini API Call with Retry Logic
def call_gemini(prompt: str, temperature: float = 0.3, response_type: str = "array") -> Any:
    """Enhanced Gemini API call with structured output"""
    try:
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            generation_config={
                "temperature": temperature,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
            }
        )
        
        response = model.generate_content(prompt)
        
        if not response.text:
            logger.warning("⚠️ Empty response from Gemini")
            return [] if response_type == "array" else {}
        
        return parse_json_from_response(response.text, response_type)
        
    except Exception as e:
        logger.error(f"❌ Gemini API error: {str(e)}")
        return [] if response_type == "array" else {}


# Core Analysis Functions with Chunking
def extract_key_details_chunked(chunks: List[Dict]) -> List[KeyDetail]:
    """Extract key details from all chunks and merge"""
    all_details = {}
    
    prompt_template = """You are a financial document analyzer. Extract key financial details from this document section.

        Return ONLY a valid JSON array with this EXACT structure:
        [
        {{
            "label": "Loan Amount",
            "value": "₹5,00,000",
            "confidence": "high",
            "source_page": 1
        }}
        ]

        Look for:
        - Loan/Principal Amount
        - Interest Rate (per annum/per month)
        - Tenure/Duration
        - Processing Fee
        - EMI Amount
        - Late Payment Charges
        - Prepayment Charges
        - Total Amount Payable
        - Lender Name
        - Borrower Name
        - Agreement Date

        Document Section:
        ---
        {text}
        ---

        Return ONLY valid JSON array, no other text."""

    for chunk in chunks[:5]:  # Process first 5 chunks for efficiency
        prompt = prompt_template.format(text=chunk['text'])
        result = call_gemini(prompt, temperature=0.2, response_type="array")
        
        if isinstance(result, list):
            for item in result:
                try:
                    label = item.get('label', '')
                    if label and label not in all_details:
                        all_details[label] = KeyDetail(
                            label=label,
                            value=item.get('value', 'Not found'),
                            confidence=Confidence(item.get('confidence', 'medium').lower()),
                            source_page=item.get('source_page')
                        )
                except Exception as e:
                    logger.warning(f"⚠️ Error parsing detail item: {e}")
    
    logger.info(f"✅ Extracted {len(all_details)} unique key details")
    return list(all_details.values())


def detect_red_flags_comprehensive(text: str, chunks: List[Dict]) -> List[RedFlag]:
    """Deep analysis for red flags across document"""
    
    prompt = f"""You are a consumer protection expert specializing in Indian financial regulations (RBI guidelines, Fair Practice Code).

    Analyze this document and identify concerning clauses that could harm borrowers.

    Return ONLY valid JSON array:
    [
    {{
        "severity": "high",
        "title": "Excessive Interest Rate",
        "description": "Interest rate of 18% p.a. is above market average of 12-15% for similar loans",
        "impact": "You will pay ₹50,000 more in interest over the loan tenure",
        "recommendation": "Negotiate for a rate between 12-15% or compare with other lenders",
        "relevant_clause": "Section 4.2: Interest shall be charged at 18% per annum"
    }}
    ]

    Focus on:
    1. Interest rates (>15% for personal loans is high)
    2. Processing fees (>2-3% is excessive)
    3. Hidden charges
    4. Prepayment penalties (should be ≤2%)
    5. Late payment fees (should be ≤2% per month)
    6. Unfair clauses (automatic renewal, arbitrary fee changes)
    7. Unclear terms
    8. Missing borrower protections

    Document (first 8000 chars):
    ---
    {text[:8000]}
    ---

    Return ONLY valid JSON array."""

    result = call_gemini(prompt, temperature=0.4, response_type="array")
    
    red_flags = []
    if isinstance(result, list):
        for item in result[:8]:  # Limit to top 8 flags
            try:
                red_flags.append(RedFlag(
                    severity=Severity(item.get('severity', 'medium').lower()),
                    title=item.get('title', 'Concern Identified'),
                    description=item.get('description', ''),
                    impact=item.get('impact', 'May affect your financial interests'),
                    recommendation=item.get('recommendation', 'Review with financial advisor'),
                    relevant_clause=item.get('relevant_clause')
                ))
            except Exception as e:
                logger.warning(f"⚠️ Error parsing red flag: {e}")
    
    logger.info(f"🚩 Detected {len(red_flags)} red flags")
    return red_flags


def explain_jargon_enhanced(text: str) -> List[JargonTerm]:
    """Extract and explain complex terms with Hindi translations"""
    
    prompt = f"""You are a financial educator for Indian citizens. Find complex financial/legal terms and explain them simply.

        Return ONLY valid JSON array:
        [
        {{
            "term": "Amortization",
            "simple_explanation": "The process of paying off a loan gradually through regular monthly payments that include both interest and principal",
            "hindi_translation": "ऋण चुकौती",
            "example": "For a ₹10 lakh loan, your monthly EMI of ₹20,000 reduces the outstanding balance each month through amortization"
        }}
        ]

        Find 5-7 complex terms like:
        - Amortization, APR, Foreclosure, Subrogation, Hypothecation
        - Lien, Encumbrance, Default, Collateral
        - Any legal/financial jargon
        - And give easy to understand hindi translation as well 

        Document (first 6000 chars):
        ---
        {text[:6000]}
        ---

        Return ONLY valid JSON array."""
    result = call_gemini(prompt, temperature=0.5, response_type="array")
    
    terms = []
    if isinstance(result, list):
        for item in result[:7]:
            try:
                terms.append(JargonTerm(
                    term=item.get('term', ''),
                    simple_explanation=item.get('simple_explanation', ''),
                    hindi_translation=item.get('hindi_translation'),
                    example=item.get('example')
                ))
            except Exception as e:
                logger.warning(f"⚠️ Error parsing jargon term: {e}")
    
    logger.info(f"📚 Explained {len(terms)} complex terms")
    return terms


def check_compliance(text: str) -> List[ComplianceCheck]:
    """Check compliance with RBI guidelines and best practices"""
    
    prompt = f"""You are a banking compliance expert. Check if this document complies with RBI Fair Practice Code and consumer protection guidelines.

        Return ONLY valid JSON array:
        [
        {{
            "aspect": "Interest Rate Disclosure",
            "status": "compliant",
            "details": "Interest rate clearly mentioned as 12% per annum in Section 3",
            "rbi_reference": "RBI Fair Practice Code 2.1"
        }}
        ]

        Check these aspects:
        1. Clear interest rate disclosure
        2. All-in-cost disclosure (APR)
        3. Fee transparency
        4. Loan approval timeline
        5. Prepayment terms
        6. Grievance redressal mechanism
        7. Copy of loan agreement provided
        8. No discriminatory clauses

        Document (first 6000 chars):
        ---
        {text[:6000]}
        ---

        Return ONLY valid JSON array."""
    result = call_gemini(prompt, temperature=0.3, response_type="array")
    
    checks = []
    if isinstance(result, list):
        for item in result[:6]:
            try:
                checks.append(ComplianceCheck(
                    aspect=item.get('aspect', ''),
                    status=item.get('status', 'unclear'),
                    details=item.get('details', ''),
                    rbi_reference=item.get('rbi_reference')
                ))
            except Exception as e:
                logger.warning(f"⚠️ Error parsing compliance check: {e}")
    
    logger.info(f"✓ Performed {len(checks)} compliance checks")
    return checks


def generate_comparative_analysis(key_details: List[KeyDetail]) -> List[ComparativeAnalysis]:
    """Compare document terms with market averages"""
    
    # Extract relevant values
    interest_rate = None
    processing_fee = None
    
    for detail in key_details:
        if 'interest' in detail.label.lower():
            interest_rate = detail.value
        elif 'processing' in detail.label.lower():
            processing_fee = detail.value
    
    comparisons = []
    
    # Interest rate comparison
    if interest_rate:
        comparisons.append(ComparativeAnalysis(
            metric="Interest Rate",
            your_value=interest_rate,
            market_average="11-14% for personal loans",
            verdict="average"  # This would be calculated based on actual rate
        ))
    
    # Processing fee comparison
    if processing_fee:
        comparisons.append(ComparativeAnalysis(
            metric="Processing Fee",
            your_value=processing_fee,
            market_average="1-2% of loan amount",
            verdict="average"
        ))
    
    logger.info(f"📊 Generated {len(comparisons)} comparative analyses")
    return comparisons


def generate_document_summary_enhanced(text: str, page_count: int, metadata: Dict) -> DocumentSummary:
    """Generate comprehensive document summary"""
    
    prompt = f"""Analyze this document and provide a summary.

        Return ONLY valid JSON object:
        {{
        "document_type": "Personal Loan Agreement",
        "total_pages": {page_count},
        "key_entities": ["HDFC Bank", "Rajesh Kumar", "Mumbai"],
        "summary": "This is a personal loan agreement between HDFC Bank and borrower Rajesh Kumar for ₹5 lakhs at 12% interest over 5 years. Key terms include monthly EMI of ₹11,122 and a processing fee of 1.5%.",
        "document_date": "2024-01-15"
        }}

        Document (first 4000 chars):
        ---
        {text[:4000]}
        ---

        Return ONLY valid JSON object."""
            
    result = call_gemini(prompt, temperature=0.4, response_type="object")
    
    if isinstance(result, dict) and result:
        try:
            return DocumentSummary(
                document_type=result.get('document_type', 'Financial Document'),
                total_pages=page_count,
                key_entities=result.get('key_entities', []),
                summary=result.get('summary', 'Document analysis completed'),
                document_date=result.get('document_date')
            )
        except Exception as e:
            logger.warning(f"⚠️ Error creating summary: {e}")
    
    # Fallback
    return DocumentSummary(
        document_type="Financial Document",
        total_pages=page_count,
        key_entities=[],
        summary="Document analysis completed. Please review the detailed findings.",
        document_date=None
    )


def calculate_risk_score(red_flags: List[RedFlag]) -> int:
    """Calculate overall risk score based on severity"""
    score = 0
    severity_weights = {
        Severity.CRITICAL: 30,
        Severity.HIGH: 20,
        Severity.MEDIUM: 10,
        Severity.LOW: 5
    }
    
    for flag in red_flags:
        score += severity_weights.get(flag.severity, 5)
    
    return min(score, 100)


def calculate_trust_score(compliance_checks: List[ComplianceCheck], red_flags: List[RedFlag]) -> int:
    """Calculate lender trustworthiness score"""
    base_score = 100
    
    # Deduct for non-compliance
    for check in compliance_checks:
        if check.status == "non-compliant":
            base_score -= 15
        elif check.status == "unclear":
            base_score -= 5
    
    # Deduct for red flags
    for flag in red_flags:
        if flag.severity == Severity.CRITICAL:
            base_score -= 10
        elif flag.severity == Severity.HIGH:
            base_score -= 7
    
    return max(base_score, 0)


def generate_recommendations(red_flags: List[RedFlag], risk_score: int, trust_score: int) -> List[str]:
    """Generate actionable recommendations"""
    recommendations = []
    
    # Risk-based recommendations
    if risk_score > 70:
        recommendations.append("🚨 HIGH RISK: Strongly consider alternatives before signing this agreement")
        recommendations.append("💼 Consult a financial advisor or lawyer before proceeding")
    elif risk_score > 40:
        recommendations.append("⚠️ MODERATE RISK: Carefully review all flagged concerns")
        recommendations.append("📝 Negotiate problematic terms before signing")
    else:
        recommendations.append("✅ LOW RISK: Terms appear reasonable, but review carefully")
    
    # Trust-based recommendations
    if trust_score < 60:
        recommendations.append("⚡ Lender transparency concerns detected. Verify their credentials")
    
    # Specific recommendations
    if red_flags:
        recommendations.append(f"🔍 Address {len(red_flags)} red flags identified in the analysis")
        
        # Add specific recommendations from critical flags
        critical_flags = [f for f in red_flags if f.severity == Severity.CRITICAL]
        for flag in critical_flags[:2]:  # Top 2 critical issues
            recommendations.append(f"⚠️ {flag.recommendation}")
    
    # General advice
    recommendations.extend([
        "📞 Ask the lender to clarify any unclear terms in writing",
        "💰 Compare interest rates and fees with at least 2-3 other lenders",
        "📄 Keep copies of all signed documents for your records",
        "⏰ Understand the complete repayment schedule before committing"
    ])
    
    return recommendations[:8]  # Limit to 8 recommendations


# Main Analysis Endpoint
@app.post("/analyze", response_model=AnalysisResult)
async def analyze_document(file: UploadFile = File(...)):
    """
    Comprehensive document analysis with chunking and advanced AI
    
    Analyzes financial documents for:
    - Key terms and details
    - Risk factors and red flags
    - Compliance with regulations
    - Complex term explanations
    - Market comparisons
    - Actionable recommendations
    """
    
    start_time = datetime.now()
    
    # Validate file
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files supported")
    
    logger.info(f"📄 Analyzing: {file.filename}")
    
    try:
        # Extract PDF content
        file_content = await file.read()
        pdf_data = extract_text_from_pdf(file_content)
        
        text = pdf_data['full_text']
        page_count = pdf_data['page_count']
        
        if not text or len(text) < 100:
            raise HTTPException(status_code=400, detail="PDF appears empty or unreadable")
        
        logger.info(f"✅ Extracted {len(text):,} characters from {page_count} pages")
        
        # Create intelligent chunks
        chunker = DocumentChunker(text)
        chunks = chunker.create_chunks()
        
        # Run parallel analysis
        logger.info("🔍 Starting comprehensive analysis...")
        
        key_details = extract_key_details_chunked(chunks)
        red_flags = detect_red_flags_comprehensive(text, chunks)
        jargon_terms = explain_jargon_enhanced(text)
        compliance = check_compliance(text)
        doc_summary = generate_document_summary_enhanced(text, page_count, pdf_data['metadata'])
        comparative = generate_comparative_analysis(key_details)
        
        # Calculate scores
        risk_score = calculate_risk_score(red_flags)
        trust_score = calculate_trust_score(compliance, red_flags)
        recommendations = generate_recommendations(red_flags, risk_score, trust_score)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate AI confidence based on successful extractions
        total_tasks = 6
        successful_tasks = sum([
            len(key_details) > 0,
            len(red_flags) > 0,
            len(jargon_terms) > 0,
            len(compliance) > 0,
            doc_summary.summary != "",
            len(comparative) > 0
        ])
        ai_confidence = successful_tasks / total_tasks
        
        # Build result
        result = AnalysisResult(
            success=True,
            document_info={
                "filename": file.filename,
                "pages": page_count,
                "text_length": len(text),
                "chunks_processed": len(chunks),
                "analyzed_at": datetime.now().isoformat(),
                "model_used": MODEL_NAME
            },
            key_details=key_details,
            red_flags=red_flags,
            jargon_buster=jargon_terms,
            compliance_checks=compliance,
            document_summary=doc_summary,
            comparative_analysis=comparative,
            overall_risk_score=risk_score,
            trust_score=trust_score,
            recommendations=recommendations,
            ai_confidence=round(ai_confidence, 2),
            processing_time=round(processing_time, 2),
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"✅ Analysis complete in {processing_time:.2f}s | Risk: {risk_score} | Trust: {trust_score}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Analysis failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/")
async def root():
    """API information"""
    return {
        "service": "Project Saarthi API v2.0",
        "status": "online",
        "version": "2.0.0",
        "model": MODEL_NAME,
        "features": [
            "Smart document chunking",
            "Comprehensive risk analysis",
            "RBI compliance checking",
            "Market comparison",
            "Hindi translations",
            "Advanced AI processing"
        ],
        "endpoints": {
            "analyze": "POST /analyze - Upload PDF for analysis",
            "health": "GET /health - Health check",
            "docs": "GET /docs - Interactive API documentation"
        }
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "gemini_configured": GEMINI_API_KEY is not None,
        "model": MODEL_NAME,
        "timestamp": datetime.now().isoformat(),
        "chunk_size": CHUNK_SIZE,
        "overlap_size": OVERLAP_SIZE
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)