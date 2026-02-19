"""
PharmaGuard: Pharmacogenomic Risk Prediction API
Complete implementation with VCF parsing, risk prediction, and LLM explanations.
"""

import os
import io
import json
import logging
from datetime import datetime, timezone
from typing import Annotated, List, Dict, Optional, Any

import requests
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------- Configuration ----------
# API_KEY = os.getenv(
#     "OPENROUTER_API_KEY",
    
# )
API_KEY = "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"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = "openrouter/free"

SYSTEM_PROMPT = (
    "You are a PharmaGuard AI Chatbot, a helpful assistant for pharmaceutical "
    "information. Do not use chars for styling. Only provide plain text. "
    "Try to give the result under 2 to 3 lines if possible."
)

DATABASE_PATH = os.path.join(os.path.dirname(__file__), "database.json")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Load pharmacogenomic database ----------
def load_database() -> dict:
    with open(DATABASE_PATH, "r") as f:
        return json.load(f)

DB = load_database()

# ---------- Pydantic Models ----------

class Data(BaseModel):
    prompt: Annotated[str, Field(description="The prompt given by the user.")]


class AllDetails(BaseModel):
    data: Annotated[str, Field(..., description="All information to generate explanation")]


class DetectedVariant(BaseModel):
    rsid: str
    gene: str
    star_allele: str
    genotype: str
    zygosity: str
    effect: str
    clinical_significance: str
    chromosome: str
    position: int
    ref: str
    alt: str
    quality: float


class RiskAssessment(BaseModel):
    risk_label: str
    confidence_score: float
    severity: str


class PharmacogenomicProfile(BaseModel):
    primary_gene: str
    diplotype: str
    phenotype: str
    activity_score: float
    detected_variants: List[DetectedVariant]


class ClinicalRecommendation(BaseModel):
    recommendation: str
    cpic_guideline: str
    monitoring: str
    alternatives: str


class LLMExplanation(BaseModel):
    summary: str
    mechanism: str
    variant_impact: str
    clinical_significance: str
    model_used: str


class QualityMetrics(BaseModel):
    vcf_parsing_success: bool
    total_variants_parsed: int
    pharmacogenomic_variants_found: int
    gene_coverage: List[str]
    missing_genes: List[str]
    confidence_factors: List[str]


class FinalOutput(BaseModel):
    patient_id: str
    drug: str
    timestamp: str
    risk_assessment: RiskAssessment
    pharmacogenomic_profile: PharmacogenomicProfile
    clinical_recommendation: ClinicalRecommendation
    llm_generated_explanation: LLMExplanation
    quality_metrics: QualityMetrics


# ---------- VCF Parser ----------

def parse_vcf(vcf_content: str) -> dict:
    """
    Parse a VCF file and extract pharmacogenomic variants.
    Returns dict with patient_id, variants list, and parsing metadata.
    """
    lines = vcf_content.strip().split("\n")
    patient_id = "PATIENT_UNKNOWN"
    variants = []
    total_lines = 0
    parse_errors = []

    for line in lines:
        # Skip meta-information lines
        if line.startswith("##"):
            continue

        # Header line — extract patient/sample ID
        if line.startswith("#CHROM"):
            fields = line.split("\t")
            if len(fields) >= 10:
                patient_id = fields[9].strip()
            continue

        # Data lines
        total_lines += 1
        fields = line.split("\t")
        if len(fields) < 8:
            parse_errors.append(f"Line {total_lines}: insufficient fields")
            continue

        try:
            chrom = fields[0].strip()
            pos = int(fields[1].strip())
            rsid = fields[2].strip()
            ref = fields[3].strip()
            alt = fields[4].strip()
            qual = float(fields[5].strip()) if fields[5].strip() != "." else 0.0
            filt = fields[6].strip()

            # Parse INFO field
            info_str = fields[7].strip()
            info = {}
            for item in info_str.split(";"):
                if "=" in item:
                    k, v = item.split("=", 1)
                    info[k] = v

            gene = info.get("GENE", "")
            star = info.get("STAR", "")
            rs = info.get("RS", rsid)

            # Parse genotype
            genotype = ""
            if len(fields) >= 10:
                gt_format = fields[8].strip()
                gt_value = fields[9].strip()
                if "GT" in gt_format:
                    genotype = gt_value

            # Determine zygosity
            if genotype in ("0/0", "0|0"):
                zygosity = "homozygous_reference"
            elif genotype in ("0/1", "1/0", "0|1", "1|0"):
                zygosity = "heterozygous"
            elif genotype in ("1/1", "1|1"):
                zygosity = "homozygous_variant"
            else:
                zygosity = "unknown"

            variant = {
                "rsid": rs if rs else rsid,
                "gene": gene,
                "star_allele": star,
                "genotype": genotype,
                "zygosity": zygosity,
                "chromosome": chrom,
                "position": pos,
                "ref": ref,
                "alt": alt,
                "quality": qual,
                "filter": filt,
            }
            variants.append(variant)

        except Exception as e:
            parse_errors.append(f"Line {total_lines}: {str(e)}")

    return {
        "patient_id": patient_id,
        "variants": variants,
        "total_lines_parsed": total_lines,
        "parse_errors": parse_errors,
    }


# ---------- Pharmacogenomic Analysis Engine ----------

def get_variant_effect(gene: str, rsid: str) -> dict:
    """Look up variant effect from the database."""
    gene_data = DB.get("genes", {}).get(gene, {})
    variant_data = gene_data.get("variants", {}).get(rsid, {})
    return variant_data


def determine_phenotype(gene: str, gene_variants: List[dict]) -> dict:
    """
    Determine the metabolizer phenotype for a gene based on detected variants.
    Uses activity score approach from CPIC.
    """
    phenotype_rules = DB.get("phenotype_rules", {}).get(gene, {})
    activity_scores_map = phenotype_rules.get("activity_scores", {})
    thresholds = phenotype_rules.get("phenotype_thresholds", {})

    # Start with two copies of normal function (*1/*1)
    allele_scores = [1.0, 1.0]  # default: two normal alleles

    # Collect all variants that are not homozygous reference
    variant_alleles = []
    for v in gene_variants:
        if v["zygosity"] == "homozygous_reference":
            continue  # no variant allele present

        db_variant = get_variant_effect(gene, v["rsid"])
        effect = db_variant.get("effect", "normal_function")
        score = activity_scores_map.get(effect, 1.0)

        if v["zygosity"] == "heterozygous":
            variant_alleles.append(score)
        elif v["zygosity"] == "homozygous_variant":
            variant_alleles.append(score)
            variant_alleles.append(score)

    # Replace default allele scores with variant scores
    # Sort variant alleles (worst first) and assign
    if len(variant_alleles) >= 2:
        allele_scores = sorted(variant_alleles)[:2]
    elif len(variant_alleles) == 1:
        allele_scores = [variant_alleles[0], 1.0]  # one variant, one normal

    total_activity = sum(allele_scores)

    # Determine phenotype from thresholds
    phenotype = "Unknown"
    for pheno, (low, high) in thresholds.items():
        if low <= total_activity <= high:
            phenotype = pheno
            break

    # Build diplotype string
    star_alleles_found = []
    for v in gene_variants:
        if v["zygosity"] != "homozygous_reference":
            star = v.get("star_allele", "")
            if star:
                if v["zygosity"] == "heterozygous":
                    star_alleles_found.append(star)
                elif v["zygosity"] == "homozygous_variant":
                    star_alleles_found.extend([star, star])

    if len(star_alleles_found) == 0:
        diplotype = "*1/*1"
    elif len(star_alleles_found) == 1:
        diplotype = f"*1/{star_alleles_found[0]}"
    else:
        diplotype = f"{star_alleles_found[0]}/{star_alleles_found[1]}"

    return {
        "phenotype": phenotype,
        "activity_score": total_activity,
        "diplotype": diplotype,
        "allele_scores": allele_scores,
    }


def assess_drug_risk(drug: str, phenotype: str) -> dict:
    """Get risk assessment for a drug based on metabolizer phenotype."""
    drug_upper = drug.upper().strip()
    drug_data = DB.get("drug_gene_mapping", {}).get(drug_upper, None)

    if not drug_data:
        return {
            "risk_label": "Unknown",
            "confidence_score": 0.0,
            "severity": "moderate",
            "recommendation": f"Drug '{drug}' not found in pharmacogenomic database.",
            "cpic_guideline": "No CPIC guideline available for this drug.",
            "primary_gene": "Unknown",
            "mechanism": "Unknown",
        }

    risk_rules = drug_data.get("risk_rules", {})
    risk = risk_rules.get(phenotype, risk_rules.get("Unknown", {}))

    return {
        "risk_label": risk.get("risk_label", "Unknown"),
        "confidence_score": risk.get("confidence", 0.3),
        "severity": risk.get("severity", "moderate"),
        "recommendation": risk.get("recommendation", "No recommendation available."),
        "cpic_guideline": risk.get("cpic_guideline", "No CPIC guideline available."),
        "primary_gene": drug_data.get("primary_gene", "Unknown"),
        "mechanism": drug_data.get("mechanism", "Unknown"),
    }


# ---------- LLM Explanation Generator ----------

def generate_llm_explanation(
    patient_id: str,
    drug: str,
    risk_data: dict,
    phenotype_data: dict,
    variants: List[dict],
) -> dict:
    """Generate a clinical explanation using the LLM."""
    detected_info = ""
    for v in variants:
        if v.get("zygosity") != "homozygous_reference":
            detected_info += (
                f"- {v['rsid']} ({v['gene']} {v.get('star_allele','')}) "
                f"genotype: {v['genotype']} ({v['zygosity']})\n"
            )

    if not detected_info:
        detected_info = "No actionable pharmacogenomic variants detected."

    prompt = f"""You are a clinical pharmacogenomics expert. Provide a clear, concise explanation for a healthcare provider.

Patient: {patient_id}
Drug: {drug}
Primary Gene: {risk_data['primary_gene']}
Diplotype: {phenotype_data['diplotype']}
Phenotype: {phenotype_data['phenotype']} (Activity Score: {phenotype_data['activity_score']})
Risk Label: {risk_data['risk_label']}
Severity: {risk_data['severity']}

Detected Variants:
{detected_info}

Drug Mechanism: {risk_data['mechanism']}
CPIC Guideline: {risk_data['cpic_guideline']}
Clinical Recommendation: {risk_data['recommendation']}

Please provide your response in EXACTLY this JSON format (no markdown, no code blocks, just raw JSON):
{{
  "summary": "A 2-3 sentence clinical summary of the pharmacogenomic finding and its clinical impact.",
  "mechanism": "Explanation of how the genetic variants affect drug metabolism at a biological level.",
  "variant_impact": "Specific description of what each detected variant does to enzyme/transporter function.",
  "clinical_significance": "What this means for the patient's treatment plan including specific actions to take."
}}"""

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://pharmaguard.app",
        "X-Title": "PharmaGuard AI",
    }

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a clinical pharmacogenomics expert. "
                    "Always respond with valid JSON only. No markdown formatting. "
                    "No code blocks. Just the raw JSON object."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }

    try:
        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=45,
        )
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"]

        # Try to parse JSON from response (handle potential markdown wrapping)
        content = content.strip()
        if content.startswith("```"):
            # Remove markdown code block
            lines = content.split("\n")
            content = "\n".join(
                l for l in lines if not l.strip().startswith("```")
            )
            content = content.strip()

        explanation = json.loads(content)
        explanation["model_used"] = LLM_MODEL
        return explanation

    except Exception as e:
        logger.error(f"LLM explanation generation failed: {e}")
        return {
            "summary": (
                f"Patient {patient_id} has {phenotype_data['phenotype']} metabolizer "
                f"status for {risk_data['primary_gene']}, resulting in "
                f"'{risk_data['risk_label']}' risk classification for {drug}. "
                f"{risk_data['recommendation']}"
            ),
            "mechanism": risk_data.get("mechanism", "Mechanism information unavailable."),
            "variant_impact": detected_info if detected_info else "No actionable variants detected.",
            "clinical_significance": risk_data.get("recommendation", "See CPIC guidelines."),
            "model_used": f"{LLM_MODEL} (fallback - LLM unavailable)",
        }


# ---------- Core Analysis Pipeline ----------

def run_analysis(vcf_content: str, drug: str) -> dict:
    """
    Full pharmacogenomic analysis pipeline.
    1. Parse VCF
    2. Identify relevant variants for the drug's primary gene
    3. Determine phenotype
    4. Assess risk
    5. Generate LLM explanation
    6. Build final output
    """
    drug_upper = drug.upper().strip()

    # Step 1: Parse VCF
    parsed = parse_vcf(vcf_content)
    patient_id = parsed["patient_id"]
    all_variants = parsed["variants"]

    # Step 2: Identify primary gene for this drug
    drug_data = DB.get("drug_gene_mapping", {}).get(drug_upper, None)
    if not drug_data:
        primary_gene = "Unknown"
    else:
        primary_gene = drug_data["primary_gene"]

    # Get all variants for the primary gene
    gene_variants = [v for v in all_variants if v["gene"] == primary_gene]

    # Enrich variants with database info
    detected_variants_enriched = []
    for v in gene_variants:
        db_info = get_variant_effect(v["gene"], v["rsid"])
        detected_variants_enriched.append(
            DetectedVariant(
                rsid=v["rsid"],
                gene=v["gene"],
                star_allele=v.get("star_allele", db_info.get("star_allele", "")),
                genotype=v["genotype"],
                zygosity=v["zygosity"],
                effect=db_info.get("effect", "unknown"),
                clinical_significance=db_info.get("clinical_significance", "Unknown"),
                chromosome=v["chromosome"],
                position=v["position"],
                ref=v["ref"],
                alt=v["alt"],
                quality=v["quality"],
            )
        )

    # Step 3: Determine phenotype
    if gene_variants:
        phenotype_result = determine_phenotype(primary_gene, gene_variants)
    else:
        phenotype_result = {
            "phenotype": "Unknown",
            "activity_score": -1.0,
            "diplotype": "Unknown",
            "allele_scores": [],
        }

    # Step 4: Risk assessment
    risk = assess_drug_risk(drug_upper, phenotype_result["phenotype"])

    # Step 5: LLM explanation
    llm_explanation = generate_llm_explanation(
        patient_id, drug_upper, risk, phenotype_result, gene_variants
    )

    # Step 6: Quality metrics
    all_target_genes = ["CYP2D6", "CYP2C19", "CYP2C9", "SLCO1B1", "TPMT", "DPYD"]
    genes_found = list(set(v["gene"] for v in all_variants if v["gene"] in all_target_genes))
    missing_genes = [g for g in all_target_genes if g not in genes_found]

    confidence_factors = []
    if parsed["parse_errors"]:
        confidence_factors.append(f"{len(parsed['parse_errors'])} parse errors encountered")
    if phenotype_result["phenotype"] == "Unknown":
        confidence_factors.append("Phenotype could not be determined")
    if not gene_variants:
        confidence_factors.append(f"No variants found for primary gene {primary_gene}")
    if len(genes_found) == len(all_target_genes):
        confidence_factors.append("Full gene panel coverage")
    if not confidence_factors:
        confidence_factors.append("Analysis completed with high confidence")

    # Step 7: Monitoring recommendations
    monitoring_map = {
        "CODEINE": "Monitor for pain relief adequacy and opioid side effects (respiratory depression, sedation, constipation).",
        "CLOPIDOGREL": "Monitor platelet function tests (P2Y12 assay). Track for cardiovascular events. Regular follow-up.",
        "WARFARIN": "Frequent INR monitoring (daily to weekly during initiation, then periodic). Watch for bleeding signs.",
        "SIMVASTATIN": "Monitor for myopathy symptoms (muscle pain, weakness). Baseline and periodic CK levels. Liver function tests.",
        "AZATHIOPRINE": "Complete blood count (CBC) weekly for first month, biweekly for months 2-3, then monthly. Liver function tests.",
        "FLUOROURACIL": "CBC with differential before each cycle. Monitor for mucositis, diarrhea, hand-foot syndrome, neurotoxicity.",
    }

    alternatives_map = {
        "CODEINE": "Morphine (direct, bypasses CYP2D6), tramadol (partial CYP2D6), non-opioid analgesics (NSAIDs, acetaminophen).",
        "CLOPIDOGREL": "Prasugrel, ticagrelor (do not require CYP2C19 activation).",
        "WARFARIN": "Direct oral anticoagulants (DOACs): rivaroxaban, apixaban, edoxaban, dabigatran.",
        "SIMVASTATIN": "Pravastatin, rosuvastatin, fluvastatin (lower SLCO1B1 dependence).",
        "AZATHIOPRINE": "Mycophenolate mofetil, methotrexate, or other immunosuppressants based on indication.",
        "FLUOROURACIL": "Alternative chemotherapy regimens based on tumor type and oncology guidelines.",
    }

    # Build final output
    timestamp = datetime.now(timezone.utc).isoformat()

    final = FinalOutput(
        patient_id=patient_id,
        drug=drug_upper,
        timestamp=timestamp,
        risk_assessment=RiskAssessment(
            risk_label=risk["risk_label"],
            confidence_score=risk["confidence_score"],
            severity=risk["severity"],
        ),
        pharmacogenomic_profile=PharmacogenomicProfile(
            primary_gene=primary_gene,
            diplotype=phenotype_result["diplotype"],
            phenotype=phenotype_result["phenotype"],
            activity_score=phenotype_result["activity_score"],
            detected_variants=detected_variants_enriched,
        ),
        clinical_recommendation=ClinicalRecommendation(
            recommendation=risk["recommendation"],
            cpic_guideline=risk["cpic_guideline"],
            monitoring=monitoring_map.get(drug_upper, "Standard clinical monitoring."),
            alternatives=alternatives_map.get(drug_upper, "Consult clinical guidelines for alternatives."),
        ),
        llm_generated_explanation=LLMExplanation(
            summary=llm_explanation.get("summary", ""),
            mechanism=llm_explanation.get("mechanism", ""),
            variant_impact=llm_explanation.get("variant_impact", ""),
            clinical_significance=llm_explanation.get("clinical_significance", ""),
            model_used=llm_explanation.get("model_used", LLM_MODEL),
        ),
        quality_metrics=QualityMetrics(
            vcf_parsing_success=len(parsed["parse_errors"]) == 0,
            total_variants_parsed=parsed["total_lines_parsed"],
            pharmacogenomic_variants_found=len(
                [v for v in all_variants if v["gene"] in all_target_genes]
            ),
            gene_coverage=genes_found,
            missing_genes=missing_genes,
            confidence_factors=confidence_factors,
        ),
    )

    return final.model_dump()


# ---------- FastAPI App ----------

app = FastAPI(
    title="PharmaGuard AI — Pharmacogenomic Risk Prediction API",
    description="Analyze VCF files and predict drug-specific pharmacogenomic risks.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "PharmaGuard AI API is running.",
        "endpoints": {
            "/chat": "POST - AI chatbot",
            "/final-data": "POST - Full pharmacogenomic analysis (upload VCF + drug name)",
            "/analyze-text": "POST - Analysis with VCF content as text",
            "/supported-drugs": "GET - List of supported drugs",
        },
    }


@app.get("/supported-drugs")
async def supported_drugs():
    """Return list of supported drugs and their primary genes."""
    drugs = ["CODEINE", "WARFARIN", "CLOPIDOGREL", "SIMVASTATIN", "AZATHIOPRINE", "FLUOROURACIL"
]
    # drugs = {}
    # for drug_name, drug_info in DB.get("drug_gene_mapping", {}).items():
    #     drugs[drug_name] = {
    #         "primary_gene": drug_info["primary_gene"],
    #         "mechanism_summary": drug_info["mechanism"][:100] + "...",
    #     }
    return {"supported_drugs": drugs}


# ========== MAIN ENDPOINT ==========

@app.post("/final-data", response_model=None)
async def final_data(
    vcf_file: UploadFile = File(..., description="VCF file (.vcf format, max 5MB)"),
    drug: str = Form(..., description="Drug name (Supported:>  CODEINE, WARFARIN, CLOPIDOGREL,SIMVASTATIN, AZATHIOPRINE, FLUOROURACIL ). Comma-separated for multiple."),
):
    """
    Main pharmacogenomic analysis endpoint.

    Upload a VCF file and specify drug name(s) to receive:
    - Risk assessment with confidence scores
    - Pharmacogenomic profile (diplotype, phenotype, detected variants)
    - Clinical recommendations aligned with CPIC guidelines
    - LLM-generated clinical explanation
    - Quality metrics

    Supports multiple drugs via comma separation.
    """

    # Validate file extension
    """if not vcf_file.filename.lower().endswith(".vcf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Please upload a .vcf file.",
        )"""

    # Validate file size (5 MB limit)
    contents = await vcf_file.read()
    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds 5 MB limit.",
        )

    try:
        vcf_content = contents.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is not valid UTF-8 text. Please check the VCF file encoding.",
        )

    # Validate VCF format
    if "##fileformat=VCF" not in vcf_content and "#CHROM" not in vcf_content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File does not appear to be a valid VCF file. Missing VCF header.",
        )

    # Parse drug names (comma-separated)
    drug_names = [d.strip().upper() for d in drug.split(",") if d.strip()]
    if not drug_names:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No drug name provided.",
        )

    # Run analysis for each drug
    results = []
    for drug_name in drug_names:
        try:
            result = run_analysis(vcf_content, drug_name)
            results.append(result)
        except Exception as e:
            logger.error(f"Analysis failed for drug {drug_name}: {e}")
            
            results.append(
                {
                    "patient_id": "UNKNOWN",
                    "drug": drug_name,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "error": str(e),
                    "risk_assessment": {
                        "risk_label": "Unknown",
                        "confidence_score": 0.0,
                        "severity": "moderate",
                    },
                }
            )

    # Return single result if one drug, array if multiple
    if len(results) == 1:
        return results[0]
    return {"analyses": results, "total_drugs_analyzed": len(results)}


# ========== TEXT-BASED ANALYSIS (alternative to file upload) ==========

class TextAnalysisRequest(BaseModel):
    vcf_content: str = Field(..., description="Raw VCF file content as text")
    drug: str = Field(..., description="Drug name or comma-separated drug names")


@app.post("/analyze-text")
async def analyze_text(request: TextAnalysisRequest):
    """
    Alternative analysis endpoint that accepts VCF content as text.
    Useful for testing and programmatic access.
    """
    if "#CHROM" not in request.vcf_content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid VCF content. Missing #CHROM header line.",
        )

    drug_names = [d.strip().upper() for d in request.drug.split(",") if d.strip()]
    results = []
    for drug_name in drug_names:
        result = run_analysis(request.vcf_content, drug_name)
        results.append(result)

    if len(results) == 1:
        return results[0]
    return {"analyses": results, "total_drugs_analyzed": len(results)}


# ========== CHAT ENDPOINT ==========

@app.post("/chat")
async def chat(request: Data):
    """AI chatbot for pharmaceutical questions."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://pharmaguard.app",
        "X-Title": "PharmaGuard AI Chatbot",
    }

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": request.prompt},
        ],
    }

    try:
        response = requests.post(
            OPENROUTER_URL, headers=headers, json=payload, timeout=30
        )
        response.raise_for_status()
        result = response.json()
        return {"response": result["choices"][0]["message"]["content"]}
    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="AI service timed out.",
        )
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI service error: {str(e)}",
        )
    except (KeyError, json.JSONDecodeError):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Invalid response from AI service.",
        )


# ========== LLM EXPLANATION ENDPOINT ==========

@app.post("/llm_generated_explanation")
async def llm_generated_explanation(data: AllDetails):
    """Generate LLM explanation from provided clinical data."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://pharmaguard.app",
        "X-Title": "PharmaGuard AI",
    }

    prompt = (
        f"You are a clinical pharmacogenomics expert. Based on the following data, "
        f"provide a clear clinical explanation in plain text:\n\n{data.data}"
    )

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a clinical pharmacogenomics expert. "
                    "Provide clear, concise explanations in plain text."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }

    try:
        response = requests.post(
            OPENROUTER_URL, headers=headers, json=payload, timeout=30
        )
        response.raise_for_status()
        result = response.json()
        return {"response": result["choices"][0]["message"]["content"]}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM service error: {str(e)}",
        )


# ---------- Run ----------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)