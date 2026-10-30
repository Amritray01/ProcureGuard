"""Dynamic LLM Integration Layer supporting Groq, Google Gemini, OpenAI, and Local Fallbacks.

Leverages Pydantic structured output models with explicit field descriptions to enforce
well-formed JSON schemas for invoice extraction and contract clause analysis.
"""

from decimal import Decimal
import json
import os
from typing import Any, Dict, List, Optional

from src.core.schema import ContractClauseAnalysisSchema, InvoiceExtractionSchema


class LLMClient:
    """Dynamic multi-provider LLM client with structured Pydantic format enforcement."""

    def __init__(self) -> None:
        self.groq_key = (os.getenv("GROQ_API_KEY") or "").strip() or None
        self.gemini_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip() or None
        self.openai_key = (os.getenv("OPENAI_API_KEY") or "").strip() or None

        # Candidate models for each provider (first working model is chosen)
        user_groq_model = os.getenv("GROQ_MODEL")
        self.groq_models: List[str] = [m for m in [
            user_groq_model,
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b",
        ] if m]

        user_gemini_model = os.getenv("GEMINI_MODEL")
        self.gemini_models: List[str] = [m for m in [
            user_gemini_model,
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
        ] if m]

        self.openai_model = os.getenv("OPENAI_MODEL") or "gpt-4o-mini"

    def has_api_key(self) -> bool:
        """Check if any LLM API provider key is configured."""
        return bool(self.groq_key or self.gemini_key or self.openai_key)

    def get_active_provider(self) -> str:
        """Return the highest priority configured provider."""
        if self.groq_key:
            return "groq"
        if self.gemini_key:
            return "gemini"
        if self.openai_key:
            return "openai"
        return "none"

    def extract_invoice_dynamic(self, raw_text: str) -> Dict[str, Any]:
        """Dynamically extract structured invoice data using LLM structured output.
        
        Tries Groq -> Gemini -> OpenAI -> Local Fallback in order.
        Uses Pydantic InvoiceExtractionSchema to enforce strict typing.
        """
        # 1. Groq Path
        if self.groq_key:
            try:
                from groq import Groq
                client = Groq(api_key=self.groq_key)
                schema_json = json.dumps(InvoiceExtractionSchema.model_json_schema(), indent=2)
                prompt = (
                    "You are an expert procurement invoice extractor. Extract structured invoice fields "
                    "from the document text into valid JSON matching this exact schema:\n"
                    f"{schema_json}\n\n"
                    f"Document Text:\n{raw_text[:4000]}"
                )

                for model in self.groq_models:
                    try:
                        resp = client.chat.completions.create(
                            model=model,
                            messages=[
                                {
                                    "role": "system",
                                    "content": "You are a professional invoice parser. Always return valid, well-formed JSON only.",
                                },
                                {"role": "user", "content": prompt},
                            ],
                            response_format={"type": "json_object"},
                            temperature=0.0,
                        )
                        raw_content = resp.choices[0].message.content or "{}"
                        parsed = json.loads(raw_content)
                        validated = InvoiceExtractionSchema(**parsed)
                        data = validated.model_dump()
                        if data.get("lines"):
                            return data
                    except Exception as model_err:
                        # Try next Groq model candidate
                        continue
            except Exception as e:
                print(f"[LLM Warning] Groq dynamic extraction failed: {e}. Trying next provider.")

        # 2. Google Gemini Path
        if self.gemini_key:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                for model in self.gemini_models:
                    try:
                        llm = ChatGoogleGenerativeAI(
                            model=model,
                            google_api_key=self.gemini_key,
                            temperature=0.0,
                        )
                        structured_llm = llm.with_structured_output(InvoiceExtractionSchema)
                        result = structured_llm.invoke(f"Extract invoice data from text:\n{raw_text[:4000]}")
                        if isinstance(result, InvoiceExtractionSchema):
                            return result.model_dump()
                        elif isinstance(result, dict) and result.get("lines"):
                            return result
                    except Exception:
                        continue
            except Exception as e:
                print(f"[LLM Warning] Gemini extraction error: {e}. Trying next provider.")

        # 3. OpenAI Path
        if self.openai_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.openai_key)
                schema_json = json.dumps(InvoiceExtractionSchema.model_json_schema(), indent=2)
                prompt = (
                    "Extract structured invoice fields from the document text into valid JSON matching this schema:\n"
                    f"{schema_json}\n\n"
                    f"Document Text:\n{raw_text[:4000]}"
                )
                resp = client.chat.completions.create(
                    model=self.openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                parsed = json.loads(resp.choices[0].message.content or "{}")
                validated = InvoiceExtractionSchema(**parsed)
                return validated.model_dump()
            except Exception as e:
                print(f"[LLM Warning] OpenAI extraction error: {e}. Falling back to visual extractor.")

        # Return empty dict so pdf_extractor handles parsing via visual horizontal coordinates
        return {}

    def analyze_contract_clause(
        self,
        invoice_summary: str,
        clause_text: str,
        surcharge_amount: Decimal,
    ) -> Dict[str, Any]:
        """Evaluate contract clause compliance using structured Pydantic output.
        
        Tries Groq -> Gemini -> OpenAI -> Local Semantic Fallback in order.
        """
        # 1. Groq Path
        if self.groq_key:
            try:
                from groq import Groq
                client = Groq(api_key=self.groq_key)
                schema_json = json.dumps(ContractClauseAnalysisSchema.model_json_schema(), indent=2)
                prompt = (
                    "Analyze whether the billed surcharge or fee violates the contract clause. "
                    f"Respond with JSON matching this exact schema:\n{schema_json}\n\n"
                    f"Contract Clause:\n{clause_text}\n\n"
                    f"Invoice Context:\n{invoice_summary}\n\n"
                    f"Charge in question: ${surcharge_amount}"
                )

                for model in self.groq_models:
                    try:
                        resp = client.chat.completions.create(
                            model=model,
                            messages=[
                                {
                                    "role": "system",
                                    "content": "You are a legal contract auditor. Always respond with valid JSON matching the schema.",
                                },
                                {"role": "user", "content": prompt},
                            ],
                            response_format={"type": "json_object"},
                            temperature=0.0,
                        )
                        raw_content = resp.choices[0].message.content or "{}"
                        parsed = json.loads(raw_content)
                        validated = ContractClauseAnalysisSchema(**parsed)
                        return validated.model_dump()
                    except Exception:
                        continue
            except Exception as e:
                print(f"[LLM Warning] Groq contract analysis error: {e}. Trying next provider.")

        # 2. Google Gemini Path
        if self.gemini_key:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                for model in self.gemini_models:
                    try:
                        llm = ChatGoogleGenerativeAI(
                            model=model,
                            google_api_key=self.gemini_key,
                            temperature=0.0,
                        )
                        structured_llm = llm.with_structured_output(ContractClauseAnalysisSchema)
                        prompt = (
                            f"Evaluate contract compliance for this invoice.\n"
                            f"Clause:\n{clause_text}\n\n"
                            f"Invoice: {invoice_summary}\n"
                            f"Charge: ${surcharge_amount}"
                        )
                        result = structured_llm.invoke(prompt)
                        if isinstance(result, ContractClauseAnalysisSchema):
                            return result.model_dump()
                        elif isinstance(result, dict):
                            return result
                    except Exception:
                        continue
            except Exception as e:
                print(f"[LLM Warning] Gemini contract analysis error: {e}. Trying next provider.")

        # 3. OpenAI Path
        if self.openai_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.openai_key)
                schema_json = json.dumps(ContractClauseAnalysisSchema.model_json_schema(), indent=2)
                prompt = (
                    "Analyze whether the invoice charge violates the contract clause. "
                    f"Respond with JSON matching this schema:\n{schema_json}\n\n"
                    f"Contract Clause:\n{clause_text}\n\n"
                    f"Invoice Context:\n{invoice_summary}\n\n"
                    f"Charge in question: ${surcharge_amount}"
                )
                resp = client.chat.completions.create(
                    model=self.openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                parsed = json.loads(resp.choices[0].message.content or "{}")
                validated = ContractClauseAnalysisSchema(**parsed)
                return validated.model_dump()
            except Exception as e:
                print(f"[LLM Warning] OpenAI contract analysis error: {e}. Falling back to semantic analyzer.")

        # Robust semantic fallback: lowercases text and inspects terms
        clause_lower = clause_text.lower()
        prohibition_terms = ["not bill", "shall not", "prohibit", "prohibited", "included", "no separate", "forbidden", "fixed"]
        surcharge_terms = ["freight", "fuel", "surcharge", "handling", "delivery fee", "shipping"]

        has_prohibition = any(p in clause_lower for p in prohibition_terms)
        has_surcharge_mention = any(s in clause_lower for s in surcharge_terms)

        if surcharge_amount > Decimal("0.00") and has_prohibition and has_surcharge_mention:
            return ContractClauseAnalysisSchema(
                is_violation=True,
                severity="reject",
                explanation=f"Contract prohibits separate freight/fuel charges. The invoice billed ${surcharge_amount} in unauthorized surcharges.",
            ).model_dump()

        return ContractClauseAnalysisSchema(
            is_violation=False,
            severity="info",
            explanation="No explicit contractual clause violation detected for this line item.",
        ).model_dump()

    def chat_reply(
        self,
        query: str,
        invoice_context: str,
        contract_context: str,
    ) -> str:
        """Conversational copilot answering auditor queries."""
        system_prompt = (
            "You are an expert Procurement Invoice & Contract Auditor Copilot. "
            "Answer user questions accurately based strictly on the provided invoice audit context, "
            "ERP purchase orders, goods receipts, and contract clauses. "
            "Be concise, professional, and reference specific line numbers or clause sections."
        )
        user_content = (
            f"Active Invoice & Audit Context:\n{invoice_context}\n\n"
            f"Contract Clauses Retrieved:\n{contract_context}\n\n"
            f"User Question:\n{query}"
        )

        # 1. Groq Path
        if self.groq_key:
            try:
                from groq import Groq
                client = Groq(api_key=self.groq_key)
                for model in self.groq_models:
                    try:
                        resp = client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_content},
                            ],
                            temperature=0.2,
                        )
                        reply = resp.choices[0].message.content
                        if reply:
                            return reply
                    except Exception:
                        continue
            except Exception as e:
                print(f"[LLM Warning] Groq chatbot completion error: {e}. Trying next provider.")

        # 2. Google Gemini Path
        if self.gemini_key:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                for model in self.gemini_models:
                    try:
                        llm = ChatGoogleGenerativeAI(
                            model=model,
                            google_api_key=self.gemini_key,
                            temperature=0.2,
                        )
                        resp = llm.invoke(f"{system_prompt}\n\n{user_content}")
                        if hasattr(resp, "content") and resp.content:
                            return str(resp.content)
                    except Exception:
                        continue
            except Exception as e:
                print(f"[LLM Warning] Gemini chat error: {e}. Trying next provider.")

        # 3. OpenAI Path
        if self.openai_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.openai_key)
                resp = client.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0.2,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                print(f"[LLM Warning] OpenAI chatbot completion error: {e}. Falling back to semantic reply.")

        # Conversational fallback
        q_lower = query.lower()
        if "why" in q_lower:
            return (
                f"**Audit Findings Summary:**\n{invoice_context}\n\n"
                f"The LangGraph multi-agent system cross-referenced the invoice PDF with the ERP Purchase Order, "
                f"Warehouse Goods Receipt, and Contract Clauses to identify variances."
            )
        if "clause" in q_lower or "contract" in q_lower:
            if contract_context:
                return f"**Relevant Contract Clauses:**\n{contract_context}"
            return "No matching contract clause found in the Vector DB for this query."
        return (
            f"Audit Copilot Status: Context loaded for active invoice. "
            f"You can ask why it was flagged, query contract clauses, or verify warehouse receipt quantities."
        )


# Global singleton LLM client
llm_client = LLMClient()
