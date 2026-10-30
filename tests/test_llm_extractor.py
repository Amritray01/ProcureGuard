from decimal import Decimal
import pytest
from unittest.mock import MagicMock, patch

from src.core.llm import LLMClient
from src.core.schema import InvoiceExtractionSchema


def test_llm_client_initialization_and_provider_priority():
    with patch.dict("os.environ", {"GROQ_API_KEY": "gsk_dummy_key"}, clear=True):
        client = LLMClient()
        assert client.has_api_key() is True
        assert client.get_active_provider() == "groq"
        assert "openai/gpt-oss-120b" in client.groq_models

    with patch.dict("os.environ", {"GEMINI_API_KEY": "dummy_gemini_key"}, clear=True):
        client = LLMClient()
        assert client.has_api_key() is True
        assert client.get_active_provider() == "gemini"


def test_llm_client_groq_extraction_mock():
    client = LLMClient()
    client.groq_key = "gsk_test"

    sample_response_data = {
        "invoice_number": "INV-MOCK-99",
        "vendor": "Acme Widgets Corp",
        "po_number": "PO-99",
        "currency": "USD",
        "subtotal": "100.00",
        "tax": "10.00",
        "total": "110.00",
        "lines": [
            {
                "line_number": 1,
                "description": "Widget A",
                "sku": "W-A",
                "quantity": "2.0",
                "unit_price": "50.00",
                "line_total": "100.00",
            }
        ],
    }

    mock_chat_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"invoice_number": "INV-MOCK-99", "vendor": "Acme Widgets Corp", "po_number": "PO-99", "currency": "USD", "subtotal": "100.00", "tax": "10.00", "total": "110.00", "lines": [{"line_number": 1, "description": "Widget A", "sku": "W-A", "quantity": "2.0", "unit_price": "50.00", "line_total": "100.00"}]}'
    mock_chat_completion.choices = [mock_choice]

    mock_groq_instance = MagicMock()
    mock_groq_instance.chat.completions.create.return_value = mock_chat_completion

    with patch("groq.Groq", return_value=mock_groq_instance):
        result = client.extract_invoice_dynamic("Dummy Raw Invoice Text")
        assert result["invoice_number"] == "INV-MOCK-99"
        assert result["vendor"] == "Acme Widgets Corp"
        assert len(result["lines"]) == 1
        assert result["lines"][0]["description"] == "Widget A"
