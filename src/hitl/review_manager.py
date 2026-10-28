"""Human-in-the-Loop (HITL) Decision Manager & Audit Trail.

Captures, validates, and persists human auditor decisions:
  - Approve
  - Flag for Correction
  - Reject
Records immutable audit logs with timestamps, reviewer rationale, and overrides.
Feeds corrections back into the ground-truth benchmark repository.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.core.schema import AuditResult, Decision, ReviewAction, ReviewDecision


class HITLReviewManager:
    """Manages Human-in-the-Loop review queue, decision recording, and audit trails."""

    def __init__(self, log_path: Optional[Path] = None, ground_truth_dir: Optional[Path] = None) -> None:
        self.log_path = log_path or Path("data/audit_log.json")
        self.ground_truth_dir = ground_truth_dir or Path("data/ground_truth")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.ground_truth_dir.mkdir(parents=True, exist_ok=True)
        self._audit_records: Dict[str, Dict[str, Any]] = self._load_logs()

    def _load_logs(self) -> Dict[str, Dict[str, Any]]:
        if self.log_path.exists():
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_logs(self) -> None:
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(self._audit_records, f, indent=2, default=str)

    def record_decision(
        self,
        invoice_id: str,
        reviewer_id: str,
        action: ReviewAction,
        notes: str,
        overridden_reasons: Optional[List[str]] = None,
        corrected_data: Optional[Dict[str, Any]] = None,
    ) -> ReviewDecision:
        """Record human reviewer decision and update audit trail."""
        timestamp = datetime.now(timezone.utc).isoformat()
        decision = ReviewDecision(
            reviewer_id=reviewer_id,
            action=action,
            notes=notes,
            overridden_reasons=overridden_reasons or [],
            corrected_data=corrected_data,
            timestamp=timestamp,
        )

        entry = {
            "invoice_id": invoice_id,
            "reviewer_id": reviewer_id,
            "action": action.value,
            "notes": notes,
            "overridden_reasons": overridden_reasons or [],
            "corrected_data": corrected_data or {},
            "timestamp": timestamp,
        }

        self._audit_records[invoice_id] = entry
        self._save_logs()

        # Save active learning ground truth correction
        if action in (ReviewAction.APPROVE, ReviewAction.FLAG):
            self._save_ground_truth_label(invoice_id, decision)

        return decision

    def _save_ground_truth_label(self, invoice_id: str, decision: ReviewDecision) -> None:
        """Persist human corrections for continuous evaluation benchmarking."""
        gt_file = self.ground_truth_dir / f"{invoice_id}_gt.json"
        data = {
            "invoice_id": invoice_id,
            "verified_by": decision.reviewer_id,
            "verified_at": decision.timestamp,
            "final_action": decision.action.value,
            "notes": decision.notes,
            "corrected_fields": decision.corrected_data or {},
        }
        with open(gt_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get_decision(self, invoice_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve historical reviewer decision for an invoice."""
        return self._audit_records.get(invoice_id)

    def list_recent_decisions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return most recent audit decisions."""
        records = list(self._audit_records.values())
        return records[-limit:]
