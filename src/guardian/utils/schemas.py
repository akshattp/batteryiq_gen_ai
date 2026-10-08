"""Data schemas and type definitions for AI Battery Safety Guardian."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class BatteryMode(str, Enum):
    CHARGE = "charge"
    DISCHARGE = "discharge"
    IMPEDANCE = "impedance"


class TelemetryLabel(str, Enum):
    NORMAL = "normal"
    INJECTED_FAULT = "injected_fault"


class RiskLevel(str, Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class TelemetryStep:
    """Represents a single time-step telemetry reading."""
    battery_id: str
    cycle: int
    mode: str
    t_s: float
    timestamp: str
    voltage_v: float
    current_a: float
    temp_c: float
    ambient_c: float
    soc_pct: float
    capacity_ah: Optional[float] = None
    soh_pct: Optional[float] = None
    label: str = TelemetryLabel.NORMAL.value
    fault_type: Optional[str] = None


@dataclass
class RiskFactor:
    """Contributing factor to risk calculation."""
    name: str
    raw_value: float
    component_score: float  # [0.0, 1.0]
    weight: float
    weighted_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "raw_value": self.raw_value,
            "component_score": self.component_score,
            "weight": self.weight,
            "weighted_score": self.weighted_score,
        }


@dataclass
class RiskAssessment:
    """Result of deterministic risk assessment."""
    battery_id: str
    cycle: int
    t_s: float
    timestamp: str
    risk_score: float                  # [0.0, 100.0]
    risk_level: RiskLevel
    anomaly_score: float               # [0.0, 1.0]
    predicted_temp_c: float
    factors: List[RiskFactor] = field(default_factory=list)
    hard_overrides: List[str] = field(default_factory=list)
    raw_telemetry: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert assessment to dictionary."""
        return {
            "battery_id": self.battery_id,
            "cycle": self.cycle,
            "t_s": self.t_s,
            "timestamp": self.timestamp,
            "risk_score": round(self.risk_score, 2),
            "risk_level": self.risk_level.value,
            "anomaly_score": round(self.anomaly_score, 4),
            "predicted_temp_c": round(self.predicted_temp_c, 2),
            "factors": [
                {
                    "name": f.name,
                    "raw_value": round(f.raw_value, 4),
                    "component_score": round(f.component_score, 4),
                    "weight": f.weight,
                    "weighted_score": round(f.weighted_score, 4),
                }
                for f in self.factors
            ],
            "hard_overrides": self.hard_overrides,
            "raw_telemetry": {k: round(v, 4) for k, v in self.raw_telemetry.items()},
        }


@dataclass
class SafetyEvent:
    """Detected anomaly or risk level transition event."""
    event_id: str
    battery_id: str
    cycle: int
    t_s: float
    timestamp: str
    event_type: str
    severity: RiskLevel
    contributing_factors: List[str]
    telemetry_values: Dict[str, float]
    description: str
    risk_score: float = 0.0
    assessment: Optional[Any] = None

    @property
    def risk_level(self) -> RiskLevel:
        return self.severity

    @property
    def reasons(self) -> List[str]:
        return self.contributing_factors



@dataclass
class Recommendation:
    """Actionable safety advisory."""
    event_id: Optional[str]
    risk_level: RiskLevel
    actions: List[str]
    disclaimer: str = "Decision-support prototype. Not for vehicle control. Escalate to human engineer."


@dataclass
class ExplanationOutput:
    """Structured explanation returned by LLM or fallback."""
    summary: str
    reasons: List[str]
    recommended_actions: List[str]
    confidence_note: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert explanation output to dictionary."""
        return {
            "summary": self.summary,
            "reasons": list(self.reasons),
            "recommended_actions": list(self.recommended_actions),
            "confidence_note": self.confidence_note,
        }

    def model_dump(self) -> Dict[str, Any]:
        """Alias for to_dict for pydantic compatibility."""
        return self.to_dict()
