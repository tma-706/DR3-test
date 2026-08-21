# Copyright 2025 Miromind.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base classes and configuration for all evaluators."""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class EvaluationAPIError(Exception):
    """Raised when an API call fails during evaluation."""
    def __init__(self, message: str, metric_name: str = "", details: Dict[str, Any] = None):
        super().__init__(message)
        self.metric_name = metric_name
        self.details = details or {}


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    api_key: str = field(default_factory=lambda: os.getenv(
        "OPENROUTER_API_KEY",
        os.getenv("ALIBABA_API_KEY", os.getenv("OPENAI_API_KEY", "")),
    ))
    base_url: str = field(default_factory=lambda: os.getenv(
        "OPENROUTER_BASE_URL",
        os.getenv("ALIBABA_BASE_URL", os.getenv("OPENAI_BASE_URL", "")),
    ))
    model_name: str = field(default_factory=lambda: os.getenv(
        "OPENROUTER_MODEL", os.getenv("ALIBABA_MODEL", "gpt-51-1113-global")
    ))
    max_segment_length: int = 8000
    temperature: float = field(default_factory=lambda: float(os.getenv("EVAL_TEMPERATURE", "0")))


@dataclass
class EvalResult:
    """Evaluation result."""
    metric_name: str
    score: float  # 0-100
    details: Dict[str, Any]
    weight: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'metric_name': self.metric_name,
            'score': self.score,
            'weight': self.weight,
            'details': self.details
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class BaseEvaluator(ABC):
    """Base class for all evaluators."""
    
    metric_name: str = "base"
    weight: float = 1.0
    
    def __init__(self, config: Optional[EvalConfig] = None):
        self.config = config or EvalConfig()
    
    @abstractmethod
    def evaluate(self, result_text: str, **kwargs) -> EvalResult:
        pass
    
    def load_json(self, path: Path) -> Any:
        return json.loads(path.read_text(encoding='utf-8'))
    
    def save_result(self, result: EvalResult, output_path: Path) -> None:
        output_path.write_text(result.to_json(), encoding='utf-8')
