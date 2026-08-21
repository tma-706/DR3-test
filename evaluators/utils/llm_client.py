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

"""LLM client wrapping OpenAI API calls."""

import json
import re
from typing import Dict, Optional
from openai import AuthenticationError, OpenAI, PermissionDeniedError
from .base import EvalConfig


def extract_json_from_text(text: str) -> Optional[Dict]:
    """Extract a JSON object from text, handling code blocks and truncation."""
    if not text:
        return None
    
    text = text.strip()
    
    # Normalize smart quotes
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Extract from markdown code blocks
    code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    matches = re.findall(code_block_pattern, text)
    
    if not matches and text.startswith('```'):
        content = re.sub(r'^```(?:json)?\s*', '', text)
        matches = [content]
    
    for match in matches:
        cleaned = match.strip()
        cleaned = cleaned.replace('\u201c', '"').replace('\u201d', '"')
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            fixed = _try_fix_truncated_json(cleaned)
            if fixed:
                try:
                    return json.loads(fixed)
                except json.JSONDecodeError:
                    pass
            continue
    
    # Try extracting between first { and last }
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
        json_str = text[first_brace:last_brace + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            fixed = _try_fix_truncated_json(json_str)
            if fixed:
                try:
                    return json.loads(fixed)
                except json.JSONDecodeError:
                    pass
    
    # Last resort: from first { with truncation repair
    if first_brace != -1:
        json_str = text[first_brace:]
        fixed = _try_fix_truncated_json(json_str)
        if fixed:
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass
    
    return None


def _try_fix_truncated_json(text: str) -> Optional[str]:
    """Attempt to fix truncated JSON by closing unclosed brackets/braces."""
    if not text:
        return None
    
    text = text.strip()
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass
    
    result = text
    
    if result.rstrip().endswith('...'):
        result = result.rstrip()[:-3].rstrip()
    
    if result.rstrip().endswith('"...'):
        result = result.rstrip()[:-3]
    
    # Truncate incomplete trailing key-value pairs
    last_comma = result.rfind(',')
    if last_comma > 0:
        after_comma = result[last_comma + 1:].strip()
        if after_comma and not after_comma.endswith(('}', ']', '"')):
            truncated = result[:last_comma]
            open_braces = truncated.count('{')
            close_braces = truncated.count('}')
            open_brackets = truncated.count('[')
            close_brackets = truncated.count(']')
            
            if truncated.count('"') % 2 == 1:
                truncated += '"'
            truncated += ']' * (open_brackets - close_brackets)
            truncated += '}' * (open_braces - close_braces)
            
            try:
                json.loads(truncated)
                return truncated
            except json.JSONDecodeError:
                pass
    
    # Close unclosed string
    if result.count('"') % 2 == 1:
        result += '"'
    
    open_braces = result.count('{')
    close_braces = result.count('}')
    open_brackets = result.count('[')
    close_brackets = result.count(']')
    
    missing_brackets = open_brackets - close_brackets
    if missing_brackets > 0:
        result += ']' * missing_brackets
    
    missing_braces = open_braces - close_braces
    if missing_braces > 0:
        result += '}' * missing_braces
    
    try:
        json.loads(result)
        return result
    except json.JSONDecodeError:
        return _try_extract_partial_json(text)


def _try_extract_partial_json(text: str) -> Optional[str]:
    """Extract valid partial content from truncated JSON with "results" or "claims" arrays."""
    if not text:
        return None
    
    results_match = re.search(r'"(?:results|claims)"\s*:\s*\[', text)
    if not results_match:
        return None
    
    start = results_match.end()
    
    depth = 1
    i = start
    last_complete_element = start
    in_string = False
    escape_next = False
    
    while i < len(text) and depth > 0:
        char = text[i]
        
        if escape_next:
            escape_next = False
            i += 1
            continue
        
        if char == '\\':
            escape_next = True
            i += 1
            continue
        
        if char == '"':
            in_string = not in_string
        elif not in_string:
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 1:
                    j = i + 1
                    while j < len(text) and text[j] in ' \t\n\r':
                        j += 1
                    if j < len(text) and text[j] in ',]':
                        last_complete_element = j + 1 if text[j] == ',' else j
            elif char == '[':
                depth += 1
            elif char == ']':
                depth -= 1
        
        i += 1
    
    if last_complete_element > start:
        partial = text[:last_complete_element]
        if not partial.rstrip().endswith(']'):
            partial = partial.rstrip()
            if partial.endswith(','):
                partial = partial[:-1]
            partial += ']'
        if not partial.rstrip().endswith('}'):
            partial += '}'
        
        try:
            json.loads(partial)
            return partial
        except json.JSONDecodeError:
            pass
    
    return None


class LLMClient:
    """LLM client for evaluation API calls."""
    
    DEFAULT_MAX_TOKENS = 16384
    
    MODEL_MAX_TOKENS_LIMITS = {
        "qwen-max": 8192,
        "qwen-max-latest": 8192,
        "qwen-plus": 8192,
        "qwen-turbo": 8192,
        "glm-4": 4096,
        "glm-4-plus": 4096,
    }
    
    def __init__(self, config: Optional[EvalConfig] = None):
        self.config = config or EvalConfig()
        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            max_retries=0,
            timeout=120.0,
        )
        self.call_count = 0
    
    def _get_max_tokens_for_model(self, model_name: str, requested_max_tokens: Optional[int] = None) -> int:
        """Get appropriate max_tokens for the given model."""
        model_limit = None
        for model_prefix, limit in self.MODEL_MAX_TOKENS_LIMITS.items():
            if model_name.startswith(model_prefix) or model_prefix in model_name.lower():
                model_limit = limit
                break
        
        if model_limit is None:
            return requested_max_tokens or self.DEFAULT_MAX_TOKENS
        
        if requested_max_tokens:
            return min(requested_max_tokens, model_limit)
        
        return min(self.DEFAULT_MAX_TOKENS, model_limit)
    
    def call(self, system: str, user: str, json_mode: bool = True, max_tokens: int = None) -> Optional[Dict]:
        """Call the LLM and return parsed JSON response."""
        self.call_count += 1
        
        actual_max_tokens = self._get_max_tokens_for_model(self.config.model_name, max_tokens)
        
        try:
            kwargs = {
                "model": self.config.model_name,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                "temperature": self.config.temperature,
                "max_tokens": actual_max_tokens
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            
            response = self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            
            if not json_mode:
                return {"text": content}
            
            result = extract_json_from_text(content)
            if result is None:
                print(f"  Warning: Failed to parse JSON response: {content[:200]}...")
            return result
            
        except (AuthenticationError, PermissionDeniedError):
            raise
        except Exception as e:
            print(f"  Warning: LLM call error: {e}")
            return None
    
    def call_with_retry(self, system: str, user: str, 
                        json_mode: bool = True, max_retries: int = 3,
                        max_tokens: int = None) -> Optional[Dict]:
        """Call LLM with retry logic."""
        for attempt in range(max_retries):
            result = self.call(system, user, json_mode, max_tokens=max_tokens)
            if result is not None:
                return result
            print(f"  Retry {attempt + 1}/{max_retries}...")
        return None
