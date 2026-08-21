#!/usr/bin/env python3
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

"""

Usage:
    python -m evaluators.overall_quality \
        --result examples/001/final_report.md \
        --output examples/001/eval_overall_quality.json
"""

import json
import re
import logging
import argparse
from pathlib import Path
from typing import Optional, Any, Dict

from .utils.base import BaseEvaluator, EvalConfig, EvalResult
from .utils.llm_client import LLMClient

logger = logging.getLogger(__name__)

class OverallQualityEvaluator(BaseEvaluator):
    
    metric_name = "overall_quality"
    weight = 1.0
    
    def __init__(self, config: Optional[EvalConfig] = None):
        super().__init__(config)
        self.llm = LLMClient(self.config)
        self.max_retries = 3
    
    def _parse_evaluation_response(self, response: str, attempt: int = 1):
        """
        Parses the evaluation XML response into a structured dictionary.

        Args:
            response (str): The raw LLM output in XML-like format.
            attempt (int): Current attempt number for logging purposes.

        Returns:
            dict: A dictionary with score and justification for depth_quality.
        """

        def extract_block(tag):
            score_pattern = rf"<{tag}>\s*<score>\s*(.*?)\s*</score>\s*<justification>\s*(.*?)\s*</justification>\s*</{tag}>"
            match = re.search(score_pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                try:
                    score_str, justification = match.groups()
                    score = int(float(score_str.strip()))  # Convert to integer
                    if score > 10:
                        score = 10
                    elif score < 1:
                        score = 1
                    return {"score": score / 10.0, "justification": justification.strip()}  # Divide by 10 for final score
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Error parsing score for {tag} (attempt {attempt}): {e}")
                    return {"score": 0.0, "justification": f"Failed to parse response for {tag}"}
            
            # Try alternative parsing patterns
            # Look for just score tags
            score_only_pattern = rf"<{tag}>.*?<score>\s*(.*?)\s*</score>.*?</{tag}>"
            score_match = re.search(score_only_pattern, response, re.DOTALL | re.IGNORECASE)
            if score_match:
                try:
                    score = int(float(score_match.group(1).strip()))  # Convert to integer
                    if score > 10:
                        score = 10
                    elif score < 1:
                        score = 1
                    return {"score": score / 10.0, "justification": "No justification provided"}  # Divide by 10
                except (ValueError, AttributeError):
                    pass
            
            # Look for plain number patterns
            number_pattern = rf"{tag}.*?(\d+\.?\d*)"
            number_match = re.search(number_pattern, response, re.IGNORECASE)
            if number_match:
                try:
                    score = int(float(number_match.group(1)))  # Convert to integer
                    if score > 10:
                        score = 10
                    elif score < 1:
                        score = 1
                    return {"score": score / 10.0, "justification": "Extracted from unstructured response"}  # Divide by 10
                except (ValueError, AttributeError):
                    pass
                    
            return {"score": 0.0, "justification": f"No valid response found for {tag}"}

        # Only parse depth_quality
        evaluation = {"metric_result": {}}
        result = extract_block("depth_quality")
        evaluation["metric_result"]["depth_quality"] = result
        
        valid = result["score"] > 0 or "No valid response" not in result["justification"]
        
        # Score is just depth_quality score
        evaluation["score"] = result["score"]
        
        # Build summary
        evaluation["summary"] = f"**depth_quality-score:** {result['score']:.4f}\n\n"
        evaluation["summary"] += f"**depth_quality-justification:** {result['justification']}\n\n"

        if not valid:
            logger.warning(f"Failed to parse depth_quality (attempt {attempt})")
        
        return evaluation, 1 if valid else 0

    def evaluate(self, result_text: str, **kwargs) -> EvalResult:
        """Evaluate overall quality of a research report."""
        query_data = kwargs.get('query_data')
        
        if query_data is None:
            error_msg = "Missing query_data: cannot evaluate overall quality without the research question"
            logger.error(error_msg)
            print(f"❌ ERROR: {error_msg}")
            return EvalResult(
                metric_name=self.metric_name,
                score=-1,  # Error indicator
                details={
                    'error': error_msg,
                    'status': 'failed'
                },
                weight=self.weight
            )
        
        dr_question = query_data.get("query", "No question specified")

        prompt = f"""
        You are an expert Research Report Evaluator.

        You are given:
        1. A research report
        2. A research question that the report attempts to answer

        ------------------
        <research_question>
        {dr_question}
        </research_question>

        <report>
        {result_text}
        </report>
        ------------------

        ## Instructions:

        **ANALYZE THOROUGHLY**: Examine the report in detail and identify any issues, even small ones. Look for subtle problems, minor inconsistencies, areas that could be improved, or any shortcomings that might affect the quality.

        Evaluate the report according to the **Depth & Quality of Analysis** criterion. Provide:

        - A **score between 1 and 10** (must be an integer) using the scale defined below.
        - A **detailed justification** (2–3 sentences) in **simple plain English** explaining why you gave that score, including any specific issues or strengths you identified.

        ### General Scoring Scale (1-10, integers only):
        - **1** = Completely Unusable: No meaningful content, completely off-topic, or gibberish
        - **2** = Severely Deficient: Major fundamental flaws, almost no useful information
        - **3** = Very Poor: Significant problems throughout, barely addresses the topic
        - **4** = Poor: Multiple major issues, lacks coherence or depth, many gaps
        - **5** = Below Average: Notable weaknesses, incomplete coverage, superficial analysis
        - **6** = Adequate: Meets minimum requirements but with clear limitations and room for improvement
        - **7** = Good: Solid work that addresses the topic well, minor issues present
        - **8** = Very Good: High quality with comprehensive coverage, few minor issues
        - **9** = Excellent: Outstanding work with deep insights, nearly flawless execution
        - **10** = Perfect: Exceptional, publication-ready quality with innovative insights

        **CRITICAL EVALUATION REQUIREMENTS**:
        1. **Use the FULL scoring range**: Distribute scores across 1-10 based on actual quality differences. Do NOT cluster scores in a narrow range.
        2. **Differentiate clearly**: A mediocre report should score 4-5, a good report 6-7, an excellent report 8-9. Only truly exceptional work deserves 10.
        3. **Be discriminating**: Look for specific quality differences between reports. Better analysis, clearer structure, and deeper insights should result in higher scores.
        4. **Penalize appropriately**: Minor issues = small deductions (0.5-1 point), major issues = significant deductions (2-3 points).
        5. **Reward excellence**: If a report demonstrates exceptional depth, clarity, or insight, give it the high score it deserves.
        6. **Compare mentally**: Consider how this report compares to the best and worst possible reports on this topic.

        ### Evaluation Criterion: **Depth & Quality of Analysis**
        
        Evaluate how thoroughly the report analyzes the research question. **BE HARSH**: Look for superficiality, missing details, lack of evidence, weak reasoning.
        - **1-2**: Completely superficial, no real analysis, just lists facts
        - **3-4**: Very basic analysis, misses most key factors, no depth at all
        - **5**: Basic analysis present but very incomplete, misses many important aspects
        - **6**: Covers main points but analysis is shallow, lacks nuance and deeper insights
        - **7**: Adequate analysis with some depth, but still missing sophistication or critical thinking
        - **8**: Good depth with multiple factors explored, shows some sophisticated reasoning (RARE)
        - **9**: Exceptional depth with nuanced understanding and insights (VERY RARE)
        - **10**: Perfect analysis with groundbreaking insights and comprehensive understanding (ALMOST NEVER)

        ------------------

        ## Output format:

        <evaluation>
        <depth_quality>
            <score>1–10 (integer only, based on the scoring scale above)</score>
            <justification>Give a detailed 2–3 sentence justification for your score in simple plain English, including specific issues or strengths.</justification>
        </depth_quality>
        </evaluation>
        """

        max_retries = self.max_retries
        
        for attempt in range(max_retries):
            try:
                result = self.llm.call(
                    system="You are an expert evaluator focusing on deep research report quality.",
                    user=prompt,
                    json_mode=False,
                )
                
                scoring_result = result.get("text", "") if result else ""
                evaluation, valid_scores = self._parse_evaluation_response(scoring_result, attempt + 1)
                
                # Check if we got a valid score
                if valid_scores >= 1:
                    final_score = evaluation["score"] * 100
                    details = {
                        'sub_scores': {
                            'depth_quality': evaluation["metric_result"]["depth_quality"]["score"] * 100
                        },
                        'raw_evaluation': evaluation
                    }
                    
                    return EvalResult(
                        metric_name=self.metric_name,
                        score=final_score,
                        details=details,
                        weight=self.weight
                    )
                else:
                    logger.warning(f"Failed to parse depth_quality on attempt {attempt + 1}")
                    
            except Exception as e:
                logger.warning(f"Error in overall quality evaluation (attempt {attempt + 1}): {e}")
            
            # Modify prompt for retry attempts
            if attempt < max_retries - 1:
                prompt = f"""
                You are an expert Research Report Evaluator.

                You are given:
                1. A research report
                2. A research question that the report attempts to answer

                ------------------
                <research_question>
                {dr_question}
                </research_question>

                <report>
                {result_text}
                </report>
                ------------------

                IMPORTANT: You MUST respond in the EXACT format shown below. Do not add any extra text.

                **ANALYZE THOROUGHLY**: Find any issues, even small ones. Be critical - most reports should score 4-7. Only exceptional work deserves 9-10.

                Evaluate the report on **Depth & Quality of Analysis** (score from 1 to 10 as integer):
                - How thoroughly does the report analyze the research question?
                - Look for superficiality, missing details, lack of evidence, weak reasoning.

                **Scoring Scale**: 1-2=Unusable, 3-4=Poor, 5=Below Average, 6=Adequate, 7=Good, 8=Very Good, 9=Excellent, 10=Perfect
                **USE THE FULL RANGE**: Do NOT cluster all scores in 6-7. Differentiate quality clearly.

                Format your response EXACTLY as:

                <evaluation>
                <depth_quality>
                <score>6</score>
                <justification>Your detailed justification here with specific issues or strengths identified</justification>
                </depth_quality>
                </evaluation>
                """

        # If all retries failed, return error
        error_msg = f"Failed to get valid overall quality evaluation after {max_retries} attempts"
        logger.error(error_msg)
        print(f"❌ ERROR: {error_msg}")
        
        return EvalResult(
            metric_name=self.metric_name,
            score=-1,  # Error indicator
            details={
                'error': error_msg,
                'status': 'failed'
            },
            weight=self.weight
        )

def main():
    parser = argparse.ArgumentParser(description="Evaluate overall quality (depth_quality only)")
    parser.add_argument("--result", type=str, required=True, help="Path to result markdown file")
    parser.add_argument("--output", type=str, help="Output file for evaluation result")
    
    args = parser.parse_args()
    
    result_text = Path(args.result).read_text(encoding='utf-8')
    
    evaluator = OverallQualityEvaluator()
    result = evaluator.evaluate(result_text)
    
    print(f"\n📝 Overall Quality Score (Depth Quality): {result.score:.1f}/100")
    print(result.to_json())
    
    if args.output:
        evaluator.save_result(result, Path(args.output))
        print(f"\n📄 Result saved to: {args.output}")

if __name__ == "__main__":
    main()
