import json
import numpy as np
import re
from collections import Counter, defaultdict
import math
from tqdm import tqdm
import argparse
import os
import requests
import time

# Global variables for API
API_KEY = os.getenv("OPENAI_API_KEY", "")
API_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
API_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

def call_llm_api(prompt):
    """
    Call LLM API for semantic scoring
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": API_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.01,
        "max_tokens": 512
    }
    
    retries = 3
    for i in range(retries):
        try:
            response = requests.post(f"{API_BASE_URL}/chat/completions", headers=headers, json=data, timeout=30)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            print(f"API call failed (attempt {i+1}/{retries}): {e}")
            time.sleep(2)
    return ""

class MedicalDiagnosisEvaluator:
    def __init__(self, simulation_data_item, alpha=1.0, beta=0.01):
        """
        Initialize evaluator with simulation data and reference data for information retrieval evaluation.

        Parameters:
            simulation_data_item (dict): A single simulation dialogue data item
            alpha (float): Weight for interaction turns in DEI calculation
            beta (float): Weight for average token count in DEI calculation
        """
        self.simulation_data = simulation_data_item
        self.alpha = alpha
        self.beta = beta

    def tokenize_text(self, text):
        """
        Tokenize text, split by characters for Chinese, by spaces for English

        Parameters:
            text (str): Input text

        Returns:
            list: List after tokenization
        """
        if not text:
            return []

        # For Chinese text, tokenize by character; for English text, tokenize by space
        if any('\u4e00' <= c <= '\u9fff' for c in text):
            # Chinese text, tokenize by character
            tokens = list(text)
        else:
            # English text, tokenize by space
            tokens = text.split()

        return tokens

    def calculate_bleu(self, candidate, reference, n_gram=1):
        """
        Calculate BLEU-N score

        Parameters:
            candidate (str): Candidate text
            reference (str): Reference text
            n_gram (int): n-gram size (1, 2, 3, 4)

        Returns:
            float: BLEU-N score
        """
        if not candidate or not reference:
            if not candidate and not reference:  # Both are empty
                return 1.0
            return 0.0

        # Tokenize
        candidate_tokens = self.tokenize_text(candidate)
        reference_tokens = self.tokenize_text(reference)

        if len(candidate_tokens) == 0:
            return 0.0

        # Calculate N-gram precision
        candidate_ngrams = self._get_ngrams(candidate_tokens, n_gram)
        reference_ngrams = self._get_ngrams(reference_tokens, n_gram)

        # Calculate common n-gram count
        common_count = sum((candidate_ngrams & reference_ngrams).values())

        # Calculate precision
        if sum(candidate_ngrams.values()) == 0:
            precision = 0.0
        else:
            precision = common_count / sum(candidate_ngrams.values())

        # Optional: Calculate simple brevity penalty
        bp = 1.0
        if len(candidate_tokens) < len(reference_tokens):
            bp = math.exp(1 - len(reference_tokens) / len(candidate_tokens))

        return bp * precision

    def calculate_rouge(self, candidate, reference, n_gram=1, use_lcs=False):
        """
        Calculate ROUGE-N/L score

        Parameters:
            candidate (str): Candidate text
            reference (str): Reference text
            n_gram (int): n-gram size
            use_lcs (bool): Whether to use LCS algorithm (ROUGE-L)

        Returns:
            tuple: (F1, Precision, Recall)
        """
        if not candidate or not reference:
            if not candidate and not reference:  # Both are empty
                return 1.0, 1.0, 1.0
            return 0.0, 0.0, 0.0

        # Tokenize
        candidate_tokens = self.tokenize_text(candidate)
        reference_tokens = self.tokenize_text(reference)

        if use_lcs:  # ROUGE-L
            lcs_len = self._longest_common_subsequence_length(candidate_tokens, reference_tokens)

            # Calculate precision and recall
            precision = lcs_len / len(candidate_tokens) if candidate_tokens else 0.0
            recall = lcs_len / len(reference_tokens) if reference_tokens else 0.0

        else:  # ROUGE-N
            # Get n-gram counts
            candidate_ngrams = self._get_ngrams(candidate_tokens, n_gram)
            reference_ngrams = self._get_ngrams(reference_tokens, n_gram)

            # Calculate overlapping n-gram count
            overlap_count = sum((candidate_ngrams & reference_ngrams).values())

            # Calculate precision and recall
            precision = overlap_count / sum(candidate_ngrams.values()) if sum(candidate_ngrams.values()) > 0 else 0.0
            recall = overlap_count / sum(reference_ngrams.values()) if sum(reference_ngrams.values()) > 0 else 0.0

        # Calculate F1 score
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        return f1, precision, recall

    def _get_ngrams(self, tokens, n):
        """
        Get n-gram counts from text

        Parameters:
            tokens (list): Tokenized text
            n (int): n-gram size

        Returns:
            Counter: n-gram counts
        """
        ngrams = Counter()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams[ngram] += 1
        return ngrams

    def _longest_common_subsequence_length(self, tokens1, tokens2):
        """
        Calculate the length of the longest common subsequence between two token sequences

        Parameters:
            tokens1 (list): First token sequence
            tokens2 (list): Second token sequence

        Returns:
            int: Length of the longest common subsequence
        """
        if not tokens1 or not tokens2:
            return 0

        m, n = len(tokens1), len(tokens2)
        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if tokens1[i-1] == tokens2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[m][n]

    def calculate_metrics(self, diagnosis_semantic_score, recommendation_semantic_score):
        """
        Calculate various evaluation metrics for the model's diagnosis and recommendations
        compared to the ground truth, including BLEU-1/2/3/4 and ROUGE-1/2/L.
        Includes pre-calculated semantic scores.

        Parameters:
            diagnosis_semantic_score (float): Pre-calculated semantic score for diagnosis
            recommendation_semantic_score (float): Pre-calculated semantic score for recommendation
        """
        # Get the final dialogue turn containing the diagnosis
        diagnosis_turns = [turn for turn in self.simulation_data["simulation_dialogue"]
                         if turn.get("role") == "doctor"] # Check last doctor turn

        if not diagnosis_turns:
            # Return empty evaluation results
            empty_metrics = {
                "combined_score": 0.0,
                "diagnosis": {
                    "bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0,
                    "rouge_1": {"f1": 0.0, "precision": 0.0, "recall": 0.0},
                    "rouge_2": {"f1": 0.0, "precision": 0.0, "recall": 0.0},
                    "rouge_l": {"f1": 0.0, "precision": 0.0, "recall": 0.0},
                    "semantic_score": 0.0
                },
                "recommendation": {
                    "bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0,
                    "rouge_1": {"f1": 0.0, "precision": 0.0, "recall": 0.0},
                    "rouge_2": {"f1": 0.0, "precision": 0.0, "recall": 0.0},
                    "rouge_l": {"f1": 0.0, "precision": 0.0, "recall": 0.0},
                    "semantic_score": 0.0
                }
            }
            return empty_metrics

        final_diagnosis_turn = diagnosis_turns[-1]
        model_diagnosis = final_diagnosis_turn["content"]
        



        # Get ground truth
        # Reference data structure: {"诊断结果": "...", "诊断依据": "...", "治疗方案": "..."}
        true_diagnosis_text = self.simulation_data.get("true_diagnosis", "")
        true_evidence_text = self.simulation_data.get("true_evidence", "")
        # true_recommendation = "" # Ignore treatment as requested
        
        true_diagnosis = f"{true_diagnosis_text} {true_evidence_text}".strip()

        # Calculate evaluation metrics for diagnosis (excluding semantic score here)
        diagnosis_metrics = {
            "bleu_1": self.calculate_bleu(model_diagnosis, true_diagnosis, n_gram=1),
            "bleu_2": self.calculate_bleu(model_diagnosis, true_diagnosis, n_gram=2),
            "bleu_3": self.calculate_bleu(model_diagnosis, true_diagnosis, n_gram=3),
            "bleu_4": self.calculate_bleu(model_diagnosis, true_diagnosis, n_gram=4),
            "rouge_1": {},
            "rouge_2": {},
            "rouge_l": {},
            "semantic_score": diagnosis_semantic_score # Use pre-calculated score
        }

        # Calculate ROUGE metrics
        f1, p, r = self.calculate_rouge(model_diagnosis, true_diagnosis, n_gram=1)
        diagnosis_metrics["rouge_1"] = {"f1": f1, "precision": p, "recall": r}

        f1, p, r = self.calculate_rouge(model_diagnosis, true_diagnosis, n_gram=2)
        diagnosis_metrics["rouge_2"] = {"f1": f1, "precision": p, "recall": r}

        f1, p, r = self.calculate_rouge(model_diagnosis, true_diagnosis, use_lcs=True)
        diagnosis_metrics["rouge_l"] = {"f1": f1, "precision": p, "recall": r}

        # Calculate evaluation metrics for recommendation (excluding semantic score here)
        # Since we ignore treatment comparison, we can set these to 0 or just calculate against empty string
        # But to keep structure consistent, we'll just leave them as 0s or calculated against empty if true_rec is empty
        recommendation_metrics = {
            "bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0,
            "rouge_1": {"f1": 0.0, "precision": 0.0, "recall": 0.0},
            "rouge_2": {"f1": 0.0, "precision": 0.0, "recall": 0.0},
            "rouge_l": {"f1": 0.0, "precision": 0.0, "recall": 0.0},
            "semantic_score": 0.0 # Ignore recommendation semantic score
        }

        # Use ROUGE-L F1 and semantic scores for the combined score (Only Diagnosis)
        rouge_combined = diagnosis_metrics["rouge_l"]["f1"]
        semantic_combined = diagnosis_semantic_score / 5.0  # Normalize to 0-1 (assuming score is 0-5)

        # Combine both metrics (equal weight)
        combined_score = (rouge_combined + semantic_combined) / 2

        return {
            "combined_score": combined_score,
            "diagnosis": diagnosis_metrics,
            "recommendation": recommendation_metrics
        }

    def calculate_interaction_efficiency(self):
        """
        Calculate interaction efficiency based on total turns and token count.
        Efficiency = weighted dialogue turns × average tokens per turn
        """
        total_turns = self.simulation_data["total_turns"]

        # Calculate average tokens per turn
        doctor_turns = [turn for turn in self.simulation_data["simulation_dialogue"]
                       if turn.get("role") == "doctor"]

        if not doctor_turns:
            return 0, 0, 0  # No doctor turns found

        total_tokens = sum(turn.get("tokens", 0) for turn in doctor_turns)
        avg_tokens = total_tokens / len(doctor_turns)

        # Calculate interaction efficiency (lower is better)
        interaction_efficiency = total_turns * avg_tokens

        return interaction_efficiency, avg_tokens, total_turns

    def calculate_information_retrieval_rate(self):
        """
        计算信息检索率（返回原始准确率和大模型评分）
        返回格式：(precision, model_score)
        """
        # 定义错误信息模式
        error_patterns = [
            r"Sorry, you've asked this question before",
            r"Sorry, I cannot answer your question"
        ]
        
        # 获取所有有效患者回答（过滤错误信息）
        valid_responses = []
        for turn in self.simulation_data.get("simulation_dialogue", []):
            if turn.get("role") == "patient" and "content" in turn:
                response = turn["content"].strip()
                # 过滤错误回答
                if not any(re.search(pattern, response) for pattern in error_patterns):
                    valid_responses.append(response)

        # 计算原始准确率
        total_responses = len([t for t in self.simulation_data.get("simulation_dialogue", []) 
                            if t.get("role") == "patient"])
        valid_count = len(valid_responses)
        precision = valid_count / total_responses if total_responses > 0 else 0.0

        return precision

    def calculate_diagnostic_efficiency_index(self, combined_score, total_turns, avg_tokens):
        """
        Calculate Diagnostic Efficiency Index
        DEI = (score)² / (α × interaction turns + β × average token count)
        Uses combined score as the scoring metric
        Higher DEI indicates better efficiency while maintaining accuracy.

        Parameters:
            combined_score (float): The combined diagnostic performance score
            total_turns (int): Total interaction turns
            avg_tokens (float): Average tokens per turn
        """
        # Calculate denominator using formula
        denominator = self.alpha * total_turns + self.beta * avg_tokens

        # Avoid division by zero
        if denominator == 0:
            return 0.0

        # Calculate DEI
        dei = (combined_score ** 2) / denominator

        return dei

    def extract_model_outputs(self):
        """
        Extract model diagnosis and recommendation text from the simulation data.
        """
        diagnosis_turns = [turn for turn in self.simulation_data["simulation_dialogue"]
                         if turn.get("role") == "doctor" and turn.get("is_diagnosis", False)]

        if not diagnosis_turns:
            return "", ""

        final_diagnosis_turn = diagnosis_turns[-1]
        model_output = final_diagnosis_turn["content"]

        # Extract diagnosis and recommendation text
        model_diagnosis= model_output
        model_recommendation = ""  # Ignore treatment recommendation as per instructions

        # 定义错误信息模式
        error_patterns = [
            r"不知道",
            r"不清楚"
        ]
        
        # 获取所有有效患者回答（过滤错误信息）
        valid_responses = []
        for turn in self.simulation_data.get("simulation_dialogue", []):
            if turn.get("role") == "patient" and "content" in turn:
                response = turn["content"].strip()
                # 过滤错误回答
                if not any(re.search(pattern, response) for pattern in error_patterns):
                    valid_responses.append(response)

        # 构建sentence1（有效回答的拼接）
        model_gathered_info = " ".join(valid_responses)

        return model_diagnosis, model_recommendation, model_gathered_info


from concurrent.futures import ThreadPoolExecutor, as_completed

def calculate_semantic_similarity_score_batch(data_pairs, info_gather=False):
    """
    Calculate semantic similarity scores for a batch of (candidate, reference) pairs
    using LLM API.

    Parameters:
        data_pairs (list of tuples): List of (candidate_text, reference_text) tuples

    Returns:
        list of float: List of semantic similarity scores (0-5)
    """
    if not data_pairs:
        return []

    prompts = []
    for candidate, reference in data_pairs:
        if not candidate or not reference:
             # Handle empty cases directly
             prompts.append(None) # Use None as a placeholder for empty cases
             continue
        
        if info_gather:
            with open('ragen/env/medical_consultation/evaluation/TCM_eval_information_prompt_template.txt', 'r') as file:
                prompt = file.read()
                prompt = prompt.format(patient_self_report=reference, doctor_gathered_info=candidate)
        else:
            with open('ragen/env/medical_consultation/evaluation/TCM_eval_prompt_template_v2.txt', 'r') as file:
                prompt = file.read()
                prompt = prompt.format(candidate=candidate, reference=reference)
        prompts.append(prompt)

    scores = [0.0] * len(prompts)
    
    def process_single_prompt(idx, prompt):
        if prompt is None:
            return idx, 0.0
            
        response_text = call_llm_api(prompt)
        
        # Parse score
        score_pattern = r'(\d{1,3})(?:\s*\/\s*5)?$'
        match = re.search(r"<answer>(.*?)</answer>", response_text, re.DOTALL)
        if match:
            matched_response = match.group(0)
        else:
            matched_response = response_text.strip()
            
        match = re.search(score_pattern, matched_response, re.DOTALL)
        if match:
            score = float(match.group(1))
            return idx, min(max(score, 0), 5)
        else:
            # If no clear score is found, try to find any number in the text
            numbers = re.findall(r'\b\d{1,3}\b', response_text.strip())
            if numbers:
                for num in numbers:
                    num_val = float(num)
                    if 0 <= num_val <= 5:
                        return idx, num_val
            print(f"Warning: Could not extract score from: {response_text.strip()}")
            return idx, 0.0

    # Use ThreadPoolExecutor for parallel API calls
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for i, prompt in enumerate(prompts):
            futures.append(executor.submit(process_single_prompt, i, prompt))
            
        for future in as_completed(futures):
            idx, score = future.result()
            scores[idx] = score

    return scores


def calculate_category_averages(results_by_category):
    """
    Calculate average metrics for each category.
    
    Parameters:
        results_by_category (dict): Dictionary with category names as keys and lists of results as values
        
    Returns:
        dict: Average results by category
    """
    category_averages = {}
    
    for category, results in results_by_category.items():
        if not results:
            continue
            
        # Calculate averages for this category
        avg_result = {
            "diagnostic_performance": {
                "combined_score": np.mean([r["diagnostic_performance"]["combined_score"] for r in results]),
                "diagnosis": {
                    "bleu_1": np.mean([r["diagnostic_performance"]["diagnosis"]["bleu_1"] for r in results]),
                    "bleu_2": np.mean([r["diagnostic_performance"]["diagnosis"]["bleu_2"] for r in results]),
                    "bleu_3": np.mean([r["diagnostic_performance"]["diagnosis"]["bleu_3"] for r in results]),
                    "bleu_4": np.mean([r["diagnostic_performance"]["diagnosis"]["bleu_4"] for r in results]),
                    "rouge_1": {"f1": np.mean([r["diagnostic_performance"]["diagnosis"]["rouge_1"]["f1"] for r in results]),
                                "precision": np.mean([r["diagnostic_performance"]["diagnosis"]["rouge_1"]["precision"] for r in results]),
                                "recall": np.mean([r["diagnostic_performance"]["diagnosis"]["rouge_1"]["recall"] for r in results])},
                    "rouge_2": {"f1": np.mean([r["diagnostic_performance"]["diagnosis"]["rouge_2"]["f1"] for r in results]),
                                "precision": np.mean([r["diagnostic_performance"]["diagnosis"]["rouge_2"]["precision"] for r in results]),
                                "recall": np.mean([r["diagnostic_performance"]["diagnosis"]["rouge_2"]["recall"] for r in results])},
                    "rouge_l": {"f1": np.mean([r["diagnostic_performance"]["diagnosis"]["rouge_l"]["f1"] for r in results]),
                                "precision": np.mean([r["diagnostic_performance"]["diagnosis"]["rouge_l"]["precision"] for r in results]),
                                "recall": np.mean([r["diagnostic_performance"]["diagnosis"]["rouge_l"]["recall"] for r in results])},
                    "semantic_score": np.mean([r["diagnostic_performance"]["diagnosis"]["semantic_score"] for r in results])
                },
                "recommendation": {
                    "bleu_1": np.mean([r["diagnostic_performance"]["recommendation"]["bleu_1"] for r in results]),
                    "bleu_2": np.mean([r["diagnostic_performance"]["recommendation"]["bleu_2"] for r in results]),
                    "bleu_3": np.mean([r["diagnostic_performance"]["recommendation"]["bleu_3"] for r in results]),
                    "bleu_4": np.mean([r["diagnostic_performance"]["recommendation"]["bleu_4"] for r in results]),
                    "rouge_1": {"f1": np.mean([r["diagnostic_performance"]["recommendation"]["rouge_1"]["f1"] for r in results]),
                                "precision": np.mean([r["diagnostic_performance"]["recommendation"]["rouge_1"]["precision"] for r in results]),
                                "recall": np.mean([r["diagnostic_performance"]["recommendation"]["rouge_1"]["recall"] for r in results])},
                    "rouge_2": {"f1": np.mean([r["diagnostic_performance"]["recommendation"]["rouge_2"]["f1"] for r in results]),
                                "precision": np.mean([r["diagnostic_performance"]["recommendation"]["rouge_2"]["precision"] for r in results]),
                                "recall": np.mean([r["diagnostic_performance"]["recommendation"]["rouge_2"]["recall"] for r in results])},
                    "rouge_l": {"f1": np.mean([r["diagnostic_performance"]["recommendation"]["rouge_l"]["f1"] for r in results]),
                                "precision": np.mean([r["diagnostic_performance"]["recommendation"]["rouge_l"]["precision"] for r in results]),
                                "recall": np.mean([r["diagnostic_performance"]["recommendation"]["rouge_l"]["recall"] for r in results])},
                    "semantic_score": np.mean([r["diagnostic_performance"]["recommendation"]["semantic_score"] for r in results])
                }
            },
            "interaction_efficiency": {
                "total_turns": np.mean([r["interaction_efficiency"]["total_turns"] for r in results]),
                "avg_tokens": np.mean([r["interaction_efficiency"]["avg_tokens"] for r in results]),
                "interaction_efficiency": np.mean([r["interaction_efficiency"]["interaction_efficiency"] for r in results])
            },
            "information_retrieval": {
                "precision": np.mean([r["information_retrieval"]["precision"] for r in results]),
                "model_score": np.mean([r["information_retrieval"]["model_score"] for r in results])
            },
            "diagnostic_efficiency_index": np.mean([r["diagnostic_efficiency_index"] for r in results]),
            "case_count": len(results)
        }
        
        category_averages[category] = avg_result
    
    return category_averages


def evaluate_all_cases(simulation_data_list, alpha=1.0, beta=0.01, batch_size=16):
    """
    Evaluate all simulation cases using batch processing for semantic similarity.

    Parameters:
        simulation_data_list (list): List of simulation dialogue data items.
        alpha (float): Weight for interaction turns in DEI calculation.
        beta (float): Weight for average token count in DEI calculation.
        batch_size (int): Number of cases to process in each batch for semantic scoring.

    Returns:
        dict: Evaluation results including case-level results, overall average results, and category-based averages.
    """
    if not simulation_data_list:
        print("No simulation data provided")
        return {}

    case_results = []
    all_diagnosis_pairs = []
    all_recommendation_pairs = []
    all_info_gather_pairs = []
    results_by_category = defaultdict(list)  # Store results grouped by category

    # --- Step 1: Extract model outputs and prepare data for batching ---
    print("Extracting model outputs and preparing for batching...")
    
    for i, sim_item in enumerate(tqdm(simulation_data_list, desc="Preparing Data")):
        evaluator = MedicalDiagnosisEvaluator(
            simulation_data_item=sim_item,
            alpha=alpha,
            beta=beta
        )
        model_diagnosis, model_recommendation, model_gathered_info = evaluator.extract_model_outputs()
        
        # Get ground truth
        true_diagnosis_text = sim_item.get("true_diagnosis", "")
        true_evidence_text = sim_item.get("true_evidence", "")
        true_recommendation = "" # Ignore
        true_info = sim_item.get("enhanced_description", sim_item.get("patient_context", ""))
        
        true_diagnosis = f"{true_diagnosis_text} {true_evidence_text}".strip()

        all_diagnosis_pairs.append((model_diagnosis, true_diagnosis))
        all_recommendation_pairs.append((model_recommendation, true_recommendation))
        all_info_gather_pairs.append((model_gathered_info, true_info))

    # --- Step 2: Calculate semantic scores in batches ---
    print(f"Calculating semantic scores in batches (batch size: {batch_size})...")
    all_diagnosis_semantic_scores = []
    all_recommendation_semantic_scores = []
    all_info_gather_semantic_scores = []

    # Process diagnosis pairs in batches
    for i in tqdm(range(0, len(all_diagnosis_pairs), batch_size), desc="Semantic Scoring (Diagnosis)"):
        batch_pairs = all_diagnosis_pairs[i:i + batch_size]
        batch_scores = calculate_semantic_similarity_score_batch(batch_pairs)
        all_diagnosis_semantic_scores.extend(batch_scores)

    # Process recommendation pairs in batches
    for i in tqdm(range(0, len(all_recommendation_pairs), batch_size), desc="Semantic Scoring (Recommendation)"):
        batch_pairs = all_recommendation_pairs[i:i + batch_size]
        batch_scores = calculate_semantic_similarity_score_batch(batch_pairs)
        all_recommendation_semantic_scores.extend(batch_scores)

    # Process info gather pairs in batches
    for i in tqdm(range(0, len(all_info_gather_pairs), batch_size), desc="Semantic Scoring (Info Gather)"):
        batch_pairs = all_info_gather_pairs[i:i + batch_size]
        batch_scores = calculate_semantic_similarity_score_batch(batch_pairs, info_gather=True)
        all_info_gather_semantic_scores.extend(batch_scores)

    # --- Step 3: Calculate other metrics and assemble results ---
    print("Calculating other metrics and assembling results...")
    
    for i in tqdm(range(len(simulation_data_list)), desc="Calculating Other Metrics"):
        sim_item = simulation_data_list[i]
        evaluator = MedicalDiagnosisEvaluator(
            simulation_data_item=sim_item,
            alpha=alpha,
            beta=beta
        )

        # Get pre-calculated semantic scores for this case
        diagnosis_semantic_score = all_diagnosis_semantic_scores[i]
        recommendation_semantic_score = all_recommendation_semantic_scores[i]
        info_gather_semantic_score = all_info_gather_semantic_scores[i]

        # Calculate metrics, passing the semantic scores
        metrics = evaluator.calculate_metrics(diagnosis_semantic_score, recommendation_semantic_score)
        interaction_efficiency, avg_tokens, total_turns = evaluator.calculate_interaction_efficiency()
        precision = evaluator.calculate_information_retrieval_rate()

        # Calculate DEI using the combined score and efficiency metrics
        dei = evaluator.calculate_diagnostic_efficiency_index(
            combined_score=metrics["combined_score"],
            total_turns=total_turns,
            avg_tokens=avg_tokens
        )

        # Add ID and category for identification
        result = {
            "id": sim_item.get("id"),
            "category": sim_item.get("category", "unknown"),  # Get category from simulation data
            "diagnostic_performance": {
                "combined_score": metrics["combined_score"],
                "diagnosis": metrics["diagnosis"],
                "recommendation": metrics["recommendation"]
            },
            "interaction_efficiency": {
                "total_turns": total_turns,
                "avg_tokens": avg_tokens,
                "interaction_efficiency": interaction_efficiency
            },
            "information_retrieval": {
                "precision": precision,
                "model_score": info_gather_semantic_score
            },
            "diagnostic_efficiency_index": dei
        }
        
        case_results.append(result)
        
        # Group results by category
        category = sim_item.get("category", "unknown")
        results_by_category[category].append(result)

    # Calculate overall average results
    if not case_results:
        return {"case_results": [], "average_result": {}, "category_results": {}}

    # Calculate overall averages
    avg_result = {
        "diagnostic_performance": {
            "combined_score": np.mean([r["diagnostic_performance"]["combined_score"] for r in case_results]),
            "diagnosis": {
                "bleu_1": np.mean([r["diagnostic_performance"]["diagnosis"]["bleu_1"] for r in case_results]),
                "bleu_2": np.mean([r["diagnostic_performance"]["diagnosis"]["bleu_2"] for r in case_results]),
                "bleu_3": np.mean([r["diagnostic_performance"]["diagnosis"]["bleu_3"] for r in case_results]),
                "bleu_4": np.mean([r["diagnostic_performance"]["diagnosis"]["bleu_4"] for r in case_results]),
                "rouge_1": {"f1": np.mean([r["diagnostic_performance"]["diagnosis"]["rouge_1"]["f1"] for r in case_results]),
                            "precision": np.mean([r["diagnostic_performance"]["diagnosis"]["rouge_1"]["precision"] for r in case_results]),
                            "recall": np.mean([r["diagnostic_performance"]["diagnosis"]["rouge_1"]["recall"] for r in case_results])},
                "rouge_2": {"f1": np.mean([r["diagnostic_performance"]["diagnosis"]["rouge_2"]["f1"] for r in case_results]),
                            "precision": np.mean([r["diagnostic_performance"]["diagnosis"]["rouge_2"]["precision"] for r in case_results]),
                            "recall": np.mean([r["diagnostic_performance"]["diagnosis"]["rouge_2"]["recall"] for r in case_results])},
                "rouge_l": {"f1": np.mean([r["diagnostic_performance"]["diagnosis"]["rouge_l"]["f1"] for r in case_results]),
                            "precision": np.mean([r["diagnostic_performance"]["diagnosis"]["rouge_l"]["precision"] for r in case_results]),
                            "recall": np.mean([r["diagnostic_performance"]["diagnosis"]["rouge_l"]["recall"] for r in case_results])},
                "semantic_score": np.mean([r["diagnostic_performance"]["diagnosis"]["semantic_score"] for r in case_results])
            },
            "recommendation": {
                "bleu_1": np.mean([r["diagnostic_performance"]["recommendation"]["bleu_1"] for r in case_results]),
                "bleu_2": np.mean([r["diagnostic_performance"]["recommendation"]["bleu_2"] for r in case_results]),
                "bleu_3": np.mean([r["diagnostic_performance"]["recommendation"]["bleu_3"] for r in case_results]),
                "bleu_4": np.mean([r["diagnostic_performance"]["recommendation"]["bleu_4"] for r in case_results]),
                "rouge_1": {"f1": np.mean([r["diagnostic_performance"]["recommendation"]["rouge_1"]["f1"] for r in case_results]),
                            "precision": np.mean([r["diagnostic_performance"]["recommendation"]["rouge_1"]["precision"] for r in case_results]),
                            "recall": np.mean([r["diagnostic_performance"]["recommendation"]["rouge_1"]["recall"] for r in case_results])},
                "rouge_2": {"f1": np.mean([r["diagnostic_performance"]["recommendation"]["rouge_2"]["f1"] for r in case_results]),
                            "precision": np.mean([r["diagnostic_performance"]["recommendation"]["rouge_2"]["precision"] for r in case_results]),
                            "recall": np.mean([r["diagnostic_performance"]["recommendation"]["rouge_2"]["recall"] for r in case_results])},
                "rouge_l": {"f1": np.mean([r["diagnostic_performance"]["recommendation"]["rouge_l"]["f1"] for r in case_results]),
                            "precision": np.mean([r["diagnostic_performance"]["recommendation"]["rouge_l"]["precision"] for r in case_results]),
                            "recall": np.mean([r["diagnostic_performance"]["recommendation"]["rouge_l"]["recall"] for r in case_results])},
                "semantic_score": np.mean([r["diagnostic_performance"]["recommendation"]["semantic_score"] for r in case_results])
            }
        },
        "interaction_efficiency": {
            "total_turns": np.mean([r["interaction_efficiency"]["total_turns"] for r in case_results]),
            "avg_tokens": np.mean([r["interaction_efficiency"]["avg_tokens"] for r in case_results]),
            "interaction_efficiency": np.mean([r["interaction_efficiency"]["interaction_efficiency"] for r in case_results])
        },
        "information_retrieval": {
            "precision": np.mean([r["information_retrieval"]["precision"] for r in case_results]),
            "model_score": np.mean([r["information_retrieval"]["model_score"] for r in case_results])
        },
        "diagnostic_efficiency_index": np.mean([r["diagnostic_efficiency_index"] for r in case_results])
    }

    # Calculate category-based averages
    category_averages = calculate_category_averages(results_by_category)

    return {
        "case_results": case_results,
        "average_result": avg_result,
        "category_results": category_averages
    }


def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Medical Dialogue Diagnosis Evaluation')
    parser.add_argument('--input_file', type=str,
                        default='outputs/debug_logs/debug_output.json',
                        help='Path to input JSON file containing simulation and ground truth')
    parser.add_argument('--output', type=str,
                        default='evaluation_results.json',
                        help='Path to output results JSON file')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Weight for interaction turns in DEI calculation')
    parser.add_argument('--beta', type=float, default=0.01,
                        help='Weight for average token count in DEI calculation')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for semantic similarity calculation')

    args = parser.parse_args()

    # Load data
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            simulation_data = json.load(f)
            # simulation_data.sort(key=lambda x: x["id"]) # Optional
    except FileNotFoundError:
        print(f"Input file not found: {args.input_file}")
        return

    # Run evaluation
    results = evaluate_all_cases(
        simulation_data_list=simulation_data,
        alpha=args.alpha,
        beta=args.beta,
        batch_size=args.batch_size
    )

    # Save results to output file
    if results:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # Print summary
        if "average_result" in results and results["average_result"]:
            avg = results["average_result"]
            print("\n=== Overall Average Evaluation Results ===")
            print(f"Combined Score: {avg['diagnostic_performance']['combined_score']:.4f}")
            print(f"Diagnostic Efficiency Index: {avg['diagnostic_efficiency_index']:.4f}")
            print(f"Interaction Turns: {avg['interaction_efficiency']['total_turns']:.2f}")
            print(f"Average Tokens: {avg['interaction_efficiency']['avg_tokens']:.2f}")
            print(f"Info Retrieval Precision: {avg['information_retrieval']['precision']:.4f}")
            print(f"Info Retrieval Model Score: {avg['information_retrieval']['model_score']:.4f}")
            
            # Print category-based results
            if "category_results" in results and results["category_results"]:
                print("\n=== Category-based Evaluation Results ===")
                for category, cat_avg in results["category_results"].items():
                    print(f"\nCategory: {category} ({cat_avg['case_count']} cases)")
                    print(f"  Combined Score: {cat_avg['diagnostic_performance']['combined_score']:.4f}")
                    print(f"  Diagnostic Efficiency Index: {cat_avg['diagnostic_efficiency_index']:.4f}")
                    print(f"  Interaction Turns: {cat_avg['interaction_efficiency']['total_turns']:.2f}")
                    print(f"  Average Tokens: {cat_avg['interaction_efficiency']['avg_tokens']:.2f}")
                    print(f"  Info Retrieval Model Score: {cat_avg['information_retrieval']['model_score']:.4f}")
            
            print(f"\nFull results saved to: {args.output}")
        else:
            print("\nNo results to display.")


if __name__ == "__main__":
    main()