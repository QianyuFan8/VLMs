"""
Evaluation Metrics Module for TextVQA
Includes: Accuracy, BLEU, METEOR, ROUGE, F1 Score
"""

import re
from typing import List, Dict, Tuple
from collections import Counter
import numpy as np


def normalize_answer(answer: str) -> str:
    """
    Normalize answer string for comparison
    - Lowercase
    - Remove punctuation
    - Remove extra whitespace
    """
    answer = answer.lower().strip()
    # Remove punctuation
    answer = re.sub(r'[^\w\s]', '', answer)
    # Remove extra whitespace
    answer = ' '.join(answer.split())
    return answer


def textvqa_accuracy(prediction: str, ground_truths: List[str]) -> float:
    """
    Calculate TextVQA accuracy for a single prediction
    Uses exact string matching with normalization
    Following TextVQA evaluation protocol:
    acc = min(1, #humans that provided that answer / 3)

    Args:
        prediction: Model prediction
        ground_truths: List of human answers (typically 10 annotators)

    Returns:
        Accuracy score between 0 and 1
    """
    if ground_truths is None or len(ground_truths) == 0:
        return 0.0

    pred_normalized = normalize_answer(prediction)
    gt_normalized = [normalize_answer(gt) for gt in ground_truths]

    # Count how many annotators gave the same answer as prediction
    match_count = sum(1 for gt in gt_normalized if gt == pred_normalized)

    # TextVQA accuracy formula
    return min(1.0, match_count / 3.0)


def compute_accuracy(predictions: List[str], ground_truths: List[List[str]]) -> Dict[str, float]:
    """
    Compute accuracy metrics over a batch of predictions

    Args:
        predictions: List of model predictions
        ground_truths: List of answer lists (one list per question)

    Returns:
        Dictionary with accuracy metrics
    """
    accuracies = []
    exact_matches = []

    for pred, gts in zip(predictions, ground_truths):
        acc = textvqa_accuracy(pred, gts)
        accuracies.append(acc)

        # Also compute exact match (any annotator)
        pred_norm = normalize_answer(pred)
        gt_norms = [normalize_answer(gt) for gt in gts]
        exact_matches.append(1.0 if pred_norm in gt_norms else 0.0)

    return {
        "accuracy": np.mean(accuracies),
        "exact_match": np.mean(exact_matches),
        "num_samples": len(predictions),
    }


def compute_bleu(prediction: str, references: List[str], n: int = 4) -> float:
    """
    Compute BLEU score for a single prediction

    Args:
        prediction: Model prediction
        references: Reference answers
        n: Maximum n-gram order

    Returns:
        BLEU score
    """
    from collections import Counter
    import math

    def get_ngrams(text: str, n: int) -> Counter:
        words = text.split()
        return Counter(tuple(words[i:i+n]) for i in range(max(0, len(words) - n + 1)))

    pred_words = normalize_answer(prediction).split()
    if not pred_words:
        return 0.0

    # Compute n-gram precisions
    precisions = []
    for i in range(1, n + 1):
        pred_ngrams = get_ngrams(normalize_answer(prediction), i)
        if not pred_ngrams:
            precisions.append(0.0)
            continue

        max_counts = Counter()
        for ref in references:
            ref_ngrams = get_ngrams(normalize_answer(ref), i)
            for ngram in pred_ngrams:
                max_counts[ngram] = max(max_counts[ngram], ref_ngrams[ngram])

        clipped_count = sum(min(count, max_counts[ngram]) for ngram, count in pred_ngrams.items())
        total_count = sum(pred_ngrams.values())
        precisions.append(clipped_count / total_count if total_count > 0 else 0.0)

    # Compute brevity penalty
    pred_len = len(pred_words)
    ref_lens = [len(normalize_answer(ref).split()) for ref in references]
    closest_ref_len = min(ref_lens, key=lambda x: (abs(x - pred_len), x)) if ref_lens else 0

    if pred_len == 0:
        bp = 0
    elif pred_len >= closest_ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - closest_ref_len / pred_len)

    # Compute BLEU score
    if min(precisions) > 0:
        log_precisions = [math.log(p) for p in precisions]
        bleu = bp * math.exp(sum(log_precisions) / len(log_precisions))
    else:
        bleu = 0.0

    return bleu


def compute_rouge_l(prediction: str, reference: str) -> Dict[str, float]:
    """
    Compute ROUGE-L score based on longest common subsequence

    Args:
        prediction: Model prediction
        reference: Reference answer

    Returns:
        Dictionary with precision, recall, and F1
    """
    def lcs_length(x: List[str], y: List[str]) -> int:
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()

    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs = lcs_length(pred_tokens, ref_tokens)

    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def compute_meteor(prediction: str, references: List[str]) -> float:
    """
    Simplified METEOR score computation
    Based on unigram precision and recall with harmonic mean

    Args:
        prediction: Model prediction
        references: Reference answers

    Returns:
        METEOR score
    """
    pred_tokens = set(normalize_answer(prediction).split())
    if not pred_tokens:
        return 0.0

    best_score = 0.0
    for ref in references:
        ref_tokens = set(normalize_answer(ref).split())
        if not ref_tokens:
            continue

        matches = len(pred_tokens & ref_tokens)
        precision = matches / len(pred_tokens) if pred_tokens else 0
        recall = matches / len(ref_tokens) if ref_tokens else 0

        if precision + recall > 0:
            # METEOR uses weighted harmonic mean (recall weighted higher)
            alpha = 0.9  # Weight for recall
            score = (precision * recall) / (alpha * precision + (1 - alpha) * recall)
            best_score = max(best_score, score)

    return best_score


def compute_f1_score(prediction: str, ground_truths: List[str]) -> float:
    """
    Compute token-level F1 score

    Args:
        prediction: Model prediction
        ground_truths: Reference answers

    Returns:
        Maximum F1 score across all references
    """
    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0

    best_f1 = 0.0
    for gt in ground_truths:
        gt_tokens = normalize_answer(gt).split()
        if not gt_tokens:
            continue

        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_common = sum(common.values())

        if num_common == 0:
            continue

        precision = num_common / len(pred_tokens)
        recall = num_common / len(gt_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)

    return best_f1


def compute_all_metrics(
    predictions: List[str],
    ground_truths: List[List[str]]
) -> Dict[str, float]:
    """
    Compute all evaluation metrics

    Args:
        predictions: List of model predictions
        ground_truths: List of answer lists

    Returns:
        Dictionary with all metrics
    """
    # Accuracy metrics
    accuracy_metrics = compute_accuracy(predictions, ground_truths)

    # Other metrics
    bleu_scores = []
    meteor_scores = []
    rouge_l_f1_scores = []
    f1_scores = []

    for pred, gts in zip(predictions, ground_truths):
        bleu_scores.append(compute_bleu(pred, gts))
        meteor_scores.append(compute_meteor(pred, gts))
        f1_scores.append(compute_f1_score(pred, gts))

        # ROUGE-L against best matching reference
        best_rouge = 0.0
        for gt in gts:
            rouge = compute_rouge_l(pred, gt)
            best_rouge = max(best_rouge, rouge["f1"])
        rouge_l_f1_scores.append(best_rouge)

    metrics = {
        "accuracy": accuracy_metrics["accuracy"],
        "exact_match": accuracy_metrics["exact_match"],
        "bleu": np.mean(bleu_scores),
        "meteor": np.mean(meteor_scores),
        "rouge_l": np.mean(rouge_l_f1_scores),
        "f1": np.mean(f1_scores),
        "num_samples": len(predictions),
    }

    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Results"):
    """Pretty print evaluation metrics"""
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    print(f"Number of samples: {metrics.get('num_samples', 'N/A')}")
    print(f"-"*50)
    print(f"Accuracy (Primary):     {metrics.get('accuracy', 0.0)*100:.2f}%")
    print(f"Exact Match:            {metrics.get('exact_match', 0.0)*100:.2f}%")
    print(f"-"*50)
    print(f"BLEU:                   {metrics.get('bleu', 0.0)*100:.2f}")
    print(f"METEOR:                 {metrics.get('meteor', 0.0)*100:.2f}")
    print(f"ROUGE-L:                {metrics.get('rouge_l', 0.0)*100:.2f}")
    print(f"F1 Score:               {metrics.get('f1', 0.0)*100:.2f}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    # Test metrics
    predictions = ["nokia", "samsung", "apple iphone"]
    ground_truths = [
        ["nokia", "nokia", "nokia", "toshiba"],
        ["samsung", "samsung galaxy", "samsung"],
        ["iphone", "apple", "iphone x"],
    ]

    metrics = compute_all_metrics(predictions, ground_truths)
    print_metrics(metrics, "Test Metrics")
