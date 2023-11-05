from typing import Union, Tuple, List, Any
import numpy as np
import torch
import editdistance


def edit_distance(prediction_tokens: List[Any], reference_tokens: List[Any]) -> int:
    """Dynamic programming algorithm to compute the edit distance.

    Args:
        prediction_tokens: A tokenized predicted sentence
        reference_tokens: A tokenized reference sentence
    Returns:
        Edit distance between the predicted sentence and the reference sentence

    """
    dp = [[0] * (len(reference_tokens) + 1) for _ in range(len(prediction_tokens) + 1)]
    for i in range(len(prediction_tokens) + 1):
        dp[i][0] = i
    for j in range(len(reference_tokens) + 1):
        dp[0][j] = j
    for i in range(1, len(prediction_tokens) + 1):
        for j in range(1, len(reference_tokens) + 1):
            if prediction_tokens[i - 1] == reference_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    return dp[-1][-1]

def cer(
    preds: Union[str, List[Any]],
    target: Union[str, List[Any]],
) -> torch.Tensor:
    """Update the cer score with the current set of references and predictions.

    Args:
        preds: Transcription(s) to score as a string or list of strings
        target: Reference(s) for each speech input as a string or list of strings

    Returns:
        Number of edit operations to get from the reference to the prediction, summed over all samples
        Number of character overall references

    """
    # return editdistance.eval(preds, target) / len(target)
    return editdistance.eval(preds.numpy().astype(np.int32), target.numpy().astype(np.int32)) / len(target)

    if isinstance(preds, str):
        preds = [preds]
    if isinstance(target, str):
        target = [target]
    errors = torch.tensor(0, dtype=torch.float)
    total = torch.tensor(0, dtype=torch.float)
    for pred, tgt in zip(preds, target):
        pred_tokens = pred
        tgt_tokens = tgt
        errors += editdistance.eval(list(pred_tokens), list(tgt_tokens))
        total += len(tgt_tokens)
    return errors / total
