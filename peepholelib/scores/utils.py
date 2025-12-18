def has_score(scores, score_name, inference_names):
    """
    Return True if score_name exists in scores for every base_loader-inference
    combination in inference_names ({base_loader: [inference, ...]}).
    """
    for base_loader, inferences in inference_names.items():
        for inf in inferences:
            loader_key = f'{base_loader}-{inf}'
            if not ((scores['dataset'] == loader_key) & (scores['score name'] == score_name)).any():
                return False
    return True

def dmd_score_name(prefix, loader):
    return f'{prefix}-{loader}'