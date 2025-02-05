from loguru import logger
from detectors.utils.metrics import get_precision_recall_metrics, get_roc_metrics

# TODO: Rename function
# Modified code from: DetectGPT
def run_perturbation_experiment(results, predictions, name, info, detector):
    fpr, tpr, roc_auc = get_roc_metrics(predictions['human'], predictions['llm'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['human'], predictions['llm'])
    logger.info(f"Results preview: {name} ROC AUC: {roc_auc}, PR AUC: {pr_auc}")
    return {
        'name': name,
        'detector': detector,
        'predictions': predictions,
        'info': info,
        'raw_results': results,
        'metrics': {
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
        },
        'pr_metrics': {
            'pr_auc': pr_auc,
            'precision': p,
            'recall': r,
        },
        'loss': 1 - pr_auc,
    }