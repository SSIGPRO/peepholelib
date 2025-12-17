#python stuff
import pandas as pd
import seaborn as sb
import numpy as np
from matplotlib import pyplot as plt

#Sklearn stuff
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score, recall_score, f1_score

#torch stuff
import torch

def comprehensive_test_analysis(model, test_loader, device, threshold, path):
        """
        Comprehensive test analysis with accuracy, score distributions, and AUROC
        
        Args:
            test_loader: Test data loader
            threshold: Decision threshold for binary classification
        """

        model._model.eval()
        all_scores = []
        all_predictions = []
        all_targets = []
        
        print("Running comprehensive test analysis...")
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                
                # Get raw logits and convert to probabilities
                logits = model(data).squeeze()
                probabilities = torch.sigmoid(logits)
                
                # Apply threshold for predictions
                predictions = (probabilities > threshold).float()
                
                # Store results
                all_scores.extend(probabilities.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Convert to numpy arrays
        all_scores = np.array(all_scores)
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # Calculate metrics
        accuracy = np.mean(all_predictions == all_targets)
        auc_score = roc_auc_score(all_targets, all_scores)
        avg_precision = average_precision_score(all_targets, all_scores)
        
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE TEST ANALYSIS")
        print(f"{'='*60}")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"AUC-ROC: {auc_score:.4f}")
        print(f"Average Precision (AP): {avg_precision:.4f}")
        
        # Class distribution
        class_0_count = np.sum(all_targets == 0)
        class_1_count = np.sum(all_targets == 1)
        print(f"\nClass Distribution:")
        print(f"Class 0: {class_0_count} samples ({class_0_count/len(all_targets)*100:.1f}%)")
        print(f"Class 1: {class_1_count} samples ({class_1_count/len(all_targets)*100:.1f}%)")
        
        # Detailed classification report
        print(f"\nDetailed Classification Report:")
        print(classification_report(all_targets, all_predictions, 
                                  target_names=['Class 0', 'Class 1'], digits=4))
        print(threshold)
        # Create comprehensive visualization
        _plot_test_analysis(all_targets, all_scores, all_predictions, 
                               auc_score, avg_precision, threshold, path)
        
        return {
            'accuracy': accuracy,
            'auc_roc': auc_score,
            'avg_precision': avg_precision,
            'predictions': all_predictions,
            'targets': all_targets,
            'scores': all_scores
        }
    
def _plot_test_analysis(targets, scores, predictions, auc_score, avg_precision, threshold, path):
    """Create comprehensive visualization of test results"""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Score Distribution
    ax1 = plt.subplot(2, 3, 1)
    
    # Separate scores by class
    class_0_scores = scores[targets == 0]
    class_1_scores = scores[targets == 1]
    
    # Plot histograms
    plt.hist(class_0_scores, bins=50, alpha=0.7, label='Class 0', color='red', density=True)
    plt.hist(class_1_scores, bins=50, alpha=0.7, label='Class 1', color='blue', density=True)
    plt.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
    
    plt.xlabel('Prediction Score (Probability)')
    plt.ylabel('Density')
    plt.title('Score Distribution by Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    plt.text(0.02, 0.95, f'Class 0: μ={np.mean(class_0_scores):.3f}, σ={np.std(class_0_scores):.3f}', 
            transform=ax1.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='red', alpha=0.1))
    plt.text(0.02, 0.88, f'Class 1: μ={np.mean(class_1_scores):.3f}, σ={np.std(class_1_scores):.3f}', 
            transform=ax1.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='blue', alpha=0.1))
    
    # 2. ROC Curve
    ax2 = plt.subplot(2, 3, 2)
    fpr, tpr, _ = roc_curve(targets, scores)
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Precision-Recall Curve
    ax3 = plt.subplot(2, 3, 3)
    precision, recall, _ = precision_recall_curve(targets, scores)
    plt.plot(recall, precision, linewidth=2, label=f'PR Curve (AP = {avg_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Confusion Matrix
    ax4 = plt.subplot(2, 3, 4)
    cm = confusion_matrix(targets, predictions)
    sb.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Class 0', 'Class 1'], 
                yticklabels=['Class 0', 'Class 1'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # 5. Score Box Plot
    ax5 = plt.subplot(2, 3, 5)
    box_data = [class_0_scores, class_1_scores]
    box_plot = plt.boxplot(box_data, labels=['Class 0', 'Class 1'], patch_artist=True)
    box_plot['boxes'][0].set_facecolor('red')
    box_plot['boxes'][0].set_alpha(0.7)
    box_plot['boxes'][1].set_facecolor('blue')
    box_plot['boxes'][1].set_alpha(0.7)
    plt.ylabel('Prediction Score')
    plt.title('Score Distribution Box Plot')
    plt.grid(True, alpha=0.3)
    
    # 6. Class-wise Performance Metrics
    ax6 = plt.subplot(2, 3, 6)
    
    # Calculate per-class metrics
    
    precision_0 = precision_score(targets, predictions, pos_label=0)
    precision_1 = precision_score(targets, predictions, pos_label=1)
    recall_0 = recall_score(targets, predictions, pos_label=0)
    recall_1 = recall_score(targets, predictions, pos_label=1)
    f1_0 = f1_score(targets, predictions, pos_label=0)
    f1_1 = f1_score(targets, predictions, pos_label=1)
    
    metrics_data = {
        'Precision': [precision_0, precision_1],
        'Recall': [recall_0, recall_1],
        'F1-Score': [f1_0, f1_1]
    }
    
    x = np.arange(len(metrics_data))
    width = 0.35
    
    class_0_values = [metrics_data['Precision'][0], metrics_data['Recall'][0], metrics_data['F1-Score'][0]]
    class_1_values = [metrics_data['Precision'][1], metrics_data['Recall'][1], metrics_data['F1-Score'][1]]
    
    plt.bar(x - width/2, class_0_values, width, label='Class 0', color='red', alpha=0.7)
    plt.bar(x + width/2, class_1_values, width, label='Class 1', color='blue', alpha=0.7)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Per-Class Performance Metrics')
    plt.xticks(x, ['Precision', 'Recall', 'F1-Score'])
    plt.legend()
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (v0, v1) in enumerate(zip(class_0_values, class_1_values)):
        plt.text(i - width/2, v0 + 0.01, f'{v0:.3f}', ha='center', va='bottom')
        plt.text(i + width/2, v1 + 0.01, f'{v1:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Print additional insights
    print(f"\n{'='*60}")
    print("INSIGHTS AND RECOMMENDATIONS")
    print(f"{'='*60}")
    
    # Score separation analysis
    score_separation = np.mean(class_1_scores) - np.mean(class_0_scores)
    print(f"Score Separation: {score_separation:.4f}")
    
    if score_separation > 0.3:
        print("✓ Good class separation - model distinguishes well between classes")
    elif score_separation > 0.1:
        print("⚠ Moderate class separation - consider model improvements")
    else:
        print("✗ Poor class separation - model struggles to distinguish classes")
    
    # Threshold analysis
    optimal_threshold_idx = np.argmax(tpr - fpr)
    optimal_threshold = _[optimal_threshold_idx] if len(_) > optimal_threshold_idx else threshold
    print(f"Suggested optimal threshold: {optimal_threshold:.4f} (current: {threshold})")
    
    # Imbalance impact
    if abs(len(class_0_scores) - len(class_1_scores)) / len(scores) > 0.3:
        print("⚠ Significant class imbalance detected - consider using class weights or resampling")
    
    plt.savefig(path/'overview.png')