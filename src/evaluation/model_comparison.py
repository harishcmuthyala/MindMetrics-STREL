"""
Model Comparison and Evaluation
Compares multiple classification models using accuracy, precision, recall, F1-score
and ROC curves
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style("whitegrid")

class ModelEvaluator:
    """Collects and compares metrics from multiple classification models"""
    
    def __init__(self):
        self.metrics = []
        self.roc_data = []
        self.all_fold_details = []
    
    def add_model(self, model_name, accuracy, precision, recall, f1_score, fpr=None, tpr=None, tn=None, fp=None, fn=None, tp=None):
        self.metrics.append({
            'Model': model_name, 
            'Accuracy': accuracy, 
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1_score,
            'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp
        })
        if fpr is not None and tpr is not None:
            self.roc_data.append({'model': model_name, 'fpr': fpr, 'tpr': tpr})

    def add_fold_details(self, fold_details):
        """Collect per-fold metrics from a model"""
        self.all_fold_details.extend(fold_details)

    def save_fold_metrics(self, filepath):
        """Save per-fold metrics to CSV"""
        pd.DataFrame(self.all_fold_details).to_csv(filepath, index=False)
        print(f"Fold metrics saved to: {filepath}")

    def plot_fold_accuracy(self, save_path=None):
        """Plot accuracy per fold for each model as a point-line chart"""
        if not self.all_fold_details:
            print("No fold details available")
            return

        df = pd.DataFrame(self.all_fold_details)
        colors = ['steelblue', 'coral', 'green', 'purple']

        plt.figure(figsize=(10, 6))
        for i, model in enumerate(df['model'].unique()):
            model_df = df[df['model'] == model].sort_values('fold')
            plt.plot(model_df['fold'], model_df['accuracy'], marker='o', label=model,
                     color=colors[i % len(colors)], linewidth=2, markersize=7)

        plt.xlabel('Fold')
        plt.ylabel('Accuracy')
        plt.title('Accuracy per Fold - Model Comparison')
        plt.xticks(range(1, 6))
        plt.ylim([0, 1.05])
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Fold accuracy plot saved to: {save_path}")
        plt.show()
    
    def print_summary(self):
        """Print summary table of all model metrics"""
        df = pd.DataFrame(self.metrics)
        print("\n" + "=" * 80)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 80)
        print(df[['Model','Accuracy','Precision','Recall','F1-Score']].to_string(index=False))
        print("=" * 80)

    def plot_confusion_matrices(self, save_path=None):
        """Plot confusion matrices for all models as heatmaps"""
        n = len(self.metrics)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
        if n == 1:
            axes = [axes]

        for ax, m in zip(axes, self.metrics):
            tn, fp, fn, tp = m.get('TN', 0), m.get('FP', 0), m.get('FN', 0), m.get('TP', 0)
            cm = np.array([[tn, fp], [fn, tp]])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['Pred: No Stress', 'Pred: Stress'],
                        yticklabels=['Actual: No Stress', 'Actual: Stress'])
            ax.set_title(m['Model'], fontweight='bold')

        plt.suptitle('Confusion Matrices (Summed across 5 folds)', fontweight='bold', y=1.02)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrices saved to: {save_path}")
        plt.show()
    
    def plot_metrics(self, save_path=None):
        """
        Plot grouped bar charts comparing all metrics across models
        
        Args:
            save_path: Path to save the plot (optional)
        """
        df = pd.DataFrame(self.metrics)
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        colors = ['steelblue', 'coral', 'green', 'purple', 'orange']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            x = range(len(df))
            bars = ax.bar(x, df[metric], color=[colors[i % len(colors)] for i in x], alpha=0.8)
            
            ax.set_ylabel(metric, fontsize=12)
            ax.set_title(f'{metric} Comparison', fontweight='bold', fontsize=14)
            ax.set_ylim([0, 1.1])  # Extra space for labels
            ax.set_xticks(x)
            ax.set_xticklabels(df['Model'], rotation=45, ha='right', fontsize=10)
            
            # Add value labels on bars
            for i, (bar, v) in enumerate(zip(bars, df[metric])):
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.03, 
                       f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nMetrics saved to: {save_path}")
        plt.show()
    
    def plot_roc_curves(self, save_path=None):
        """
        Plot ROC curves for all models on the same graph
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if not self.roc_data:
            print("\nNo ROC data available")
            return
        
        plt.figure(figsize=(10, 8))
        colors = ['steelblue', 'coral', 'green', 'purple', 'orange']
        
        for i, data in enumerate(self.roc_data):
            plt.plot(data['fpr'], data['tpr'], color=colors[i % len(colors)], lw=2,
                    label=f"{data['model']}")
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Comparison', fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curves saved to: {save_path}")
        plt.show()
    
    def save_metrics(self, filepath):
        """
        Save metrics to CSV file
        
        Args:
            filepath: Path to save the CSV file
        """
        pd.DataFrame(self.metrics).to_csv(filepath, index=False)
        print(f"Metrics saved to: {filepath}")


# ─────────────────────────────────────────────
# COMPARE MODELS
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
    
    evaluator = ModelEvaluator()
    
    # XGBoost
    from xg_boost_model import train_and_evaluate_xgboost
    result = train_and_evaluate_xgboost()
    evaluator.add_model(result['model_name'], result['accuracy'], result['precision'],
        result['recall'], result['f1_score'], fpr=result['fpr'], tpr=result['tpr'],
        tn=result['tn'], fp=result['fp'], fn=result['fn'], tp=result['tp'])
    evaluator.add_fold_details(result['fold_details'])
    
    # Random Forest
    from random_forest_model import train_and_evaluate_random_forest
    result = train_and_evaluate_random_forest()
    evaluator.add_model(result['model_name'], result['accuracy'], result['precision'],
        result['recall'], result['f1_score'], fpr=result['fpr'], tpr=result['tpr'],
        tn=result['tn'], fp=result['fp'], fn=result['fn'], tp=result['tp'])
    evaluator.add_fold_details(result['fold_details'])
    
    # SVM
    from svm_model import train_svm
    result = train_svm()
    evaluator.add_model(result['model_name'], result['accuracy'], result['precision'],
        result['recall'], result['f1_score'], fpr=result['fpr'], tpr=result['tpr'],
        tn=result['tn'], fp=result['fp'], fn=result['fn'], tp=result['tp'])
    evaluator.add_fold_details(result['fold_details'])
    
    # Logistic Regression (PyTorch)
    from logreg_model import train_and_evaluate_logistic_regression
    result = train_and_evaluate_logistic_regression()
    evaluator.add_model(result['model_name'], result['accuracy'], result['precision'],
        result['recall'], result['f1_score'], fpr=result['fpr'], tpr=result['tpr'],
        tn=result['tn'], fp=result['fp'], fn=result['fn'], tp=result['tp'])
    evaluator.add_fold_details(result['fold_details'])
    
    evaluator.print_summary()
    evaluator.plot_confusion_matrices(save_path="../../results/metrics/confusion_matrices.png")
    evaluator.plot_metrics(save_path="../../results/metrics/model_comparison.png")
    evaluator.plot_roc_curves(save_path="../../results/metrics/roc_curves.png")
    evaluator.save_metrics("../../results/metrics/model_metrics.csv")
    evaluator.save_fold_metrics("../../results/metrics/fold_metrics.csv")
    evaluator.plot_fold_accuracy(save_path="../../results/metrics/fold_accuracy.png")
