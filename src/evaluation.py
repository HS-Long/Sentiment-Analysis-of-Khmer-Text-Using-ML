"""
Model evaluation and visualization module
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score


def evaluate_model(model, X_test, y_test, label_encoder=None, model_name="Model"):
    """
    Evaluate a trained model and print metrics
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        label_encoder: LabelEncoder object for class names
        model_name (str): Name of the model for display
        
    Returns:
        dict: Dictionary containing predictions and metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"\n{'='*50}")
    print(f"{model_name} Evaluation")
    print(f"{'='*50}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"F1 Score (Macro): {f1:.4f}")
    
    # Get class names
    if label_encoder is not None:
        target_names = [label_encoder.classes_[i] for i in sorted(np.unique(y_test))]
    else:
        target_names = [str(i) for i in sorted(np.unique(y_test))]
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    return {
        'predictions': y_pred,
        'accuracy': accuracy,
        'f1_score': f1,
        'target_names': target_names
    }


def plot_confusion_matrix(y_test, y_pred, target_names, model_name="Model", 
                         cmap='Blues', save_path=None):
    """
    Plot confusion matrix
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        target_names (list): List of class names
        model_name (str): Name of the model for title
        cmap (str): Color map for heatmap
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, 
                xticklabels=target_names,
                yticklabels=target_names)
    plt.title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_sentiment_distribution(df, target_column='target', save_path=None):
    """
    Plot distribution of sentiment labels
    
    Args:
        df (pd.DataFrame): Dataframe with target column
        target_column (str): Name of the target column
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x=target_column, data=df, 
                       order=df[target_column].value_counts().index, 
                       palette="viridis", stat="percent")
    
    # Show percentages on graph
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.1f}%', (p.get_x() + p.get_width() / 2, height), 
                   ha='center', va='bottom')
    
    plt.title("Distribution of Sentiment Labels")
    plt.xlabel("Sentiment")
    plt.ylabel("Percentage")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def compare_models(results_dict, save_path=None):
    """
    Compare multiple models and visualize results
    
    Args:
        results_dict (dict): Dictionary mapping model names to accuracy scores
        save_path (str, optional): Path to save the plot
    """
    # Create comparison dataframe
    model_comparison = pd.DataFrame({
        'Model': list(results_dict.keys()),
        'Accuracy': list(results_dict.values())
    })
    
    model_comparison = model_comparison.sort_values('Accuracy', ascending=False)
    print("\nModel Comparison:")
    print(model_comparison.to_string(index=False))
    
    # Visualize comparison
    plt.figure(figsize=(12, 6))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA15E']
    bars = plt.bar(range(len(model_comparison)), model_comparison['Accuracy'], 
                   color=colors[:len(model_comparison)])
    
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.xticks(range(len(model_comparison)), model_comparison['Model'], 
               rotation=45, ha='right')
    plt.ylim([0, 1])
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, model_comparison['Accuracy'])):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.2%}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.grid(axis='y', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()
    
    # Print best model
    best_model = model_comparison.iloc[0]
    print(f"\n🎯 Best Model: {best_model['Model']}")
    print(f"   Accuracy: {best_model['Accuracy']:.2%}")
    
    return model_comparison


def save_results(results_dict, filepath):
    """
    Save evaluation results to a CSV file
    
    Args:
        results_dict (dict): Dictionary mapping model names to accuracy scores
        filepath (str): Path to save the CSV file
    """
    df = pd.DataFrame({
        'Model': list(results_dict.keys()),
        'Accuracy': list(results_dict.values())
    })
    df = df.sort_values('Accuracy', ascending=False)
    df.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")
