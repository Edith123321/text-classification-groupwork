"""
Minimal evaluation helper for GRU
Only create this one new file in existing evaluation directory
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def analyze_predictions(y_true, y_pred, y_proba, texts=None):
    """
    Analyze model predictions
    """
    results = {
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'classification_report': classification_report(y_true, y_pred, output_dict=True)
    }
    
    # Create detailed analysis DataFrame
    analysis_df = pd.DataFrame({
        'true_label': y_true,
        'predicted_label': y_pred,
        'probability': y_proba,
        'correct': y_true == y_pred
    })
    
    if texts is not None:
        analysis_df['text'] = texts
    
    return results, analysis_df

def plot_training_history(history, save_path=None):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    axes[0].plot(history['train_losses'], label='Train Loss')
    axes[0].plot(history['val_losses'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(history['train_accs'], label='Train Accuracy')
    axes[1].plot(history['val_accs'], label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()