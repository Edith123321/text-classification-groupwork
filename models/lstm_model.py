"""
LSTM Model for SMS Spam Classification

This module implements a Long Short-Term Memory (LSTM) neural network
for binary text classification (spam vs ham).

Why LSTM for this task:
- Captures sequential patterns in text
- Handles variable-length inputs
- Remembers long-term dependencies
- Better than simple RNN (no vanishing gradient)
- State-of-the-art for text classification

LSTM advantages over alternatives:
- vs RNN: LSTM has gates to prevent vanishing gradients
- vs GRU: LSTM has separate forget/input gates (more control)
- vs CNN: Better for sequential dependencies in text
- vs Transformer: More efficient for short texts and small datasets
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os


class LSTMClassifier:
    """
    LSTM-based binary classifier for SMS spam detection.
    
    Why this architecture:
    - LSTM layer: Captures sequential patterns in embeddings
    - Dropout: Prevents overfitting on small dataset
    - Dense layers: Learn classification decision boundaries
    - Binary output: Spam probability (0-1)
    """
    
    def __init__(self, input_dim, lstm_units=64, dropout_rate=0.5, learning_rate=0.001):
        """
        Initialize LSTM classifier parameters.
        
        Parameters explained:
        
        input_dim:
        - Dimensionality of input embeddings
        - TF-IDF: 5000 (sparse high-dimensional)
        - Word2Vec Skip-gram: 100 (dense low-dimensional)
        - Word2Vec CBOW: 100 (dense low-dimensional)
        - Must match embedding output size
        
        lstm_units=64:
        - Number of LSTM cells (memory units)
        - Trade-off: More units = more capacity but more parameters
        - 64 is good balance for small dataset (5572 samples)
        - Each unit learns different sequential patterns
        - Why 64: Captures complex patterns without overfitting
        - Alternative values: 32 (simpler), 128 (more complex)
        
        dropout_rate=0.5:
        - Fraction of units to randomly drop during training
        - Critical for preventing overfitting on small datasets
        - 0.5 means 50% of neurons dropped each batch
        - Forces network to learn robust features
        - Why 0.5: Standard value, proven effective
        - Lower (0.3): Less regularization, may overfit
        - Higher (0.7): More regularization, may underfit
        
        learning_rate=0.001:
        - Step size for gradient descent optimization
        - Trade-off: Large = faster but unstable, Small = stable but slow
        - 0.001 is Adam optimizer default (good starting point)
        - Why 0.001: Balances training speed and stability
        - Can adjust based on training curves
        """
        self.input_dim = input_dim
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
        
    def build_model(self, use_bidirectional=True):
        """
        Build LSTM model architecture.
        
        Architecture choices explained:
        
        Why this specific architecture:
        1. Reshape layer: LSTM expects 3D input (samples, timesteps, features)
        2. LSTM layer: Learns sequential patterns
        3. Dropout: Prevents overfitting
        4. Dense: Final classification layer
        5. Sigmoid: Output probability (0=ham, 1=spam)
        
        use_bidirectional=True:
        - Bidirectional LSTM processes sequence forward AND backward
        - Forward: Learns patterns from start to end
        - Backward: Learns patterns from end to start
        - Combined: Full context in both directions
        - Why useful: "Call now for free" - "free" at end is important
        - Trade-off: 2x parameters but better performance
        - Recommended for text where word order matters both ways
        """
        print("\nBuilding LSTM model...")
        print(f"  Input dimension: {self.input_dim}")
        print(f"  LSTM units: {self.lstm_units}")
        print(f"  Dropout rate: {self.dropout_rate}")
        print(f"  Bidirectional: {use_bidirectional}")
        
        self.model = Sequential([
            # Input layer
            # Why reshape: LSTM expects (samples, timesteps, features)
            # We have (samples, features) from embeddings
            # Reshape to (samples, 1, features) - treat embedding as single timestep
            # Alternative: Could use multiple timesteps if we had sequential embeddings
            layers.Reshape((1, self.input_dim), input_shape=(self.input_dim,)),
            
            # Bidirectional LSTM layer
            # Why bidirectional: Processes sequence both directions
            # return_sequences=False: Only return last output (for classification)
            # If return_sequences=True: Would return output for each timestep (for seq2seq)
            Bidirectional(LSTM(self.lstm_units, return_sequences=False)) if use_bidirectional 
            else LSTM(self.lstm_units, return_sequences=False),
            
            # Dropout for regularization
            # Why here: Prevents LSTM from overfitting
            # Applied after LSTM before dense layer
            # During training: Randomly drops lstm_units * 2 * dropout_rate neurons
            # During inference: Uses all neurons with scaled weights
            Dropout(self.dropout_rate),
            
            # Dense hidden layer
            # Why 32 units: Intermediate layer between LSTM and output
            # ReLU activation: Non-linearity allows learning complex patterns
            # Could skip this layer for simpler model
            # Helps when LSTM output is high-dimensional (128 from bidirectional 64)
            Dense(32, activation='relu'),
            
            # Another dropout
            # Why: Additional regularization for dense layer
            # Especially important with small training set
            Dropout(self.dropout_rate),
            
            # Output layer
            # Why 1 unit: Binary classification
            # Why sigmoid: Outputs probability between 0 and 1
            # threshold 0.5: >0.5 = spam, <0.5 = ham
            # Could use 2 units with softmax for multi-class
            Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        # Why Adam optimizer: Adaptive learning rate, works well out-of-box
        # Why binary_crossentropy: Standard loss for binary classification
        # Metrics: Track accuracy and AUC during training
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        # Print model summary
        print("\nModel architecture:")
        self.model.summary()
        
        # Count parameters
        trainable_params = self.model.count_params()
        print(f"\nTotal trainable parameters: {trainable_params:,}")
        
        # Why parameter count matters:
        # - More parameters = more capacity but risk overfitting
        # - Rule of thumb: Need 10x samples per parameter
        # - With 3343 training samples and ~50k parameters: Moderate risk
        # - Dropout helps manage this risk
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, class_weight=None, use_early_stopping=True):
        """
        Train LSTM model on training data.
        
        Parameters explained:
        
        epochs=50:
        - Number of complete passes through training data
        - Why 50: Usually enough for convergence with early stopping
        - Early stopping prevents training full 50 if no improvement
        - Could increase to 100 for more thorough training
        
        batch_size=32:
        - Number of samples per gradient update
        - Trade-offs:
          - Larger batch (64, 128): More stable gradients, faster computation
          - Smaller batch (16, 8): More frequent updates, better generalization
        - Why 32: Good balance, standard choice
        - With 3343 training samples: 105 batches per epoch
        
        class_weight=None:
        - Weight for each class to handle imbalance
        - Dataset: 87% ham, 13% spam (imbalanced)
        - Could use: {0: 1.0, 1: 6.7} to balance classes
        - We'll handle with stratified splits and F1 metric instead
        - Alternative: SMOTE oversampling
        
        use_early_stopping=True:
        - Whether to use early stopping callback
        - Set to False to train for full number of epochs
        """
        print("\nTraining LSTM model...")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val)}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Early stopping: {use_early_stopping}")
        
        # Define callbacks
        # Why callbacks: Monitor training and prevent overfitting
        
        callback_list = []
        
        if use_early_stopping:
            # Early stopping: Stop if validation loss doesn't improve
            # Why patience=50: Allow many epochs of no improvement (suitable for 200 epochs)
            # restore_best_weights: Load best model, not last model
            # This prevents overfitting to training data
            early_stop = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=50,
                restore_best_weights=True,
                verbose=1
            )
            callback_list.append(early_stop)
        
        # Reduce learning rate on plateau
        # Why: If validation loss stops decreasing, reduce learning rate
        # Helps escape local minima and fine-tune weights
        # Factor=0.5: Reduce learning rate by half
        # Patience=10: Wait 10 epochs before reducing (suitable for 200 epochs)
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=0.00001,
            verbose=1
        )
        callback_list.append(reduce_lr)
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=callback_list,
            verbose=1
        )
        
        print("\nTraining complete!")
        print(f"  Trained for {len(self.history.history['loss'])} epochs")
        print(f"  Best validation loss: {min(self.history.history['val_loss']):.4f}")
        print(f"  Best validation accuracy: {max(self.history.history['val_accuracy']):.4f}")
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test set with comprehensive metrics.
        
        Why these metrics:
        
        1. Accuracy: Overall correctness
           - Simple to understand
           - Misleading with imbalanced data (87% ham)
           - Predicting all ham gives 87% accuracy!
        
        2. Precision: TP / (TP + FP)
           - Of predicted spam, how many actually spam?
           - Important: Don't want to block legitimate messages
           - High precision = Few false positives
        
        3. Recall: TP / (TP + FN)
           - Of actual spam, how many did we catch?
           - Important: Don't want spam reaching users
           - High recall = Few false negatives
        
        4. F1-Score: 2 * (precision * recall) / (precision + recall)
           - Harmonic mean of precision and recall
           - Best single metric for imbalanced data
           - PRIMARY METRIC for this task
           - Balances precision and recall
        
        5. AUC-ROC: Area under ROC curve
           - Measures separability of classes
           - Independent of classification threshold
           - Higher = Better class separation
           - Good complementary metric
        
        Returns:
            Dictionary with all metrics
        """
        print("\nEvaluating model on test set...")
        print(f"  Test samples: {len(X_test)}")
        
        # Get predictions
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).reshape(-1)
        
        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Classification report gives precision, recall, F1 per class
        report = classification_report(y_test, y_pred, 
                                      target_names=['ham', 'spam'],
                                      output_dict=True)
        
        # Extract metrics
        metrics = {
            'accuracy': accuracy,
            'precision_ham': report['ham']['precision'],
            'recall_ham': report['ham']['recall'],
            'f1_ham': report['ham']['f1-score'],
            'precision_spam': report['spam']['precision'],
            'recall_spam': report['spam']['recall'],
            'f1_spam': report['spam']['f1-score'],
            'macro_f1': report['macro avg']['f1-score'],
            'weighted_f1': report['weighted avg']['f1-score'],
            'auc': auc_score
        }
        
        # Print results
        print("\n" + "="*60)
        print("TEST SET EVALUATION RESULTS")
        print("="*60)
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:    {accuracy:.4f}")
        print(f"  AUC-ROC:     {auc_score:.4f}")
        print(f"  Macro F1:    {metrics['macro_f1']:.4f}")
        print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
        
        print(f"\nHam (Legitimate Messages):")
        print(f"  Precision: {metrics['precision_ham']:.4f}")
        print(f"  Recall:    {metrics['recall_ham']:.4f}")
        print(f"  F1-Score:  {metrics['f1_ham']:.4f}")
        
        print(f"\nSpam:")
        print(f"  Precision: {metrics['precision_spam']:.4f}")
        print(f"  Recall:    {metrics['recall_spam']:.4f}")
        print(f"  F1-Score:  {metrics['f1_spam']:.4f}")
        
        # Print confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"              Ham    Spam")
        print(f"  Actual Ham   {cm[0][0]:4d}   {cm[0][1]:4d}")
        print(f"       Spam   {cm[1][0]:4d}   {cm[1][1]:4d}")
        
        return metrics
    
    def plot_training_history(self, save_path=None):
        """
        Plot training and validation metrics over epochs.
        
        Why plot training history:
        - Visualize learning progress
        - Identify overfitting (validation diverges from training)
        - Validate early stopping decision
        - Diagnose training issues
        """
        if self.history is None:
            print("No training history available. Train model first.")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot loss
        axes[0].plot(self.history.history['loss'], label='Training Loss')
        axes[0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot accuracy
        axes[1].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[1].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot AUC
        axes[2].plot(self.history.history['auc'], label='Training AUC')
        axes[2].plot(self.history.history['val_auc'], label='Validation AUC')
        axes[2].set_title('Model AUC-ROC', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('AUC')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nSaved training history plot to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, X_test, y_test, save_path=None):
        """
        Plot confusion matrix heatmap.
        
        Why confusion matrix:
        - Shows exactly where model makes mistakes
        - TN (top-left): Correctly identified ham
        - FP (top-right): Ham misclassified as spam (bad for UX)
        - FN (bottom-left): Spam misclassified as ham (security risk)
        - TP (bottom-right): Correctly identified spam
        """
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).reshape(-1)
        
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Ham', 'Spam'],
                   yticklabels=['Ham', 'Spam'])
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nSaved confusion matrix to {save_path}")
        
        plt.show()
    
    def save_model(self, path):
        """Save trained model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"\nSaved model to {path}")
    
    @staticmethod
    def load_model(path):
        """Load saved model."""
        model = keras.models.load_model(path)
        print(f"\nLoaded model from {path}")
        return model


if __name__ == "__main__":
    print("LSTM Classifier module ready!")
    print("\nTo use:")
    print("1. Load embeddings (TF-IDF, Skip-gram, or CBOW)")
    print("2. Create classifier: model = LSTMClassifier(input_dim=embedding_dim)")
    print("3. Build architecture: model.build_model()")
    print("4. Train: model.train(X_train, y_train, X_val, y_val)")
    print("5. Evaluate: model.evaluate(X_test, y_test)")
