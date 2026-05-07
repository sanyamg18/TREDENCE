"""
Training pipeline for Edhas - Self-Pruning Neural Network
Trains model with 3 different lambda values and compares results
Author: Kudrat Anand (RA2311026010844)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import copy
from tqdm import tqdm

from model import GreenSparseNet
from utils import (get_device, load_digits_dataset, create_dataloaders, 
                   compute_cost_savings, get_model_size_mb, TrainingLogger)


def train_model(model, train_loader, test_loader, lambda_sparse, 
                epochs=20, lr=0.001, logger=None):
    """
    Train the self-pruning neural network
    
    Args:
        model: GreenSparseNet instance
        train_loader: Training data loader
        test_loader: Test data loader
        lambda_sparse: Sparsity regularization coefficient
        epochs: Number of training epochs
        lr: Learning rate
        logger: TrainingLogger instance
    
    Returns:
        dict: Training results
    """
    device = get_device()
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    if logger:
        logger.log(f"Starting training with lambda={lambda_sparse}, epochs={epochs}")
    
    start_time = time.time()
    
    best_accuracy = 0
    best_model_state = None
    train_losses = []
    test_accuracies = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            
            # Classification loss
            class_loss = criterion(output, target)
            
            # Sparsity loss (L1 on gates)
            sparsity_loss = model.get_gate_l1_loss()
            
            # Total loss
            loss = class_loss + lambda_sparse * sparsity_loss
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'class': f'{class_loss.item():.4f}',
                'sparse': f'{sparsity_loss.item():.2f}'
            })
        
        train_acc = 100. * correct / total
        
        # Test phase
        test_acc, _ = evaluate_model(model, test_loader, device)
        
        train_losses.append(epoch_loss / len(train_loader))
        test_accuracies.append(test_acc)
        
        # Track best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_model_state = copy.deepcopy(model.state_dict())
        
        if logger and (epoch + 1) % 5 == 0:
            sparsity = model.get_total_sparsity()
            logger.log(f"Epoch {epoch+1}: Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%, Sparsity={sparsity:.1f}%")
    
    training_time = time.time() - start_time
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Final evaluation
    final_acc, inference_time = evaluate_model(model, test_loader, device, return_time=True)
    final_sparsity = model.get_total_sparsity()
    model_size = get_model_size_mb(model)
    active_params = model.count_parameters(count_pruned=False)
    total_params = model.count_parameters(count_pruned=True)
    
    results = {
        'lambda': lambda_sparse,
        'test_accuracy': final_acc,
        'sparsity_percent': final_sparsity,
        'training_time_seconds': training_time,
        'inference_time_ms': inference_time * 1000,
        'model_size_mb': model_size,
        'active_parameters': active_params,
        'total_parameters': total_params,
        'parameter_reduction_pct': ((total_params - active_params) / total_params) * 100,
        'best_accuracy': best_accuracy,
        'train_losses': train_losses,
        'test_accuracies': test_accuracies
    }
    
    if logger:
        logger.log(f"Training complete: Acc={final_acc:.2f}%, Sparsity={final_sparsity:.1f}%, Time={training_time:.1f}s")
    
    return results, model


def evaluate_model(model, test_loader, device, return_time=False):
    """Evaluate model on test set"""
    model.eval()
    correct = 0
    total = 0
    
    start_time = time.time()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    inference_time = time.time() - start_time
    accuracy = 100. * correct / total
    
    if return_time:
        return accuracy, inference_time
    return accuracy


def run_experiment(lambda_values=None, epochs=20, hidden_dim=64):
    """
    Run full experiment with multiple lambda values
    
    Returns:
        dict: Results for each lambda value
        dict: Trained models
    """
    if lambda_values is None:
        lambda_values = [0.0001, 0.001, 0.01]
    
    logger = TrainingLogger()
    logger.log("=" * 60)
    logger.log("EDHAS - Self-Pruning Neural Network Experiment")
    logger.log("Author: Kudrat Anand (RA2311026010844)")
    logger.log("=" * 60)
    
    # Load dataset
    logger.log("Loading sklearn digits dataset...")
    X_train, y_train, X_test, y_test, input_dim, num_classes = load_digits_dataset()
    train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test, batch_size=32)
    logger.log(f"Dataset: {len(X_train)} train, {len(X_test)} test samples")
    
    results = {}
    models = {}
    
    # Train baseline model (no sparsity)
    logger.log("\n--- Training Baseline Model (no pruning) ---")
    baseline_model = GreenSparseNet(input_dim, hidden_dim, num_classes, num_hidden_layers=2)
    baseline_results, baseline_trained = train_model(
        baseline_model, train_loader, test_loader, 
        lambda_sparse=0, epochs=epochs, logger=logger
    )
    results['baseline'] = baseline_results
    models['baseline'] = baseline_trained
    
    # Train models with different lambda values
    for lambda_val in lambda_values:
        logger.log(f"\n--- Training with lambda={lambda_val} ---")
        model = GreenSparseNet(input_dim, hidden_dim, num_classes, num_hidden_layers=2)
        
        model_results, trained_model = train_model(
            model, train_loader, test_loader,
            lambda_sparse=lambda_val, epochs=epochs, logger=logger
        )
        
        # Compute cost savings
        savings = compute_cost_savings(
            baseline_results['training_time_seconds'],
            model_results['training_time_seconds'],
            baseline_results['inference_time_ms'] / 1000,
            model_results['inference_time_ms'] / 1000,
            baseline_results['model_size_mb'],
            model_results['model_size_mb']
        )
        model_results['cost_savings'] = savings
        
        results[f'lambda_{lambda_val}'] = model_results
        models[f'lambda_{lambda_val}'] = trained_model
    
    # Summary
    logger.log("\n" + "=" * 60)
    logger.log("EXPERIMENT SUMMARY")
    logger.log("=" * 60)
    
    for key, res in results.items():
        logger.log(f"{key}: Acc={res['test_accuracy']:.2f}%, Sparsity={res.get('sparsity_percent', 0):.1f}%, Time={res['training_time_seconds']:.1f}s")
    
    return results, models


if __name__ == '__main__':
    # Run full experiment
    results, models = run_experiment()
    
    # Save results
    from utils import save_results
    json_path, csv_path = save_results(results)
    print(f"\nResults saved to: {json_path} and {csv_path}")
