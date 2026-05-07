"""
Utility functions for Edhas - Self-Pruning Neural Network
Author: Kudrat Anand (RA2311026010844)
"""

import torch
import numpy as np
import time
import json
import os
from datetime import datetime


def get_device():
    """Get the best available device (CPU for compatibility)"""
    return torch.device('cpu')


def load_digits_dataset():
    """
    Load sklearn digits dataset - lightweight alternative to MNIST
    Perfect for fast training on CPU
    """
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # Load digits dataset (8x8 images, 10 classes)
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)
    
    return X_train, y_train, X_test, y_test, 64, 10


def create_dataloaders(X_train, y_train, X_test, y_test, batch_size=32):
    """Create PyTorch DataLoaders"""
    from torch.utils.data import TensorDataset, DataLoader
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def compute_cost_savings(training_time_baseline, training_time_optimized, 
                         inference_time_baseline, inference_time_optimized,
                         model_size_baseline, model_size_optimized):
    """
    Compute cost and CO2 savings from model optimization
    """
    # Assumptions for cost calculation
    GPU_HOUR_COST = 2.50  # USD per hour for cloud GPU
    CO2_PER_KWH = 0.5  # kg CO2 per kWh
    GPU_POWER_KW = 0.3  # 300W GPU
    
    # Training cost savings
    train_hours_saved = (training_time_baseline - training_time_optimized) / 3600
    train_cost_saved = train_hours_saved * GPU_HOUR_COST
    train_co2_saved = train_hours_saved * GPU_POWER_KW * CO2_PER_KWH
    
    # Inference cost savings (per 1M predictions)
    inference_ratio = inference_time_baseline / max(inference_time_optimized, 0.001)
    predictions_per_hour = 3600 / max(inference_time_optimized, 0.001)
    
    # Model size savings
    size_reduction_pct = ((model_size_baseline - model_size_optimized) / model_size_baseline) * 100
    
    savings = {
        'training_time_saved_seconds': max(0, training_time_baseline - training_time_optimized),
        'training_hours_saved': max(0, train_hours_saved),
        'training_cost_saved_usd': max(0, train_cost_saved),
        'co2_saved_kg': max(0, train_co2_saved),
        'inference_speedup': inference_ratio,
        'model_size_reduction_pct': size_reduction_pct,
        'gpu_hours_saved': max(0, train_hours_saved),
        'cloud_cost_saved_monthly': max(0, train_cost_saved * 30),  # Assuming daily training
    }
    
    return savings


def save_results(results, output_dir='outputs'):
    """Save training results to JSON and CSV"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save as JSON
    json_path = os.path.join(output_dir, f'results_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save as CSV (flattened)
    import pandas as pd
    csv_data = []
    for lambda_val, data in results.items():
        row = {'lambda': lambda_val}
        row.update(data)
        csv_data.append(row)
    
    df = pd.DataFrame(csv_data)
    csv_path = os.path.join(output_dir, f'results_{timestamp}.csv')
    df.to_csv(csv_path, index=False)
    
    return json_path, csv_path


def get_model_size_mb(model):
    """Get model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


class TrainingLogger:
    """Simple training logger"""
    def __init__(self):
        self.logs = []
        
    def log(self, message):
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        print(log_entry)
        return log_entry
    
    def get_logs(self):
        return self.logs
    
    def clear(self):
        self.logs = []
