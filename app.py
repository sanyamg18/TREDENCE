"""
Flask backend for Edhas Dashboard
Author: Kudrat Anand (RA2311026010844)
"""

from flask import Flask, render_template, jsonify, request, send_file
import os
import json
import torch
import threading
from datetime import datetime

from model import GreenSparseNet
from train import run_experiment, evaluate_model
from utils import (get_device, load_digits_dataset, create_dataloaders, 
                   save_results, get_model_size_mb, TrainingLogger, compute_cost_savings)

app = Flask(__name__)

# Global state
training_status = {
    'is_training': False,
    'progress': 0,
    'current_lambda': None,
    'logs': [],
    'results': {},
    'models': {}
}

latest_results = {}
latest_models = {}

@app.route('/')
def home():
    """Home page with project overview"""
    return render_template('index.html')

@app.route('/train')
def train_page():
    """Training page"""
    return render_template('train.html')

@app.route('/results')
def results_page():
    """Results page"""
    return render_template('results.html')

@app.route('/visualizations')
def visualizations_page():
    """Visualizations page"""
    return render_template('visualizations.html')

@app.route('/3d')
def visualization_3d():
    """3D Interactive Neural Network Visualization"""
    return render_template('index_3d.html')


# API Endpoints
@app.route('/api/project-info')
def project_info():
    """Get project information"""
    return jsonify({
        'title': 'Edhas – Self-Pruning Neural Network',
        'subtitle': 'Sustainable AI Deployment',
        'author': 'Kudrat Anand',
        'registration': 'RA2311026010844',
        'institution': 'Tredence AI Engineering Internship Case Study',
        'description': 'A self-pruning neural network that reduces model size during training using learnable gate parameters, enabling sustainable AI deployment.',
        'features': [
            'Dynamic weight pruning during training',
            'CPU-friendly lightweight implementation',
            'Cost and CO2 savings estimation',
            'Interactive training dashboard',
            'Real-time visualization'
        ],
        'sdg_alignment': [
            'SDG 9: Industry, Innovation and Infrastructure',
            'SDG 12: Responsible Consumption and Production', 
            'SDG 13: Climate Action'
        ]
    })

@app.route('/api/start-training', methods=['POST'])
def start_training():
    """Start training with selected lambda values"""
    global training_status
    
    if training_status['is_training']:
        return jsonify({'status': 'error', 'message': 'Training already in progress'}), 400
    
    data = request.json or {}
    lambda_values = data.get('lambda_values', [0.0001, 0.001, 0.01])
    epochs = data.get('epochs', 20)
    
    # Reset status
    training_status['is_training'] = True
    training_status['progress'] = 0
    training_status['logs'] = []
    training_status['results'] = {}
    training_status['current_lambda'] = 'starting'
    
    # Start training in background thread
    thread = threading.Thread(
        target=training_worker, 
        args=(lambda_values, epochs)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'status': 'success', 
        'message': 'Training started',
        'lambda_values': lambda_values
    })

def training_worker(lambda_values, epochs):
    """Background worker for training"""
    global training_status, latest_results, latest_models
    
    logger = TrainingLogger()
    
    def log_callback(message):
        training_status['logs'].append(message)
        # Keep only last 100 logs
        if len(training_status['logs']) > 100:
            training_status['logs'] = training_status['logs'][-100:]
    
    # Monkey-patch logger to update status
    original_log = logger.log
    def patched_log(message):
        result = original_log(message)
        log_callback(result)
        return result
    logger.log = patched_log
    
    try:
        # Load dataset once
        logger.log("Loading dataset...")
        X_train, y_train, X_test, y_test, input_dim, num_classes = load_digits_dataset()
        train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test)
        
        results = {}
        models = {}
        
        # Baseline
        logger.log("Training baseline model...")
        training_status['current_lambda'] = 'baseline'
        baseline_model = GreenSparseNet(input_dim, 64, num_classes, num_hidden_layers=2)
        
        from train import train_model
        baseline_results, baseline_trained = train_model(
            baseline_model, train_loader, test_loader,
            lambda_sparse=0, epochs=epochs, logger=logger
        )
        results['baseline'] = baseline_results
        models['baseline'] = baseline_trained
        training_status['progress'] = 25
        
        # Train with each lambda
        progress_step = 75 / len(lambda_values)
        
        for i, lambda_val in enumerate(lambda_values):
            logger.log(f"Training with lambda={lambda_val}...")
            training_status['current_lambda'] = f'lambda_{lambda_val}'
            
            model = GreenSparseNet(input_dim, 64, num_classes, num_hidden_layers=2)
            model_results, trained_model = train_model(
                model, train_loader, test_loader,
                lambda_sparse=lambda_val, epochs=epochs, logger=logger
            )
            
            # Compute savings
            savings = compute_cost_savings(
                baseline_results['training_time_seconds'],
                model_results['training_time_seconds'],
                baseline_results['inference_time_ms'] / 1000,
                model_results['inference_time_ms'] / 1000,
                baseline_results['model_size_mb'],
                model_results['model_size_mb']
            )
            model_results['cost_savings'] = savings
            
            key = f'lambda_{lambda_val}'
            results[key] = model_results
            models[key] = trained_model
            
            training_status['progress'] = 25 + (i + 1) * progress_step
        
        # Save results
        save_results(results)
        
        # Save best model
        best_key = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
        model_path = os.path.join('outputs', 'best_model.pth')
        os.makedirs('outputs', exist_ok=True)
        torch.save(models[best_key].state_dict(), model_path)
        logger.log(f"Best model saved: {best_key}")
        
        latest_results = results
        latest_models = models
        training_status['results'] = results
        training_status['progress'] = 100
        
    except Exception as e:
        logger.log(f"Error during training: {str(e)}")
        import traceback
        logger.log(traceback.format_exc())
    finally:
        training_status['is_training'] = False
        training_status['current_lambda'] = 'complete'

@app.route('/api/training-status')
def get_training_status():
    """Get current training status"""
    return jsonify(training_status)

@app.route('/api/results')
def get_results():
    """Get latest results"""
    return jsonify(latest_results)

@app.route('/api/export-csv')
def export_csv():
    """Export results to CSV"""
    if not latest_results:
        return jsonify({'error': 'No results available'}), 400
    
    import pandas as pd
    
    # Prepare data
    data = []
    for key, res in latest_results.items():
        row = {'configuration': key}
        row.update({k: v for k, v in res.items() if k not in ['train_losses', 'test_accuracies', 'cost_savings']})
        if 'cost_savings' in res:
            for k, v in res['cost_savings'].items():
                row[f'savings_{k}'] = v
        data.append(row)
    
    df = pd.DataFrame(data)
    csv_path = os.path.join('outputs', 'export_results.csv')
    os.makedirs('outputs', exist_ok=True)
    df.to_csv(csv_path, index=False)
    
    return send_file(csv_path, as_attachment=True, download_name='edhas_results.csv')

@app.route('/api/gate-distribution')
def gate_distribution():
    """Get gate value distribution for visualization"""
    if not latest_models:
        return jsonify({'error': 'No models available'}), 400
    
    distributions = {}
    
    for key, model in latest_models.items():
        all_gates = []
        for layer in model.layers + [model.output_layer]:
            gates = layer.get_gate_values().detach().numpy().flatten()
            all_gates.extend(gates.tolist())
        distributions[key] = all_gates
    
    return jsonify(distributions)


if __name__ == '__main__':
    # Ensure output directory exists
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    print("=" * 60)
    print("EDHAS - Self-Pruning Neural Network Dashboard")
    print("Author: Kudrat Anand (RA2311026010844)")
    print("=" * 60)
    print("Open http://localhost:5000 in your browser")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
