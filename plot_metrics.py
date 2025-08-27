#!/usr/bin/env python3
"""
Plot training and validation metrics from a training session using metrics.csv file.
This script visualizes the progress of training and validation metrics over time.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Plot training metrics from CSV file')
    parser.add_argument('metrics_file', type=str, help='Path to metrics.csv file')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for plots')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for saving plots')
    parser.add_argument('--style', type=str, default='seaborn-v0_8-whitegrid', help='Matplotlib style')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set style
    plt.style.use(args.style)
    
    # Set up output directory
    metrics_path = Path(args.metrics_file)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = metrics_path.parent / 'plots'
    
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Saving plots to: {output_dir}")
    
    # Load metrics
    df = pd.read_csv(metrics_path)
    df = df.dropna(axis=1, how='all')  # Drop columns that are all NaN
    
    # Clean up the dataframe - remove empty rows and handle missing values
    if df.columns[0].startswith('Unnamed'):
        df = df.rename(columns={df.columns[0]: 'epoch'})
    df = df[df['epoch'].notna()]
    df = df.fillna(method='ffill')  # Forward fill NaN values
    
    # Convert epoch to int for better plotting
    df['epoch'] = df['epoch'].astype(int)
    
    # Get unique epochs
    epochs = sorted(df['epoch'].unique())
    
    # Create a summary dataframe with one row per epoch
    summary_df = pd.DataFrame(index=epochs)
    
    # Identify metric columns
    loss_cols = [col for col in df.columns if 'loss' in col.lower()]
    val_cols = [col for col in df.columns if 'val_' in col.lower()]
    
    # Extract training and validation metrics per epoch
    for epoch in epochs:
        epoch_data = df[df['epoch'] == epoch]
        
        # Add training loss
        for col in loss_cols:
            if 'step' not in col:  # Skip step losses
                summary_df.loc[epoch, col] = epoch_data[col].dropna().iloc[-1] if not epoch_data[col].dropna().empty else np.nan
        
        # Add validation metrics
        for col in val_cols:
            if 'step' not in col:  # Skip step metrics
                summary_df.loc[epoch, col] = epoch_data[col].dropna().iloc[-1] if not epoch_data[col].dropna().empty else np.nan
    
    # Create plots
    
    # 1. Loss plot
    fig, ax = plt.subplots(figsize=(10, 6))
    loss_metrics = [col for col in summary_df.columns if 'loss' in col.lower() and 'step' not in col]
    
    for col in loss_metrics:
        label = col.replace('_', ' ').title()
        ax.plot(summary_df.index, summary_df[col], marker='o', linestyle='-', label=label)

    # Add 1 - val_dice for comparison
    if 'val_dice' in summary_df.columns:
        ax.plot(summary_df.index, 1 - summary_df['val_dice'], marker='x', linestyle='--', label='1 - Val Dice')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss / (1 - Dice)')
    ax.set_title('Training Loss vs. Validation Loss vs. (1 - Val Dice)')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_plot.png', dpi=args.dpi)
    plt.close()
    
    # 2. Performance metrics plot (Dice, IoU)
    fig, ax = plt.subplots(figsize=(10, 6))
    performance_metrics = ['val_dice', 'val_iou']
    performance_metrics = [m for m in performance_metrics if m in summary_df.columns]
    
    for col in performance_metrics:
        label = col.replace('val_', '').replace('_', ' ').title()
        ax.plot(summary_df.index, summary_df[col], marker='o', linestyle='-', label=label)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('Validation Performance Metrics')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_metrics.png', dpi=args.dpi)
    plt.close()
    
    # 3. Airway-specific metrics
    airway_metrics = ['val_dlr', 'val_dbr', 'val_precision', 'val_leakage', 'val_amr']
    available_airway_metrics = [m for m in airway_metrics if m in summary_df.columns]
    
    if available_airway_metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for col in available_airway_metrics:
            label = col.replace('val_', '').replace('_', ' ').title()
            ax.plot(summary_df.index, summary_df[col], marker='o', linestyle='-', label=label)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title('Airway-Specific Metrics')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='best')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'airway_metrics.png', dpi=args.dpi)
        plt.close()
    
    # 4. Learning rate plot
    lr_cols = [col for col in summary_df.columns if 'lr' in col.lower()]
    
    if lr_cols:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for col in lr_cols:
            label = col.replace('_', ' ').title()
            ax.plot(summary_df.index, summary_df[col], marker='o', linestyle='-', label=label)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_yscale('log')  # Log scale for learning rate
        ax.legend(loc='best')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'learning_rate.png', dpi=args.dpi)
        plt.close()
    
    # 5. Combined performance plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Loss subplot
    for col in loss_metrics:
        label = col.replace('_', ' ').title()
        ax1.plot(summary_df.index, summary_df[col], marker='o', linestyle='-', label=label)

    # Add 1 - val_dice to the loss subplot for comparison
    if 'val_dice' in summary_df.columns:
        ax1.plot(summary_df.index, 1 - summary_df['val_dice'], marker='x', linestyle='--', label='1 - Val Dice')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss / (1 - Dice)')
    ax1.set_title('Loss Values vs. (1 - Val Dice)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='best')
    
    # Metrics subplot
    for col in performance_metrics:
        label = col.replace('val_', '').replace('_', ' ').title()
        ax2.plot(summary_df.index, summary_df[col], marker='o', linestyle='-', label=label)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.set_title('Performance Metrics')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'combined_performance.png', dpi=args.dpi)
    plt.close()
    
    # 6. Export summary CSV for easier analysis
    summary_df.to_csv(output_dir / 'metrics_summary.csv')
    
    print(f"Generated {len(os.listdir(output_dir))} plot files in {output_dir}")
    print("Done!")

if __name__ == "__main__":
    main()
