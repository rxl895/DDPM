"""
Ablation Study: Noise Schedule Comparison

Compares different beta schedules (linear, cosine, quadratic, sigmoid, exponential)
and evaluates their impact on:
- Sample quality (FID score)
- Sampling time
- Training stability

Usage:
    python ablation_noise_schedules.py --checkpoint checkpoints/final_model.pt --num_samples 5000
"""

import torch
import numpy as np
import argparse
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

from ddpm.unet import SmallUNet
from ddpm.sample import p_sample_loop
from ddpm.ddim import ddim_sample_loop
from ddpm.forward import get_named_beta_schedule, compute_alphas
from ddpm.evaluation import calculate_fid
from ddpm.data import get_cifar10_dataloader


def get_beta_schedule_variants(num_timesteps=1000):
    """Get all beta schedule variants for comparison."""
    schedules = {
        'linear': get_named_beta_schedule('linear', num_timesteps),
        'cosine': get_named_beta_schedule('cosine', num_timesteps),
        'quadratic': get_named_beta_schedule('quadratic', num_timesteps),
        'sigmoid': get_named_beta_schedule('sigmoid', num_timesteps),
    }
    
    # Add exponential schedule
    beta_start = 1e-4
    beta_end = 0.02
    schedules['exponential'] = np.exp(
        np.linspace(np.log(beta_start), np.log(beta_end), num_timesteps)
    )
    
    return schedules


def visualize_schedules(schedules, save_path='ablation_results'):
    """Visualize the different beta schedules."""
    Path(save_path).mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot beta values
    ax = axes[0, 0]
    for name, betas in schedules.items():
        ax.plot(betas, label=name, linewidth=2)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Beta')
    ax.set_title('Beta Schedules')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot alpha values
    ax = axes[0, 1]
    for name, betas in schedules.items():
        alphas, _, _, _ = compute_alphas(betas)
        ax.plot(alphas.numpy(), label=name, linewidth=2)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Alpha')
    ax.set_title('Alpha Schedules (1 - Beta)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot cumulative alpha_bar
    ax = axes[1, 0]
    for name, betas in schedules.items():
        _, alphas_cumprod, _, _ = compute_alphas(betas)
        ax.plot(alphas_cumprod.numpy(), label=name, linewidth=2)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Alpha_bar (Cumulative Product)')
    ax.set_title('Signal Retention over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot SNR (Signal-to-Noise Ratio)
    ax = axes[1, 1]
    for name, betas in schedules.items():
        _, alphas_cumprod, _, _ = compute_alphas(betas)
        snr = alphas_cumprod / (1 - alphas_cumprod)
        ax.plot(np.log10(snr.numpy()), label=name, linewidth=2)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('log10(SNR)')
    ax.set_title('Signal-to-Noise Ratio (Log Scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/schedule_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved schedule visualization to {save_path}/schedule_comparison.png")


def evaluate_schedule(model, schedule_name, betas, device, num_samples=5000, 
                      method='ddim', ddim_steps=100, batch_size=64):
    """Evaluate a single schedule."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {schedule_name.upper()}")
    print(f"{'='*60}")
    
    model.eval()
    all_samples = []
    
    # Timing
    start_time = time.time()
    
    with torch.no_grad():
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for i in tqdm(range(num_batches), desc=f"Sampling ({schedule_name})"):
            current_batch_size = min(batch_size, num_samples - i * batch_size)
            
            if method == 'ddpm':
                samples = p_sample_loop(
                    model,
                    (current_batch_size, 3, 32, 32),
                    betas,
                    device=device
                )
            else:  # ddim
                samples = ddim_sample_loop(
                    model,
                    (current_batch_size, 3, 32, 32),
                    betas,
                    ddim_steps=ddim_steps,
                    eta=0.0,
                    device=device
                )
            
            all_samples.append(samples.cpu())
    
    sampling_time = time.time() - start_time
    all_samples = torch.cat(all_samples, dim=0)[:num_samples]
    
    # Calculate FID
    print(f"Calculating FID score for {schedule_name}...")
    dataloader = get_cifar10_dataloader(batch_size=batch_size, train=True)
    fid_score = calculate_fid(all_samples, dataloader, device=device, num_samples=num_samples)
    
    results = {
        'schedule': schedule_name,
        'fid': float(fid_score),
        'sampling_time': float(sampling_time),
        'samples_per_second': float(num_samples / sampling_time),
        'num_samples': num_samples,
        'method': method,
        'ddim_steps': ddim_steps if method == 'ddim' else 1000
    }
    
    print(f"\nResults for {schedule_name}:")
    print(f"  FID Score: {fid_score:.2f}")
    print(f"  Sampling Time: {sampling_time:.2f}s")
    print(f"  Samples/sec: {num_samples/sampling_time:.2f}")
    
    return results, all_samples


def plot_ablation_results(results_list, save_path='ablation_results'):
    """Create comprehensive visualization of ablation results."""
    Path(save_path).mkdir(exist_ok=True)
    
    # Extract data
    schedules = [r['schedule'] for r in results_list]
    fids = [r['fid'] for r in results_list]
    times = [r['sampling_time'] for r in results_list]
    samples_per_sec = [r['samples_per_second'] for r in results_list]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: FID Score comparison
    ax = axes[0]
    bars = ax.bar(schedules, fids, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E'])
    ax.set_xlabel('Noise Schedule', fontsize=12, fontweight='bold')
    ax.set_ylabel('FID Score (lower is better)', fontsize=12, fontweight='bold')
    ax.set_title('Sample Quality Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, fid in zip(bars, fids):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{fid:.2f}',
                ha='center', va='bottom', fontweight='bold')
    
    # Highlight best
    best_idx = np.argmin(fids)
    bars[best_idx].set_color('#2ECC71')
    bars[best_idx].set_edgecolor('black')
    bars[best_idx].set_linewidth(2)
    
    # Plot 2: Sampling Time comparison
    ax = axes[1]
    bars = ax.bar(schedules, times, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E'])
    ax.set_xlabel('Noise Schedule', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sampling Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Sampling Speed Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, t in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{t:.2f}s',
                ha='center', va='bottom', fontweight='bold')
    
    # Highlight fastest
    fastest_idx = np.argmin(times)
    bars[fastest_idx].set_color('#2ECC71')
    bars[fastest_idx].set_edgecolor('black')
    bars[fastest_idx].set_linewidth(2)
    
    # Plot 3: FID vs Time scatter
    ax = axes[2]
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    for i, (schedule, fid, t) in enumerate(zip(schedules, fids, times)):
        ax.scatter(t, fid, s=200, c=colors[i], label=schedule, alpha=0.7, edgecolors='black', linewidth=2)
    
    ax.set_xlabel('Sampling Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('FID Score', fontsize=12, fontweight='bold')
    ax.set_title('Quality vs Speed Trade-off', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Annotate best quality and fastest
    best_quality_idx = np.argmin(fids)
    ax.annotate('Best Quality', 
                xy=(times[best_quality_idx], fids[best_quality_idx]),
                xytext=(10, 10), textcoords='offset points',
                fontweight='bold', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', lw=2))
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/ablation_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved results visualization to {save_path}/ablation_results.png")


def save_sample_grid(samples_dict, save_path='ablation_results', n_samples=8):
    """Save a grid comparing samples from different schedules."""
    Path(save_path).mkdir(exist_ok=True)
    
    num_schedules = len(samples_dict)
    fig, axes = plt.subplots(num_schedules, n_samples, figsize=(n_samples*2, num_schedules*2))
    
    for i, (schedule_name, samples) in enumerate(samples_dict.items()):
        for j in range(n_samples):
            ax = axes[i, j] if num_schedules > 1 else axes[j]
            
            # Convert from [-1, 1] to [0, 1]
            img = (samples[j].permute(1, 2, 0).numpy() + 1) / 2
            img = np.clip(img, 0, 1)
            
            ax.imshow(img)
            ax.axis('off')
            
            if j == 0:
                ax.set_ylabel(schedule_name.upper(), fontsize=12, fontweight='bold', rotation=0, 
                            labelpad=40, va='center')
    
    plt.suptitle('Sample Quality Comparison Across Noise Schedules', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{save_path}/sample_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved sample comparison to {save_path}/sample_comparison.png")


def main():
    parser = argparse.ArgumentParser(description='Noise Schedule Ablation Study')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/final_model.pt',
                       help='Path to trained model checkpoint')
    parser.add_argument('--num_samples', type=int, default=5000,
                       help='Number of samples for FID calculation')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for sampling')
    parser.add_argument('--method', type=str, default='ddim', choices=['ddpm', 'ddim'],
                       help='Sampling method')
    parser.add_argument('--ddim_steps', type=int, default=100,
                       help='Number of DDIM steps (if using DDIM)')
    parser.add_argument('--save_path', type=str, default='ablation_results',
                       help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    
    print("="*60)
    print("ABLATION STUDY: NOISE SCHEDULE COMPARISON")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Sampling method: {args.method.upper()}")
    if args.method == 'ddim':
        print(f"DDIM steps: {args.ddim_steps}")
    print(f"Device: {args.device}")
    print("="*60)
    
    # Load model
    print("\nLoading model...")
    device = torch.device(args.device)
    model = SmallUNet(in_channels=3, base_ch=64).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Try to load the checkpoint - handle both old and new formats
    try:
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Model loaded successfully!")
    except RuntimeError as e:
        print(f"Warning: Could not load checkpoint with current architecture.")
        print(f"Error: {e}")
        print("\nTrying with base_ch=32 (old architecture)...")
        # Try smaller model that matches old checkpoint
        model = SmallUNet(in_channels=3, base_ch=32).to(device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Model loaded successfully with base_ch=32!")
    
    model.eval()
    
    # Get all schedules
    schedules = get_beta_schedule_variants(num_timesteps=1000)
    
    # Visualize schedules
    print("\nVisualizing noise schedules...")
    visualize_schedules(schedules, save_path=args.save_path)
    
    # Evaluate each schedule
    all_results = []
    all_samples = {}
    
    for schedule_name, betas in schedules.items():
        results, samples = evaluate_schedule(
            model, 
            schedule_name, 
            betas, 
            device,
            num_samples=args.num_samples,
            method=args.method,
            ddim_steps=args.ddim_steps,
            batch_size=args.batch_size
        )
        all_results.append(results)
        all_samples[schedule_name] = samples[:8]  # Keep first 8 for visualization
    
    # Save results
    results_file = Path(args.save_path) / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved detailed results to {results_file}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_ablation_results(all_results, save_path=args.save_path)
    save_sample_grid(all_samples, save_path=args.save_path)
    
    # Print summary
    print("\n" + "="*60)
    print("ABLATION STUDY SUMMARY")
    print("="*60)
    
    best_fid_idx = np.argmin([r['fid'] for r in all_results])
    fastest_idx = np.argmin([r['sampling_time'] for r in all_results])
    
    print("\n📊 FID Scores (lower is better):")
    for r in sorted(all_results, key=lambda x: x['fid']):
        marker = "🏆" if r == all_results[best_fid_idx] else "  "
        print(f"{marker} {r['schedule']:12s}: {r['fid']:6.2f}")
    
    print("\n⚡ Sampling Speed:")
    for r in sorted(all_results, key=lambda x: x['sampling_time']):
        marker = "🏆" if r == all_results[fastest_idx] else "  "
        print(f"{marker} {r['schedule']:12s}: {r['sampling_time']:6.2f}s ({r['samples_per_second']:.1f} samples/s)")
    
    print("\n💡 Key Findings:")
    best_schedule = all_results[best_fid_idx]['schedule']
    print(f"- Best quality: {best_schedule} (FID: {all_results[best_fid_idx]['fid']:.2f})")
    print(f"- Fastest: {all_results[fastest_idx]['schedule']} ({all_results[fastest_idx]['sampling_time']:.2f}s)")
    
    fid_range = max(r['fid'] for r in all_results) - min(r['fid'] for r in all_results)
    time_range = max(r['sampling_time'] for r in all_results) - min(r['sampling_time'] for r in all_results)
    print(f"- FID variance: {fid_range:.2f} points")
    print(f"- Time variance: {time_range:.2f}s")
    
    print("\n" + "="*60)
    print(f"All results saved to: {args.save_path}/")
    print("="*60)


if __name__ == "__main__":
    main()
