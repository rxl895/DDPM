"""Evaluate DDPM and DDIM with FID scores and timing comparison.

Creates a comprehensive evaluation table comparing:
- DDPM vs DDIM
- Sampling speed
- Image quality (FID)
"""
import torch
import time
import argparse
from pathlib import Path

from ddpm.unet import SmallUNet
from ddpm.forward import get_named_beta_schedule
from ddpm.sample import p_sample_loop
from ddpm.ddim import ddim_sample_loop
from ddpm.evaluation import evaluate_fid_from_loader
from ddpm.data import get_cifar10_dataloader
from ddpm.train import load_checkpoint


def time_sampling(sample_fn, model, shape, betas, device, num_runs=3, **kwargs):
    """Time how long sampling takes."""
    times = []
    
    # Check if function accepts verbose parameter
    import inspect
    sig = inspect.signature(sample_fn)
    has_verbose = 'verbose' in sig.parameters
    
    for _ in range(num_runs):
        start = time.time()
        with torch.no_grad():
            if has_verbose:
                _ = sample_fn(model, shape, betas, device=device, verbose=False, **kwargs)
            else:
                _ = sample_fn(model, shape, betas, device=device, **kwargs)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        times.append(time.time() - start)
    
    return sum(times) / len(times)


def main():
    parser = argparse.ArgumentParser(description="Evaluate DDPM/DDIM")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples for FID")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for data loading")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--output", type=str, default="evaluation_results.txt", help="Output file")
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = SmallUNet(in_channels=3, base_ch=64, time_emb_dim=128)
    info = load_checkpoint(model, None, args.checkpoint, device)
    print(f"Loaded model from epoch {info['epoch']} with loss {info['loss']:.6f}")
    
    model = model.to(device)
    model.eval()
    
    # Load data
    print("Loading CIFAR-10 dataset...")
    dataloader = get_cifar10_dataloader(batch_size=args.batch_size)
    
    # Setup
    betas = get_named_beta_schedule("linear", 1000)
    shape = (64, 3, 32, 32)  # For timing
    
    results = []
    results.append("=" * 80)
    results.append("DDPM/DDIM Evaluation Results")
    results.append("=" * 80)
    results.append("")
    
    # Evaluate DDPM
    print("\n" + "="*80)
    print("Evaluating DDPM (1000 steps)...")
    print("="*80)
    
    ddpm_time = time_sampling(p_sample_loop, model, shape, betas, device)
    print(f"DDPM sampling time: {ddpm_time:.2f}s for 64 images ({ddpm_time/64:.3f}s per image)")
    
    print(f"Calculating FID for DDPM with {args.num_samples} samples...")
    ddpm_fid = evaluate_fid_from_loader(
        model, dataloader, args.num_samples,
        p_sample_loop, device, betas=betas
    )
    print(f"DDPM FID: {ddpm_fid:.2f}")
    
    # Evaluate DDIM with different steps
    ddim_configs = [
        {"steps": 50, "eta": 0.0},
        {"steps": 100, "eta": 0.0},
        {"steps": 250, "eta": 0.0},
    ]
    
    ddim_results = []
    for config in ddim_configs:
        steps = config["steps"]
        eta = config["eta"]
        
        print("\n" + "="*80)
        print(f"Evaluating DDIM ({steps} steps, eta={eta})...")
        print("="*80)
        
        ddim_time = time_sampling(
            ddim_sample_loop, model, shape, betas, device,
            num_steps=steps, eta=eta
        )
        print(f"DDIM sampling time: {ddim_time:.2f}s for 64 images ({ddim_time/64:.3f}s per image)")
        
        print(f"Calculating FID for DDIM with {args.num_samples} samples...")
        ddim_fid = evaluate_fid_from_loader(
            model, dataloader, args.num_samples,
            ddim_sample_loop, device, betas=betas, num_steps=steps, eta=eta
        )
        print(f"DDIM FID: {ddim_fid:.2f}")
        
        ddim_results.append({
            "steps": steps,
            "eta": eta,
            "time": ddim_time,
            "fid": ddim_fid,
            "speedup": ddpm_time / ddim_time
        })
    
    # Format results table
    results.append("Quantitative Evaluation Results")
    results.append("-" * 80)
    results.append("")
    results.append(f"Model: {args.checkpoint}")
    results.append(f"Evaluation samples: {args.num_samples}")
    results.append(f"Device: {device}")
    results.append("")
    results.append("Sampling Performance Comparison:")
    results.append("")
    results.append(f"{'Method':<15} {'Steps':<10} {'Time (64 imgs)':<20} {'Per Image':<15} {'FID Score':<12} {'Speedup':<10}")
    results.append("-" * 80)
    
    # DDPM row
    ddpm_time_str = f"{ddpm_time:.2f}s"
    ddpm_per_img = f"{ddpm_time/64:.3f}s"
    ddpm_fid_str = f"{ddpm_fid:.2f}"
    results.append(f"{'DDPM':<15} {1000:<10} {ddpm_time_str:<20} {ddpm_per_img:<15} {ddpm_fid_str:<12} {'1.00x':<10}")
    
    # DDIM rows
    for dr in ddim_results:
        method = f"DDIM (η={dr['eta']})"
        time_str = f"{dr['time']:.2f}s"
        per_img = f"{dr['time']/64:.3f}s"
        fid_str = f"{dr['fid']:.2f}"
        speedup_str = f"{dr['speedup']:.2f}x"
        results.append(f"{method:<15} {dr['steps']:<10} {time_str:<20} {per_img:<15} {fid_str:<12} {speedup_str:<10}")
    
    results.append("")
    results.append("=" * 80)
    results.append("Key Findings:")
    results.append("-" * 80)
    
    best_ddim = min(ddim_results, key=lambda x: x['fid'])
    fastest_ddim = max(ddim_results, key=lambda x: x['speedup'])
    
    results.append(f"• Best FID: DDIM {best_ddim['steps']} steps (FID={best_ddim['fid']:.2f})")
    results.append(f"• Fastest: DDIM {fastest_ddim['steps']} steps ({fastest_ddim['speedup']:.1f}x speedup)")
    results.append(f"• Quality improvement: {((ddpm_fid - best_ddim['fid'])/ddpm_fid*100):.1f}% better FID with DDIM")
    results.append("")
    results.append("=" * 80)
    
    # Print and save results
    output_text = "\n".join(results)
    print("\n" + output_text)
    
    with open(args.output, 'w') as f:
        f.write(output_text)
    
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
