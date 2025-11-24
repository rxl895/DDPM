# Contributing to DDPM

Thank you for your interest in contributing! This is an educational project demonstrating diffusion models.

## How to Contribute

### Reporting Issues
- Check existing issues first
- Provide clear description and reproduction steps
- Include environment details (Python version, PyTorch version, etc.)

### Pull Requests
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and ensure code works
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Style
- Follow PEP 8 guidelines
- Use type hints where applicable
- Add docstrings to functions and classes
- Keep functions focused and modular

### Areas for Contribution
- Additional sampling methods (DPM-Solver, PNDM, etc.)
- More datasets (ImageNet, custom datasets)
- Advanced architectures (Transformer UNet, etc.)
- Training optimizations (mixed precision, distributed training)
- Evaluation metrics (IS, LPIPS, etc.)
- Documentation improvements
- Visualization tools

### Testing
- Test your changes thoroughly
- Ensure compatibility with both CPU and GPU
- Verify memory efficiency for large batches

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/DDPM.git
cd DDPM

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run a quick test
python examples/demo_sampling.py --checkpoint checkpoints/final_model.pt --num_samples 4
```

## Questions?

Open an issue for discussion or reach out to the maintainers.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
