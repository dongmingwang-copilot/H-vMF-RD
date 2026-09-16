"""Regenerate the scientific figures used by the manuscript."""
from pathlib import Path
from plot_model_evidence import denoising_evidence, category_deltas, qualitative

def main():
    root = Path(__file__).resolve().parent.parent
    (root/'paper/figures').mkdir(parents=True, exist_ok=True)
    if not all((root/'research/model_outputs'/f'{d}_validation.npz').exists()
               for d in ['3cad', 'mvtec_ad2']):
        from analyze_model_outputs import main as analyze
        analyze()
    denoising_evidence()
    category_deltas()
    qualitative()
    print('Model-derived denoising, category, and localization figures generated.')

if __name__ == '__main__':
    main()
