# HRM-MLX

MLX-Portierung des **Hierarchical Reasoning Model** (Wang et al., 2025) für Apple Silicon.

Dieses Repo ist ein Fork von [`kmkofficial/hrm-mlx`](https://github.com/kmkofficial/hrm-mlx) mit aufgeräumter Dokumentation. Die eigentliche MLX-Portierung stammt aus diesem Upstream; das Original-HRM ist [`sapientinc/HRM`](https://github.com/sapientinc/HRM) (PyTorch/CUDA).

## Was ist HRM?

HRM ist eine rekurrente Architektur mit zwei gekoppelten Modulen, die in unterschiedlichen Zeitskalen arbeiten – inspiriert von der hierarchischen Verarbeitung im Cortex:

- **H-Modul** (high-level): langsame, abstrakte Planung
- **L-Modul** (low-level): schnelle, detaillierte Berechnung

Beide laufen iterativ in einer **hierarchischen Konvergenz**-Schleife. Statt Chain-of-Thought im Token-Space zu generieren, reasoned HRM direkt im latenten Raum. Adaptive Computation Time (ACT) mit Q-Learning entscheidet, wann genug iteriert wurde.

Im Original-Paper schlägt HRM mit **27M Parametern** und **1.000 Trainingsbeispielen** (kein Pretraining, kein CoT) deutlich größere LLMs auf ARC-AGI, Sudoku-Extreme und Maze-Hard. Details: [arXiv:2506.21734](https://arxiv.org/abs/2506.21734).

## Warum MLX?

HRM ist klein, braucht kein Pretraining, kein BPTT (one-step gradient approximation) und ist damit **prädestiniert für Apple Silicon**. Ein vollständiger Trainings-Run passt auf ein MacBook mit 18 GB Unified Memory.

## Status dieses Forks

| Komponente | Status |
|---|---|
| HRM-Architektur (H/L-Module, ACT, Q-Learning) | ✅ vollständig portiert |
| AdamATan2 Optimizer (exakte Portierung) | ✅ |
| Dual-Optimizer (separate Embedding-LR) | ✅ |
| Cosine Schedule + Warmup | ✅ |
| Gradient Accumulation, Clipping, Auto-Resume | ✅ |
| Stablemax Cross-Entropy Loss | ✅ |
| Sudoku-Extreme Dataloader | ✅ |
| ARC-AGI Dataloader | ❌ noch zu portieren |
| Maze-Hard Dataloader | ❌ |
| FlashAttention → `mlx.fast.scaled_dot_product_attention` | ⏳ TODO |
| Reproduzierte Benchmark-Zahlen | ⏳ noch nicht verifiziert |
| Sparse Embedding (spezialisierter Optimizer) | ⚠️ vereinfacht (dense, zero-init) |

**Ehrlich:** Ob die MLX-Portierung Sapients Original-Performance auf Sudoku-Extreme/ARC-AGI exakt reproduziert, ist hier noch nicht gemessen. Das ist die nächste offene Aufgabe.

## Installation

Voraussetzungen: macOS auf Apple Silicon (M1 oder neuer), Python 3.10+, mindestens 16 GB Unified Memory empfohlen.

```bash
git clone <dieser-fork>
cd hrm-mlx
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Hinweis: `setup.py` listet aktuell noch PyTorch-Dependencies aus dem Upstream – die brauchst du nicht. Wird im nächsten Cleanup entfernt.

## Quickstart

### Smoke-Test (5 Minuten auf M3 Pro)

Mini-Modell, 50 Trainingsbeispiele, prüft nur ob die Pipeline läuft:

```bash
bash train_small.sh
```

Erwartung: Loss sinkt, kein NaN, Checkpoints landen in `checkpoints/`.

### Voller Sudoku-Extreme Run

Original-Architektur (17.8M Params, 2×2 cycles, 4+4 layers), 1.000 Beispiele:

```bash
# Zuerst: Daten besorgen
mkdir -p data/sudoku-extreme
# train.csv und test.csv aus dem offiziellen Sapient-Repo bzw. HuggingFace ziehen
# Format: source,question,answer,rating

bash train_sudoku.sh
```

Trainings-Dauer auf M3 Pro 18GB: erwartet mehrere Tage für volle 20.000 Epochen. Checkpoints alle 2.000 Steps, automatische Wiederaufnahme nach Unterbrechung.

### Evaluation

```bash
python evaluate.py --checkpoint checkpoints/best_model_step_XXXX.npz
```

## Architektur

```
HierarchicalReasoningModel
├── HierarchicalReasoningModel_Inner
│   ├── embed_tokens (CastedEmbedding)
│   ├── embed_pos    (learned, optional rope)
│   ├── H_level: HRMReasoningModule (4 Layer)
│   │   └── HRMTransformerBlock × N
│   │       ├── Attention (post-norm)
│   │       └── SwiGLU MLP (post-norm)
│   ├── L_level: HRMReasoningModule (4 Layer, gleiche Struktur)
│   ├── lm_head  (CastedLinear)
│   └── q_head   (CastedLinear, Init bei -5.0 für träges Halten)
└── ACT-Wrapper
    ├── halt_max_steps = 8 oder 16
    ├── halt_exploration_prob (Q-Learning ε)
    └── one-step gradient approximation via mx.stop_gradient
```

Die hierarchische Konvergenz läuft als verschachtelte Schleife: Für jeden H-Cycle laufen mehrere L-Cycles, am Ende ein Gradient-tragender Update-Schritt. Das spart Speicher gegenüber BPTT bei gleicher effektiver Tiefe.

## Konfiguration

Alle Hyperparameter über YAML in `config/`:

- `cfg_small.yaml` – Smoke-Test (256d, 1×1 cycles)
- `cfg_pretrain.yaml` – Default Full-Scale (512d, 2×2 cycles)
- `cfg_sudoku.yaml` – Sudoku mit Gradient Accumulation (effektive Batch 384)
- `cfg_sudoku_8gpu.yaml` – Original 8-GPU Hyperparameter (single-device adaptiert)

CLI-Argumente überschreiben YAML-Werte:

```bash
python train_yaml.py --config config/cfg_sudoku.yaml --batch_size 32
```

## Dateistruktur

```
hrm-mlx/
├── models/
│   ├── hrm/hrm_act_v1.py    # Hauptarchitektur, exakte Sapient-Portierung
│   ├── layers.py             # Attention, SwiGLU, RoPE, RMSNorm, CastedLinear
│   ├── losses.py             # Stablemax CE, Q-Halt/Q-Continue BCE
│   ├── common.py             # trunc_normal_init
│   └── sparse_embedding.py   # vereinfachte Variante
├── config/                   # YAML-Configs
├── mlx_adam_atan2_exact.py   # AdamATan2 (lucidrains-Port, PyTorch-genau)
├── dual_optimizer.py         # Separate LRs für Embeddings vs. Main
├── lr_scheduler.py           # Cosine + Warmup
├── config_loader.py          # YAML→Dataclass
├── load_official_sudoku.py   # Sudoku-Extreme CSV-Loader + Augmentierung
├── pretrain.py               # Trainer-Klasse + Legacy CLI
├── train_yaml.py             # Aktuelles Trainings-Entrypoint
└── evaluate.py
```

## Roadmap

**Kurzfristig (Reproduktion):**
- [ ] Sudoku-Extreme Daten beschaffen und Smoke-Test grünziehen
- [ ] Vollständiger Sudoku-Run, Vergleich mit Sapient-Baseline
- [ ] `mlx.fast.scaled_dot_product_attention` statt manueller Attention
- [ ] PyTorch-Dependencies aus `setup.py` entfernen

**Mittelfristig (Erweiterung):**
- [ ] ARC-AGI Dataloader portieren
- [ ] Maze-Hard Dataloader portieren
- [ ] 4-bit Quantisierung der Checkpoints (`mlx.core.quantize`)
- [ ] Sauberer Sparse-Embedding-Pfad mit eigenem Optimizer

**Langfristig (Forschung):**
- [ ] Test-Time Adaptation auf dem L-Modul (Hebbian fast weights / TTT-Loss), ohne H-Modul zu destabilisieren
- [ ] Adapter-Schnittstelle, um HRM als Reasoning-Co-Prozessor neben einem MLX-LLM zu nutzen
- [ ] Kombination mit V-JEPA-Latents als Weltmodell-Layer

## Credits

- **HRM-Paper & Original-Code:** Guan Wang, Jin Li, Yuhao Sun, Xing Chen, Changling Liu, Yue Wu, Meng Lu, Sen Song, Yasin Abbasi Yadkori (Sapient Intelligence) – [arXiv:2506.21734](https://arxiv.org/abs/2506.21734), [github.com/sapientinc/HRM](https://github.com/sapientinc/HRM)
- **AdamATan2:** Phil Wang (lucidrains) – [adam-atan2-pytorch](https://github.com/lucidrains/adam-atan2-pytorch)
- **MLX-Portierung Upstream:** [kmkofficial/hrm-mlx](https://github.com/kmkofficial/hrm-mlx)

## Lizenz

MIT – siehe `LICENSE`. Bei wissenschaftlicher Nutzung bitte das HRM-Originalpaper zitieren:

```bibtex
@misc{wang2025hierarchicalreasoningmodel,
  title  = {Hierarchical Reasoning Model},
  author = {Guan Wang and Jin Li and Yuhao Sun and Xing Chen and Changling Liu and Yue Wu and Meng Lu and Sen Song and Yasin Abbasi Yadkori},
  year   = {2025},
  eprint = {2506.21734},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url    = {https://arxiv.org/abs/2506.21734}
}
```
