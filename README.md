# INEST-IRL: INtrinsic Exploration via SubTask Inverse Reinforcement Learning

[![License](https://img.shields.io/badge/license-Academic-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/)

**INEST-IRL** is a framework for learning long-horizon robotic manipulation tasks from visual demonstrations using Inverse Reinforcement Learning (IRL) combined with procedural decomposition and intrinsic reward mechanisms.

---

## 📖 Overview

This thesis addresses the challenge of learning complex manipulation tasks from visual demonstrations in the **X-Magical** environment. The specific task is "Sweep to top in order": moving three colored blocks (Red, Blue, Yellow) to a green target area at the top of the environment in the correct sequential order.


Traditional Reinforcement Learning (RL) approaches face limitations:
- **Complex reward engineering**: Designing reward functions is difficult
- **Inefficient exploration**: High-dimensional spaces require extensive sampling
- **Sparse rewards**: Long-horizon tasks provide limited learning signals

INEST-IRL addresses these challenges through:
1. Learning reward functions from demonstration videos using **ResNet-18 encoder trained with TCC loss**
2. **Procedural decomposition** into subtasks (tested with 1, 3, and 6 subtasks)
3. **Novelty-based intrinsic rewards** computed from embedding distances to encourage exploration

---

## 🎯 Methodology

### Two-Phase Training Process

#### Phase 1: Representation Learning
- **Encoder**: ResNet-18 architecture
- **Training**: Temporal Cycle-Consistency (TCC) loss on demonstration videos
- **Dataset**: 1000 demonstration videos (800 training, 200 validation)
  - 4-5 seconds each at 10 fps (40 frames per video)
  - Gripper embodiment performing successful task completions
  - Both allocentric and egocentric viewpoints
- **Output**: Frozen encoder producing embeddings, with stored subtask embedding representations

#### Phase 2: Policy Learning
- **Algorithm**: Soft Actor-Critic (SAC)
- **Encoder**: Frozen (weights from Phase 1)
- **Reward Computation**: Euclidean distance in embedding space to stored subtask embeddings

### Procedural Learning Implementation

The task is decomposed into **N subtasks** (where N = 1, 3, or 6 in experiments):
- Each subtask corresponds to a phase of the manipulation task
- Subtask progression tracked by an index
- **Reward structure**:
  ```
  r_total = r(d) + s × c_subtask
  ```
  where `s` is the subtask index and `c_subtask` is a constant bonus

- **Subtask completion detection**:
  - Threshold-based mechanism (different thresholds per subtask)
  - Counter-based system requiring reward above threshold for consecutive steps
  - Prevents premature transitions from noisy signals

- **Final reward** (all subtasks complete):
  ```
  r_final = s × c_subtask
  ```

### Intrinsic Reward Implementation

**Novelty-based exploration** in embedding space:
- Separate memory buffer for each subtask (capacity: 5,000 embeddings)
- FIFO policy when buffer is full

**Intrinsic reward calculation**:
```
d_B = ||φ(s_i) - B||²
r_intrinsic(s_i) = ||φ(s_i) - min(d_B)||²
```
where B is the memory buffer and φ(s_i) is the current embedding

**Total reward**:
```
r_total = r_distance + α_intrinsic × r_intrinsic
```

**Adaptive scaling**:
- Normal conditions: α_intrinsic = 0.2
- After subtask transitions (for n steps): α_intrinsic = 0.4
- Encourages aggressive exploration when entering new subtask phases

### Coverage Analysis

Two complementary tracking mechanisms:
1. **Similarity-preserving grid mapping**: PCA projection to 2D grid (100×100)
2. **Hash-based unique state counting**: Embeddings rounded to 3 decimals and hashed

---

## 🧪 Experimental Setup

### Environment: X-Magical
- **Task**: "Sweep to top in order"
- **Objects**: Three colored blocks (Red, Blue, Yellow)
- **Goal**: Move blocks to green target area in correct order (Red → Blue → Yellow)
- **Embodiment**: Gripper (grasp and release capability)
- **Observations**: RGB images 128×128 pixels
- **Viewpoints**: Both allocentric and egocentric tested
- **Action space**: 3D continuous (position control + grasping)

### Modifications to Standard X-Magical:
- Enhanced observation space (depth, block colors, positions, flags)
- Custom dense reward function based on embedding distances
- Order-dependent success criterion

### Experimental Comparisons

**Baseline Models Evaluated**:
1. **XIRL** (Cross-Embodiment IRL with TCC)
2. **HOLD-R** (Regression-based functional distance learning)
3. **HOLD-C** (Contrastive learning approach)
4. **CLIP-based models** (Vision-language representations)

**Ablation Studies**:
1. **Number of subtasks**: 1-Subtask vs 3-Subtasks vs 6-Subtasks
2. **Visual encoders**: ResNet-18, ResNet-50, Vision Transformer (ViT)
3. **Intrinsic reward**: With vs without novelty-based exploration

---

## 📊 Key Results

### Findings:
- **3-Subtask model** achieves highest success rate for placing all blocks correctly
- **Embedding-based intrinsic reward** significantly improves:
  - Average evaluation scores
  - Consistency (lower variance)
  - State space coverage during training
- **Adaptive scaling** (α = 0.4 after transitions) enhances exploration
- **ResNet-18 with TCC** provides effective representations for manipulation

### Success Metrics:
- End-of-episode subtask completion rates
- Success rates for placing 1, 2, and 3 blocks correctly
- Evaluation scores over 150 episodes
- Embedding space coverage analysis

---

## 📚 Related Work

INEST-IRL builds upon:
- **XIRL** (Zakka et al., 2021): Cross-embodiment IRL with temporal cycle consistency
- **HOLD** (Alakuijala et al., 2022): Learning from human demonstration videos
- **TCC** (Dwibedi et al., 2019): Temporal cycle-consistency for alignment
- **Active Pretraining** (Liu & Abbeel, 2021): Intrinsic motivation for exploration

---

## 📖 Documentation

This repository contains the LaTeX thesis documenting the complete INEST-IRL framework:

### Thesis Structure:
1. **Introduction**: Problem statement, motivation, and contributions
2. **Background**: 
   - Reinforcement Learning (MDPs, value/policy methods, SAC)
   - Imitation Learning (Behavioral Cloning, Adversarial IL, IRL)
   - Representation Learning (CNNs, ResNets, ViT, TCC)
3. **Related Work**: 
   - Imitation Learning in Robotics
   - Procedural Learning approaches
   - Intrinsic Reward mechanisms
4. **Methodology**: 
   - Training procedure (representation & policy learning)
   - Reward estimation with procedural decomposition
   - Intrinsic reward integration
5. **Experiments**: 
   - X-Magical environment setup
   - Baseline comparisons
   - Ablation studies
6. **Conclusion**: Findings and future directions

---

## 🔧 Requirements

### For LaTeX Compilation:
- LaTeX distribution (TeX Live, MiKTeX, or similar)
- TOPtesi bundle version 6.x
- BibTeX for references

### For Implementation (Python-based):
- Python 3.7+
- PyTorch for deep learning
- X-Magical environment
- ResNet-18 encoder
- Soft Actor-Critic (SAC) implementation
- Standard scientific libraries (NumPy, scikit-learn for PCA)

---

## 📝 Compilation

To compile the thesis documents:

```bash
# Main thesis document
pdflatex Luca_Ianniello_Thesis.tex
bibtex Luca_Ianniello_Thesis
pdflatex Luca_Ianniello_Thesis.tex
pdflatex Luca_Ianniello_Thesis.tex

# Abstract
pdflatex abstract.tex

# Frontispiece
pdflatex frontespizio.tex

# Summary
pdflatex summary.tex
```

---

## 👥 Author and Supervisors

**Author**: Luca Ianniello  
**Supervisors**:
- Prof. Giuseppe Bruno Averta
- Prof. Francesca Pistilli
- M.Sc. Andrea Protopapa

**Institution**: Politecnico di Torino  
**Date**: October 2025

---

## 📄 License

This work is academic research conducted at Politecnico di Torino. The thesis template is based on the TOPtesi bundle (Copyright 2008-2018 Claudio Beccari, LaTeX Project Public Licence LPPL v.1.3c or later).

---

## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{ianniello2025inestirl,
  author = {Luca Ianniello},
  title = {Inverse Reinforcement Learning for Mastering Long-Horizon Procedural Tasks from Visual Demonstrations},
  school = {Politecnico di Torino},
  year = {2025},
  month = {October},
  note = {Supervisors: Giuseppe Bruno Averta, Francesca Pistilli, Andrea Protopapa}
}
```

---

## 🔬 Key Contributions

This thesis makes the following contributions:

1. **Integration of procedural decomposition with IRL**: Demonstrates how breaking tasks into 3 subtasks optimally balances complexity and learning efficiency for block placement tasks

2. **Novelty-based intrinsic reward in embedding space**: Shows that computing intrinsic rewards based on minimum distance to previously visited embeddings improves exploration and task success

3. **Adaptive exploration scaling**: Implements dynamic adjustment of intrinsic reward weight (0.2 → 0.4) during subtask transitions to enhance exploration in new task phases

4. **Comprehensive evaluation**: Extensive comparison of visual encoders (ResNet-18/50, ViT) and baseline IRL methods (XIRL, HOLD-R, HOLD-C, CLIP) on ordered block placement

5. **Coverage analysis framework**: Dual-method tracking (PCA grid mapping + hash-based counting) to quantify exploration effectiveness

---

## 🔮 Future Directions

Potential extensions discussed in the thesis:
- Real-world robot deployment on physical manipulators
- Automatic subtask boundary detection from demonstrations
- Extension to tasks with varying numbers of objects
- Integration with model-based planning
- Transfer learning to related manipulation domains
- Human demonstration collection and learning
