# EECS 6699 Final Project: The Expressive Power of Depth — and Its Robustness Under Noise

**Columbia University · Department of Electrical Engineering**
**Course:** EECSE 6699 — *Mathematics of Deep Learning* (Prof. Predrag R. Jelenković)
**Author:** Yiwen Chen · Spring 2026

---

## 1. Project Overview

This project investigates the **mathematical foundations of depth in neural networks** along two tightly coupled axes:

1. **Expressivity** — *Does depth provide an exponential representational advantage over width?*
2. **Robustness** — *Does that advantage survive when the input or the supervision signal is corrupted by noise?*

Building on the *Depth Separation Theorem* of **Telgarsky (2016)**, we first reproduce the classical result that a deep, narrow ReLU network can fit a high-frequency sawtooth function that any parameter-matched shallow network provably cannot. We then **extend the baseline** with a systematic empirical study of **noise robustness**, asking whether the deep network's exponential expressivity is a fragile artifact of clean data or a structural property that persists under adversarial and stochastic perturbations.

The compositional, piecewise-linear nature of ReLU produces $2^L$ linear regions with depth $L$ but only $O(W)$ regions with width $W$. Whether those exponentially many regions are *useful* or merely *brittle* under noise is the central empirical question of the extended study.

---

## 2. Research Questions

| # | Question | Type |
| :-- | :-- | :-- |
| **RQ1** | Under a fixed parameter budget, does a deep narrow network strictly outperform a parameter-matched shallow wide network on iterated sawtooth targets? | Reproduction |
| **RQ2** | When inputs are corrupted by Gaussian noise $x \mapsto x + \epsilon,\ \epsilon \sim \mathcal{N}(0,\sigma^2)$, does the depth advantage persist, shrink, or invert? | Extension |
| **RQ3** | When training labels are corrupted by Gaussian noise $y \mapsto y + \delta$, do deep networks overfit the noise faster than shallow ones (consistent with their higher expressivity)? | Extension |
| **RQ4** | Under bounded-$\ell_\infty$ adversarial perturbations (FGSM / PGD), is there a **critical perturbation budget** $\epsilon^\star$ at which the deep model's advantage collapses? | Extension |
| **RQ5** | How does the **empirical Lipschitz constant** of the trained network scale with depth, and does it predict the robustness gap observed in RQ2–RQ4? | Theory ↔ Experiment |

---

## 3. Theoretical Background

### 3.1 The Compositional Power of ReLU

The ReLU activation $\sigma(x) = \max(0, x)$ is piecewise linear, so any ReLU network computes a continuous piecewise-linear function. Define the *mirror operator*
$$\phi(x) \;=\; |2x - 1|,$$
which a 2-layer ReLU network represents exactly with two hidden units. Iterating $\phi$ yields the sawtooth family
$$f_k(x) \;=\; \underbrace{\phi \circ \phi \circ \cdots \circ \phi}_{k\text{ times}}(x), \qquad k \in \mathbb{N}.$$
Each composition **doubles** the number of linear regions, so $f_k$ has exactly $2^k$ peaks on $[0,1]$.

### 3.2 Depth Separation (Telgarsky, 2016)

> **Theorem (informal).** For every $k \in \mathbb{N}$, there exists a function $f_k:[0,1]\to\mathbb{R}$ representable by a ReLU network of depth $O(k)$ and width $O(1)$, such that *any* network of depth 2 must use $\Omega(2^k)$ neurons to approximate $f_k$ to within constant $L^1$ error.

The depth gain is therefore **exponential in $k$**. Width is a strictly weaker resource than depth for this function class.

### 3.3 Why Noise Matters: A Lipschitz Bound

For a piecewise-linear network $g_\theta$, the worst-case sensitivity to input perturbation is governed by its (local) Lipschitz constant
$$L(g_\theta) \;=\; \sup_{x \neq x'} \frac{\| g_\theta(x) - g_\theta(x') \|}{\| x - x' \|}.$$
A classical product bound gives
$$L(g_\theta) \;\le\; \prod_{\ell=1}^{L} \| W_\ell \|_2,$$
which grows **multiplicatively in depth**. This is precisely the source of the central tension in our extension:

- *Telgarsky* tells us depth buys exponential expressivity.
- *Lipschitz analysis* warns us depth may also buy exponential sensitivity to noise.

The empirical study in §5 asks which effect dominates in the regime we can train.

### 3.4 Linear Regions and Robustness

Following **Hanin & Rolnick (2019)**, the *expected* number of linear regions of a randomly initialized ReLU network grows polynomially in width but only modestly with depth at initialization — yet trained networks fitting $f_k$ must approach the worst-case $2^L$ count. We therefore predict a **robustness phase transition**: noise of magnitude $\sigma$ erases linear regions whose width is $\lesssim \sigma$, collapsing the deep network's effective expressivity once $\sigma \gtrsim 2^{-k}$.

---

## 4. Hypotheses

- **H1 (Clean expressivity).** With matched parameter count $N$, the deep network should achieve substantially lower test MSE than the shallow network on $f_4$. The current W=16 rescue run gives a moderate $2.9\times$ advantage, below the original one-order-of-magnitude target.
- **H2 (Input-noise resilience).** For small $\sigma$ (input noise $\sigma \ll 2^{-k}$), the deep network retains its advantage; for $\sigma \gtrsim 2^{-k}$, the gap closes monotonically and may invert.
- **H3 (Label-noise overfitting).** The deep network's training loss decays faster than the shallow network's even on noisy labels, but its *test* loss is higher — consistent with depth memorizing noise.
- **H4 (Adversarial threshold).** There exists a critical $\epsilon^\star \approx 2^{-k}$ above which the deep network's adversarial robust accuracy drops below the shallow network's.
- **H5 (Lipschitz mediation).** The robustness gap in H2–H4 is quantitatively explained by the empirical Lipschitz constant ratio $L(g_{\text{deep}})/L(g_{\text{shallow}})$.

---

## 5. Experimental Design

### 5.1 Target Function and Models

- **Target.** Iterated sawtooth $f_k(x) = \phi^{(k)}(x)$ with $k \in \{2, 3, 4, 5, 6\}$ to study how the depth advantage scales with target complexity.
- **Clean H1 rescue model.** Depth $L=9$, width $W=16$ for $k=4$; 1953 trainable parameters. The matched shallow model has depth $L=2$, width $W=651$, and 1954 parameters. This is the current Phase 1 result used for the clean expressivity table.
- **Robustness baseline model.** Phase 2--5 figures and tables were generated with the original depth-$9$, width-$8$ deep model and depth-$2$, width-$176$ shallow model (529 parameters each). Under the conservative route, those robustness results are kept as a separate baseline study rather than rerun with W=16.
- **Initialization.** Kaiming-normal (He et al. 2015) for hidden ReLU layers; Xavier for the
  output. Default PyTorch uniform init causes catastrophic dying-ReLU at this depth.
- **Optimizer.** Adam with **cosine learning-rate annealing** (lr: $5\!\times\!10^{-3} \to 10^{-5}$
  over $3\!\times\!10^4$ epochs) and **gradient-norm clipping** at $\|g\|_2 = 1$.
- **Curriculum learning.** The 30 000 epochs are weighted across four stages with ratio [1:1:2:4],
  training on $f_1, f_2, f_3, f_4$ progressively (Bengio et al. 2009). The same optimizer
  and LR scheduler persist across stages, so the cosine annealing reaches its lowest
  lr precisely in the final, hardest stage. **This is required to overcome the spectral
  bias of ReLU MLPs** — without curriculum, neither network can fit $f_4$ from scratch:
  cosine LR alone leaves the loss plateaued at MSE $\approx 0.05$, and the high-frequency
  peaks are never resolved. Because $f_4 = \phi(f_3) = \phi^2(f_2) = \phi^3(f_1)$, the
  iterated structure of the target makes each stage a small refinement of the previous.
- **Seeds.** Five random seeds per configuration; we report mean ± std.

### 5.2 Phase 1 — Baseline (Clean Depth Separation) *(reproduces existing notebook)*

Train both models on clean $(x, f_k(x))$ pairs. Record final MSE, training curves, and qualitative fits. The current W=16 rescue result is deep $0.0101\pm0.0056$ vs. shallow $0.0294\pm0.0075$, so H1 is partially supported but not confirmed at the original $10\times$ criterion.

### 5.3 Phase 2 — Input Noise Sweep *(extension)*

For $\sigma_x \in \{0,\ 10^{-4},\ 10^{-3},\ 10^{-2},\ 10^{-1}\}$:
1. Train on clean data.
2. Evaluate on $\tilde{x} = x + \mathcal{N}(0, \sigma_x^2)$.
3. Plot **MSE vs. $\sigma_x$** for both models on the same axes.
4. Identify $\sigma_x^\star$ at which the curves cross (the *critical noise level*).

Tests RQ2 / H2.

### 5.4 Phase 3 — Label Noise Sweep *(extension)*

Train on $(x,\ f_k(x) + \mathcal{N}(0, \sigma_y^2))$ for $\sigma_y \in \{0, 10^{-3}, 10^{-2}, 10^{-1}\}$, evaluate on clean targets. Track:
- Train MSE vs. epoch (memorization speed)
- Final clean test MSE (true generalization)
- *Generalization gap* = test MSE − train MSE

Tests RQ3 / H3.

### 5.5 Phase 4 — Adversarial Robustness *(extension)*

For each trained model, attack with **FGSM** and **PGD-40** at $\ell_\infty$ budgets $\epsilon \in \{0,\ 10^{-3},\ 10^{-2},\ 5\!\times\!10^{-2}\}$:
$$x_{\text{adv}} \;=\; \mathrm{Proj}_{[0,1]}\left(x + \epsilon \cdot \mathrm{sign}(\nabla_x \mathcal{L}(g_\theta(x), f_k(x)))\right).$$
Measure adversarial MSE and locate $\epsilon^\star$ per RQ4 / H4.

### 5.6 Phase 5 — Lipschitz and Linear-Region Diagnostics *(extension)*

- **Empirical Lipschitz.** Estimate $\hat{L}$ by maximum finite-difference on a 20,000-point grid and via the product of operator norms $\prod_\ell \|W_\ell\|_2$.
- **Linear region count.** Following Hanin & Rolnick, sample the activation pattern of the network on a fine grid and count distinct patterns; compare to the $2^L$ upper bound.
- **Cross-validation.** Regress the Phase 2–4 robustness gaps on $\hat{L}$ to test H5.

### 5.7 Phase 6 — Complexity Scaling *(extension)*

Repeat Phases 1--4 for $k \in \{2, 3, 4, 5, 6\}$. The saved results give qualitative support that $\sigma_x^\star$ and $\epsilon^\star$ decrease with $k$, but $k=2$ has no clean expressivity gap, $k=5$ is marginal, and $k=6$ collapses, so this should not be described as a precise quantitative law.

---

## 6. Key Metrics & Deliverables

| Metric | Phase | Reports On |
| :-- | :-- | :-- |
| Clean test MSE | 1 | Expressivity (H1) |
| Test MSE vs. $\sigma_x$ curve, crossover $\sigma_x^\star$ | 2 | Input robustness (H2) |
| Generalization gap vs. $\sigma_y$ | 3 | Label-noise overfitting (H3) |
| Adversarial MSE, $\epsilon^\star$ | 4 | Adversarial robustness (H4) |
| Empirical Lipschitz $\hat{L}$, linear-region count | 5 | Mechanistic explanation (H5) |
| Qualitative scaling trend for $\sigma_x^\star,\epsilon^\star$ vs. $k$ | 6 | Complexity-dependent transition |

**Deliverables.** (i) Reproducible Jupyter notebook covering all six phases, (ii) a 6–8 page LaTeX research report, (iii) an oral-defense slide deck summarizing theory, results, and limitations.

---

## 7. Repository Structure

```text
EECS6699_Final_Project/
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- .gitattributes
|-- notebooks/
|   |-- 6699_final.ipynb
|   |-- phase1_baseline.ipynb
|   |-- phase2_input_noise.ipynb
|   |-- phase3_label_noise.ipynb
|   |-- phase4_adversarial.ipynb
|   |-- phase5_lipschitz_regions.ipynb
|   `-- phase6_complexity_scaling.ipynb
|-- src/
|   |-- models.py
|   |-- targets.py
|   |-- train.py
|   |-- noise.py
|   |-- diagnostics.py
|   |-- io_utils.py
|   `-- experiment_registry.py
|-- scripts/
|   |-- run_phase1_fixed.py
|   |-- plot_phase1_w16.py
|   `-- check_report_consistency.py
|-- tests/
|   |-- test_models_targets.py
|   |-- test_noise.py
|   |-- test_experiment_registry.py
|   `-- test_scripts.py
|-- results/
|   |-- figures/
|   |-- tables/
|   `-- models/phase1/*_meta.json
|-- report/
|   |-- paper.tex
|   `-- refs.bib
`-- slides/
    `-- presentation_outline.md
```

---
## 8. Installation & Usage

**Prerequisites.** Python ≥ 3.10; PyTorch ≥ 2.1; NumPy; Matplotlib; SciPy.

```bash
# Clone the repository
git clone https://github.com/Ezreal-Mi/6699.git
cd 6699

# Install dependencies
pip install -r requirements.txt

# Reproduce Phase 1 and regenerate its W=16 figure
python scripts/run_phase1_fixed.py
python scripts/plot_phase1_w16.py

# Run notebooks
jupyter lab notebooks/6699_final.ipynb
jupyter lab notebooks/phase2_input_noise.ipynb

# Check report/result consistency
python scripts/check_report_consistency.py

# Run fast tests
pytest
```

Runtime depends on hardware and phase. The saved tables and figures are included
so the report can be checked without rerunning the long experiments.

---

## 9. Project Timeline

| Week | Milestone |
| :-- | :-- |
| 1 | Phase 1 reproduction; finalize README and hypotheses |
| 2 | Implement `src/noise.py`, `src/diagnostics.py`; Phase 2 sweep |
| 3 | Phase 3 (label noise) and Phase 4 (FGSM/PGD) experiments |
| 4 | Phase 5 Lipschitz / linear-region diagnostics |
| 5 | Phase 6 complexity scaling; draft LaTeX report |
| 6 | Final report and presentation |

---

## 10. References

1. Telgarsky, M. (2016). *Benefits of depth in neural networks.* COLT.
2. Eldan, R., & Shamir, O. (2016). *The power of depth for feedforward neural networks.* COLT.
3. Goodfellow, I., Shlens, J., & Szegedy, C. (2015). *Explaining and harnessing adversarial examples.* ICLR.
4. Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). *Towards deep learning models resistant to adversarial attacks.* ICLR.
5. Hanin, B., & Rolnick, D. (2019). *Deep ReLU networks have surprisingly few activation patterns.* NeurIPS.
6. Sokolić, J., Giryes, R., Sapiro, G., & Rodrigues, M. R. D. (2017). *Robust large-margin deep neural networks.* IEEE TSP.
7. Bartlett, P. L., Foster, D. J., & Telgarsky, M. (2017). *Spectrally-normalized margin bounds for neural networks.* NeurIPS.
8. Zhang, C., Bengio, S., Hardt, M., Recht, B., & Vinyals, O. (2017). *Understanding deep learning requires rethinking generalization.* ICLR.
9. Jelenković, P. R. (2026). *EECS 6699 — Mathematics of Deep Learning, Lecture Notes.* Columbia University.
