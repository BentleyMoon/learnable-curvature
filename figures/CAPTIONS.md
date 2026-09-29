# Figures

Rebuilt from `results/` by `python reproduce.py figures`.

**fig1_scoring_rule_weights.png**: What kappa does. Weight of each attention scoring rule as a function of kappa (learnable_curvature_transformer.py, CurvatureDistance). The shaded band is where trained heads end up (|kappa| 2.2 to 2.55): every trained head uses one rule.

**fig2_gain_kept_d96.png**: The d_model 96 control ladder. Share of the learned-curvature gain over plain dot-product attention that each arm keeps; bars are means, dots are seeds. A random frozen code and all-cosine heads keep all of it; all length-scaled heads keep none.

**fig3_ablation_d128.png**: Two 20-epoch ablations at d_model 128, 2 layers. The random code is best in run B; the no-geometry arm moves more between runs than the arms differ within one.

**fig4_attention_range.png**: Attention range by head. Left: the 4x2 model's heads against the same model with one rule forced on every head and nothing learned (dashed lines). Right: the 8x4 model's 32 heads.

**fig5_sweep.png**: The compression sweep matched at 55 epochs. Each configuration has one baseline and two curvature runs; the shaded band is the mean gap between the two curvature runs.

**fig6_exact_distance.png**: The separate exact-distance model on three synthetic tasks (20 epochs, one seed). Exact Euclidean distance does as well as curved geometry; both beat the dot product where the task is periodic.

**fig7_toy_seeds.png**: The numpy toy with true Poincare and sphere distances, 11 seeds. The flat-cluster control drifts negative too; the dotted lines are the script's own +/-0.15 thresholds.

**fig8_kappa_trajectories.png**: Kappa for each of the 8x4 model's 32 heads over 55 epochs. Heads leave zero in the first epochs and keep the side they choose.

**fig9_first_layer.png**: Left, 3 layers: fixed cosine scoring in layer 1 keeps the learned gain, as a whole layer (filled) or as one head (open, two different heads); in layers 2 and 3 it keeps little. Right, 6 layers: improvement over no curvature on each seed. Learned curvature buys about nothing; one fixed cosine head in layer 1 helps on every seed, about as much as cosine on all 24 heads; the same head in layer 4 hurts.

