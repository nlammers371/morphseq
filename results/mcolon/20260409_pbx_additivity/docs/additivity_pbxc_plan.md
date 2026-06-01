Methods Plan: PBX1B × PBX4 Decomposition Validation
Status: living methods outline
Scope: machinery validation before real double-additivity inference
Unit of analysis: within-time-bin bulk embryo means in VAE latent space

==================================================
Overview
==================================================

This validation analysis is organized around three experimental questions:

Q1. Control-span symmetry check
Q2. Axis reality check
Q3. Span specificity check

These three questions establish whether the pbx1b/pbx4 decomposition machinery
behaves sensibly before using it to interpret the real double mutant.

The analysis does not re-estimate phenotype onsets. Those are assumed to be
already known. All calculations are performed separately within three
predefined time bins.

Primary conditions:
- non_inj
- inj_ctrl
- pbx1b
- pbx4
- pbx1b_pbx4 double

Primary baseline:
- non_inj

Representation:
- bulk embryo mean within each time bin, in VAE latent space

==================================================
Common notation
==================================================

Within a given time bin, define:

mu_non    = mean(non_inj embryos)
mu_inj    = mean(inj_ctrl embryos)
mu_1      = mean(pbx1b embryos)
mu_4      = mean(pbx4 embryos)
mu_14     = mean(pbx1b_pbx4 embryos)

Single-gene effect vectors relative to the biological origin:

v_1 = mu_1 - mu_non
v_4 = mu_4 - mu_non

Single-gene design matrix:

V = [v_1  v_4]

All vector norms, cosine similarities, projections, residuals, and condition
numbers are computed separately within each time bin.

==================================================
Q1. Control-span symmetry check
==================================================

Experimental question:
If a target contains no real pbx1b- or pbx4-specific signal, how does it land
in the pbx1b/pbx4 span?

Purpose:
To test whether the decomposition machinery behaves symmetrically and does not
artificially assign control targets strongly to one biological axis.

Biological idea:
Because inj_ctrl and non_inj are already known to be indistinguishable, a
control-derived target projected into the pbx1b/pbx4 span should not look
strongly pbx1b-like or strongly pbx4-like.

Construction within each time bin:
1. Randomly split non_inj into two non-overlapping subsets:
   non_A and non_B

2. Optionally also split inj_ctrl into two non-overlapping subsets:
   inj_A and inj_B

3. Build the control-origin target:
   Preferred simplest version:
   z_ctrl = mean(non_B) - mean(non_A)

   Optional protocol-matched version:
   z_ctrl = mean(inj_B) - mean(non_B)

4. Fit the control target in the real single-gene span:

   z_ctrl ≈ V c + r

   where:
   - V = [v_1  v_4]
   - c = [alpha, beta]^T
   - r = residual

Primary quantities:
1. Coefficients:
   alpha, beta

2. Coefficient balance:
   alpha / (alpha + beta), if alpha and beta are both positive
   or simply compare alpha and beta directly

3. Span-explained fraction:
   R^2_span = 1 - ||r||^2 / ||z_ctrl||^2

4. Residual norm:
   ||r||

Expected behavior:
- coefficients should be small overall
- neither axis should dominate systematically
- if the target lands in the span at all, the loading should be roughly
  balanced rather than strongly one-sided
- residual should remain substantial relative to true biological targets

Interpretation:
This is a machinery symmetry check.
If control-derived targets repeatedly project strongly and asymmetrically onto
one biological axis, the decomposition is biased.
If they project weakly and without consistent axis preference, the machinery
is behaving sensibly.

Outputs:
- distributions of alpha and beta across repeated control splits
- distribution of coefficient imbalance
- distribution of R^2_span and residual norm for control-derived targets

==================================================
Q2. Axis reality check
==================================================

Experimental question:
Are the pbx1b and pbx4 single-gene axes stronger and more structured than
axes that arise from control-only sampling noise?

Purpose:
To validate that the pbx1b and pbx4 phenotype vectors represent real
biological signal.

------------------------------------------
Q2A. Real single-gene axes
------------------------------------------

Within each time bin:
1. Compute the real single-gene effect vectors:

   v_1 = mu_1 - mu_non
   v_4 = mu_4 - mu_non

2. Compute the following summaries:

   a. Axis norms
      ||v_1|| and ||v_4||

   b. Cosine similarity
      cos(theta) = (v_1 · v_4) / (||v_1|| ||v_4||)

   c. Condition number of the two-axis design matrix
      kappa(V), where V = [v_1  v_4]

Interpretation:
- norms quantify effect magnitude
- cosine similarity quantifies directional overlap
- condition number quantifies stability of coefficient attribution

------------------------------------------
Q2B. Fake-axis benchmark
------------------------------------------

Construction within each time bin:
1. Randomly split inj_ctrl into two non-overlapping subsets:
   inj_A and inj_B

2. Randomly split non_inj into two non-overlapping subsets:
   non_A and non_B

3. Construct fake control-derived axes:

   v_fake1 = mean(inj_A) - mean(non_A)
   v_fake2 = mean(inj_B) - mean(non_B)

4. Compute for the fake axes:
   - norms
   - cosine similarity
   - condition number

5. Repeat many times

Comparison to real axes:
- compare ||v_1|| and ||v_4|| to fake-axis norm distributions
- compare real cosine similarity to fake-axis cosine distribution
- compare real condition number to fake-axis condition-number distribution

Interpretation:
Real single-gene axes should exceed the control-derived noise floor in
magnitude and structure.

Outputs:
- per-bin real axis summaries
- benchmark distributions from fake axes
- overlay figure: real vs fake-axis benchmark

==================================================
Q3. Span specificity check
==================================================

Experimental question:
Does the pbx1b/pbx4 span reconstruct the correct biological targets and reject
irrelevant control-derived variation?

Purpose:
To show that the span is biologically specific, not merely flexible.

Construction within each time bin:
1. Randomly split pbx1b into non-overlapping subsets:
   pbx1b_A and pbx1b_B

2. Randomly split pbx4 into non-overlapping subsets:
   pbx4_A and pbx4_B

3. Randomly split non_inj into non-overlapping subsets:
   non_A and non_B

4. If possible, split inj_ctrl into non-overlapping subsets:
   inj_A and inj_B

5. Build the base span from subset A:

   v_1_base = mean(pbx1b_A) - mean(non_A)
   v_4_base = mean(pbx4_A) - mean(non_A)

   V_base = [v_1_base  v_4_base]

6. Build held-out targets from subset B:

   z_pbx1b = mean(pbx1b_B) - mean(non_B)
   z_pbx4  = mean(pbx4_B)  - mean(non_B)

   Preferred symmetric control target:
   z_inj   = mean(inj_B)   - mean(non_B)

7. Fit each target independently:

   z_target ≈ V_base c + r

Primary quantities for each target:
- coefficients [alpha, beta]
- R^2_span
- residual norm

Expected behavior:
A. PBX1B held-out target
- alpha approximately 1
- beta approximately 0
- high R^2_span
- low residual

B. PBX4 held-out target
- alpha approximately 0
- beta approximately 1
- high R^2_span
- low residual

C. Injection-control target
- no strong or stable assignment to either axis
- lower R^2_span than real single-gene targets
- larger residual than real single-gene targets

Interpretation:
The span is biologically specific if it reconstructs held-out real single-gene
targets well and rejects control-derived targets.

Outputs:
- distributions of alpha, beta, R^2_span, and residual norm for each target
- specificity plots comparing pbx1b, pbx4, and inj_ctrl targets

==================================================
Repetition strategy
==================================================

Q1 to Q3 all rely on repeated random splitting within each time bin.

For each repeated split, recompute:
- relevant means
- single-gene axes
- control targets
- held-out targets
- coefficients
- R^2_span
- residual norms
- condition numbers where relevant

Recommended summaries:
- mean
- SD
- empirical confidence intervals
- sign consistency or dominance consistency of coefficients

==================================================
Trust hierarchy
==================================================

Most trustworthy:
1. Q2 real-axis magnitude vs fake-axis benchmark
2. Q3 high held-out reconstruction for real single-gene targets
3. Q3 low or nonspecific reconstruction for control-derived targets

Moderately trustworthy:
4. cosine similarity between v_1 and v_4
5. condition number as a reliability annotation
6. Q1 control-span symmetry

Most fragile:
7. exact coefficient magnitudes when the design matrix is ill-conditioned

Rule:
If diagnostics disagree, trust reconstruction specificity and benchmark
separation more than exact coefficient attribution.

==================================================
Planned outputs
==================================================

Per-bin summary tables:
- Q1 control-target coefficients, R^2_span, residual norm
- Q2 real axis norms, cosine similarity, condition number
- Q3 held-out target coefficients, R^2_span, residual norm

Main figures:
1. Q1 control-target coefficient balance across repeated splits
2. Q2 real axis norms overlaid on fake-axis benchmark distributions
3. Q2 real cosine similarity overlaid on fake-axis benchmark
4. Q3 specificity plots:
   - held-out pbx1b target
   - held-out pbx4 target
   - control-derived target
5. Q3 distributions of R^2_span and residual norm across repeated splits

==================================================
Deferred
==================================================

Not included in this validation-stage methods outline:
- extracng directions from classif via classfifcation save_ditrectio (read :  classifier_Assisted_morpholgica_geometry_extraction.md for the state of the apii)
- real double-mutant additivity inference
- embryo-level decomposition as the main analysis


--- 

Updated next-step logic

Stage 1: Validate the decomposition frame
- Q1. Control-span symmetry
- Q2. Axis reality
- Q3. Span specificity

Purpose:
Make sure the pbx1b/pbx4 span is real, stable, and biologically specific.

y outputs:
- R^2_span for the real double
- residual norm for the real double
- coefficients [a, b]
- bootstrap intervals on all of the above

Purpose:
Ask whether the real double lies mostly in the pbx1b/pbx4 single-gene span, or whether it has a substantial residual component.

Stage 3: Individual embryo decomposition
For each double embryo x_i within a bin:
- center it relative to non_inj:
  z_i = x_i - mean(non_inj)

- fit:
  z_i ≈ V c_i + r_i

Per-embryo outputs:
- coefficients c_i = [a_i, b_i]
- residual norm ||r_i||
- span-explained fraction:
  R^2_i = 1 - ||r_i||^2 / ||z_i||^2

Purpose:
Ask whether the double is uniformly additive across embryos, or whether:
- some embryos are well explained by the single-gene span
- others carry large residual interaction structure

This lets you distinguish:
- additive mean with heterogeneous embryos
vs
- genuinely additive embryos throughout.

Recommended interpretation order

Most trustworthy:
1. real-double R^2_span
2. real-double residual norm
3. distribution of per-embryo residuals
4. distribution of per-embryo coefficients

Most fragile:
5. exact per-embryo attribution between pbx1b and pbx4,
   especially if condition number is high

Practical recommendation

Do not jump straight from Q3 to embryo-level interpretation.
Use this sequence:

1. validate span on held-out singles and controls
2. fit the real double mean
3. only then fit individual double embryos

That way:
- if the real-double mean already sits far outside the span, the embryo-level residuals are expected
- if the real-double mean is well explained but embryo-level fits are heterogeneous, that tells you the mean is hiding substructure

==================================================
Next steps: multi-phenotype axis comparison
==================================================

Motivation:
The current cosine analysis (Q2) and trajectory condensation only compare pbx1b
and pbx4 against a control null. This tells us the two axes are real and roughly
aligned — but it does not tell us whether that alignment is biologically meaningful
or just a generic property of any strong morphological phenotype. In other words:
is the high cos(v_1, v_4) specific to pbx1b/pbx4 sharing biology, or do all
strong phenotypes converge to a similar direction in this VAE space (degenerate
behavior — "everyone goes the same wrong direction")?

Action item:
Add CEP290 homozygotes as a third phenotype arm to Q2 cosine analysis and to
the trajectory condensation comparison.

Specifically:
- Compute v_cep290 = mu_cep290 - mu_non within each time bin (same construction as v_1, v_4)
- Compute pairwise cosines: cos(v_1, v_4), cos(v_1, v_cep290), cos(v_4, v_cep290)
- Plot all three over time against the same fake-axis null ribbon
- Interpretation key:
    - cos(v_1, v_4) >> cos(v_1, v_cep) and cos(v_4, v_cep): pbx1b/pbx4 share
      a specific axis that CEP290 does not — decomposition is non-degenerate
    - all three cosines similarly high: all strong phenotypes pile into one
      direction — the latent space may be too low-dimensional or the VAE
      collapses diverse phenotypes onto a shared direction ("going wrong" axis)
    - cos(v_cep) near null while cos(v_1, v_4) is elevated: strongest possible
      evidence that pbx1b/pbx4 share something specific

For trajectory condensation:
- Include CEP290 homozygotes as a third trajectory alongside pbx1b/pbx4
- Condensation similarity between pbx1b and pbx4 vs pbx/CEP290 comparisons
  will reveal whether the two pbx crispants are converging specifically or
  just tracking global developmental disruption