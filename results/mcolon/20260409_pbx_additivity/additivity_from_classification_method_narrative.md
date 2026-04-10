Got it. You’re trying to turn your existing “all-pairs, time-binned logistic regression” machinery into a phenotype-direction engine so you can answer:
	1.	Are PBX phenotypes aligned (same direction) across comparisons?
	2.	Is pbx1b+4 “more of the same” than pbx4 alone (additivity / severity along that direction)?
	3.	Do unrelated phenotypes (cep290) not align (negative control)?

Below is a scoped narrative + the math + exactly what to extract from sklearn, with a workflow that stays close to what you already have.

⸻

Narrative arc (tight + defensible)

Claim you want to support

PBX perturbations share a common phenotypic direction in MorphSeq space, and the double perturbation (pbx1b+4) is an amplified displacement along that same direction.

Evidence ladder

Evidence 1: Directional alignment across time

For each 2 hpf time bin, you train a linear classifier (logistic regression) for:
	•	NI (non-injected) vs IC (injection control)  ← “no phenotype / procedure control”
	•	NI vs pbx4
	•	NI vs pbx1b
	•	NI vs pbx1b+4
	•	NI vs cep290  ← “different phenotype” control

Each model gives a weight vector w_t (in feature space) defining a direction of separation.

You show:
	•	w_t^{\text{pbx4}} is aligned with w_t^{\text{pbx1b+4}} over time
	•	w_t^{\text{cep290}} is not aligned with PBX vectors
	•	w_t^{\text{NI vs IC}} is near-zero or unstable / orthogonal to PBX (procedure control)

Quantify alignment using cosine similarity (sign-corrected, see below):

\mathrm{cos}(w_t^{A}, w_t^{B})=\frac{w_t^{A}\cdot w_t^{B}}{\|w_t^{A}\|\|w_t^{B}\|}

Plot this over time.

Evidence 2: Additivity as “more displacement along the PBX axis”

Pick a reference direction for PBX, e.g.:
	•	a time-averaged unit vector \bar{u}^{\text{pbx4}}
	•	or better: a weighted average across time bins (weighted by AUROC or sample size)

Then for every embryo at every time bin, compute a scalar projection score:

s_{e,t} = \langle x_{e,t}, \bar{u} \rangle

Where x_{e,t} is the embryo’s feature vector (embedding/morphometrics) at time bin t.

Then show:
	•	pbx1b+4 distribution is shifted further in the PBX direction than pbx4
	•	trajectories s_{e,t} separate earlier/more strongly for pbx1b+4
	•	cep290 doesn’t shift consistently along the PBX axis

Evidence 3: Per-embryo summary

For each embryo, normalize within time bins (to remove time-dependent scale drift), then average:

z_{e,t} = \frac{s_{e,t} - \mu_{\text{NI},t}}{\sigma_{\text{NI},t}}
\qquad
\bar{z}_e = \frac{1}{T}\sum_t z_{e,t}

Plot distributions of \bar{z}_e by group:
	•	NI centered at ~0
	•	pbx4 shifted
	•	pbx1b+4 shifted further
	•	IC near NI
	•	cep290 not consistently shifted (or shifted in a different direction if you test its own axis)

This is a clean “additivity = more of the same” picture.

⸻

The math details you need to get right

1) Logistic regression gives you a direction

For binary logistic regression with features x:

P(y=1|x)=\sigma(w^\top x + b)
	•	w is the normal vector to the decision boundary
	•	moving in the +w direction increases probability of the positive class

So for a given time bin and pair (negative class = NI, positive class = PBX), w_t is a phenotype direction.

Important: feature scaling

The magnitude and orientation of w depend on scaling. To compare directions across time bins, you need either:
	•	consistent preprocessing across bins (same scaler fit globally), or
	•	interpret directions in a whitened space (e.g. global z-scored features)

If you fit a separate StandardScaler per bin, your w_t live in different coordinate systems and cosine comparisons become less meaningful.

MVP recommendation: fit one global scaler on a reference population (e.g. NI across all times, or all data) and use it for all bins.

2) Sign ambiguity and “direction relative to positive/negative”

Cosine similarity can flip sign if your positive class definition flips.

You said you’ll keep NI as negative, PBX/IC as positive. Good. That fixes the sign in principle.

But to be robust, define a consistent sign rule per vector:
	•	compute mean feature vector for each class in that bin: \mu_+, \mu_-
	•	enforce:

\langle w_t, (\mu_+ - \mu_-) \rangle > 0

If it’s negative, flip w_t \leftarrow -w_t, b_t \leftarrow -b_t.

This makes “positive direction” literally point from negative centroid toward positive centroid.

3) “Orthogonal vector” is not necessary

You mentioned “get the orthogonal vector to that vector.” For additive directionality, you usually don’t need that.

What you want is:
	•	the axis of separation (that is w)
	•	projections along that axis (scores)
	•	optionally, you can look at the residual subspace orthogonal to w if you want to show “the remaining variation is not shared,” but it’s not required to make the additive argument.

If you do want it:
	•	orthogonal component of a point x relative to unit u:
x_\perp = x - \langle x,u\rangle u
But again, not needed for the main narrative.

⸻

How to extract the vectors from sklearn (concrete)

Assuming you train per-bin models using sklearn LogisticRegression (binary) or a pipeline.

Case A: plain LogisticRegression

clf = LogisticRegression(...)
clf.fit(X, y)
w = clf.coef_.ravel()      # shape (n_features,)
b = float(clf.intercept_)  # scalar

Case B: Pipeline with scaler + logistic regression

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(...)),
])
pipe.fit(X, y)

w_scaled = pipe.named_steps["clf"].coef_.ravel()
b = float(pipe.named_steps["clf"].intercept_)

But note: w_scaled is in scaled feature space. If you want a direction in original feature units:

For StandardScaler with x_scaled = (x - mean)/scale, the relationship is:

w_{\text{orig}} = \frac{w_{\text{scaled}}}{\text{scale}}
b_{\text{orig}} = b_{\text{scaled}} - w_{\text{scaled}}^\top \frac{\text{mean}}{\text{scale}}

In code:

scaler = pipe.named_steps["scaler"]
w_scaled = pipe.named_steps["clf"].coef_.ravel()
w_orig = w_scaled / scaler.scale_
b_orig = float(pipe.named_steps["clf"].intercept_) - (w_scaled * scaler.mean_ / scaler.scale_).sum()

If you’re comparing directions across bins, it’s often simplest to compare in the same scaled space (provided the scaler is global and fixed), so you don’t need to invert.

Case C: multiclass

If you trained multiclass LR (softmax), then:
	•	coef_ is shape (n_classes, n_features)
	•	each row corresponds to one class vs reference in the multinomial formulation

For your additive story, it’s cleaner to use binary LR for each pair (which you already do).

⸻

What to save from your existing all-pairs framework

You already have per-time-bin AUROC and p-values. Add these artifacts per (bin, pair):

Required per model
	•	w (vector)
	•	b (intercept)
	•	metadata: time_bin, negative_label, positive_label, n_pos, n_neg
	•	model quality: AUROC, maybe CV mean/stderr

Optional but helpful
	•	the scaler parameters (if not global): mean/scale
	•	training set mask or sample IDs for reproducibility

Do not save the whole sklearn object unless you want model replay. Saving just (w, b) plus preprocessing info is enough for directionality.

⸻

The analysis you described, made concrete

Step 0: choose a reference coordinate system
	•	Use your embedding/morphometric feature vector x per embryo per bin.
	•	Apply a single global scaler (recommended).

Step 1: fit pairwise bin-wise models, extract signed unit vectors

For each time bin t and comparison c:
	•	fit LR → w_{t,c}
	•	sign-correct using (\mu_+ - \mu_-)
	•	normalize to unit vector:
u_{t,c} = \frac{w_{t,c}}{\|w_{t,c}\|}

Store u_{t,c}, and also store ||w|| as “strength” if you want.

Step 2: timewise alignment plots

Compute over time:
	•	cos(u_t_pbx4, u_t_pbx1b4)
	•	cos(u_t_pbx4, u_t_cep290)
	•	optionally cos(u_t_pbx4, u_t_ic)

Plot vs time. You expect:
	•	PBX vs PBX high positive
	•	PBX vs cep290 low / noisy / negative

Step 3: define a single PBX axis

Compute a weighted average axis:

\bar{u} = \frac{\sum_t \alpha_t u_t}{\|\sum_t \alpha_t u_t\|}

where \alpha_t could be:
	•	1 (uniform)
	•	AUROC_t (more weight when separation is strong)
	•	sample size weight

This prevents noisy early bins from dominating.

Step 4: project embryos over time

For each embryo/time point:

s_{e,t} = \langle x_{e,t}, \bar{u} \rangle

Plot trajectories (mean ± CI) of s_{e,t} by group.

Expect:
	•	pbx1b+4 further from NI than pbx4
	•	separation emerges around same windows

Step 5: embryo-level summaries with within-bin normalization

Within each bin:
	•	compute NI mean/std of scores
	•	z-score all embryos
	•	average z across bins → per embryo scalar

Compare distributions.

⸻

How to implement this with your existing infrastructure (minimal changes)

Since you already do all-pairs per bin, add an optional “save vectors” mode:
	•	after fitting each binary LR, store:
	•	time bin
	•	positive/negative class
	•	w and b
	•	||w||
	•	AUROC, pval, n_pos, n_neg

Write to a tidy dataframe like:

time_bin_center	positive_label	negative_label	w	b	w_norm	auroc	pval	n_pos	n_neg


Where w can be stored as:
	•	numpy array in .npz keyed by row id, or
	•	list/JSON string if you already serialize arrays elsewhere

Best: store vectors in .npz and store an id in the table.

⸻

One more critical note: avoid circularity

If your features include classifier outputs, you’ll be double-dipping. The “direction” should be derived from raw-ish features (embedding or morphometrics), not from predicted probabilities from some other classifier trained on the same labels.

Using your VAE embedding or morphometric features is fine.

⸻

If you want, I can plug into your exact code shape

You said you have “a good framework for all pairwise comparisons.” If you paste:
	•	the function where you fit the per-bin sklearn model
	•	how you build X, y per bin
	•	what object you currently store as “scores_df”

I can show you exactly where to grab coef_, how to sign-correct, and how to wire “vector artifacts” into your saved outputs with minimal churn.

But even without that, the extraction itself is always:
	•	w = clf.coef_.ravel()
	•	b = clf.intercept_[0]
	•	sign-correct using class means
	•	normalize to unit vector for alignment comparisons