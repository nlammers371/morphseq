Totally fair. Let’s shrink the beast into one clean first step.

Step 1: Define the method goal and inputs

Copy-paste this as the beginning of the .md doc:

# CEP290 Phenotype Label Transfer Method
## Goal
The goal of this method is to assign phenotype labels to new CEP290 embryo data using a previously labeled reference dataframe.
The method compares embeddings from a query dataframe to embeddings from a labeled reference dataframe. Labels are transferred from the closest reference examples using distance-weighted K-nearest neighbors.
The final prediction is made at the embryo level.
If a query embryo has multiple images, the method pools evidence across those images. If a query embryo has only one image, the method still returns an embryo-level label, but the output should indicate that the prediction is based on limited image evidence.
## Inputs
### Reference dataframe
The reference dataframe contains previously labeled embryo/image data.
Each row corresponds to one image from a labeled embryo.
Required columns:
```text
embryo_id
snip_id
predicted_stage_hpf
cluster_categories
embedding columns

The cluster_categories column contains the phenotype label to transfer.

For the first version, we will use only the main phenotype categories:

Low_to_High
High_to_Low
Intermediate
Not Penetrant

Rows with missing cluster_categories values should be removed before running label transfer.

Query dataframe

The query dataframe contains new embryo/image data that we want to classify.

Each row corresponds to one image from a query embryo.

Required columns:

embryo_id
snip_id
predicted_stage_hpf
embedding columns

The query dataframe does not need to contain phenotype labels.

The method should support both:

one image per embryo
multiple images per embryo

When multiple images are available for the same embryo, image-level evidence is averaged into one embryo-level prediction.

When only one image is available for an embryo, the image-level prediction is also the embryo-level prediction.

The next small chunk should be **time-window filtering**, since that controls what rows enter both the reference and query dataframes.

Great. Next stage: time-window filtering. This defines which images are eligible for comparison.

## Time Window Filtering
The method should allow the user to select a developmental time window before running label transfer.
For the CEP290 phenotype analysis, the default time window will be:
```text
30 to 48 hpf

This time window should be applied to both the reference dataframe and the query dataframe.

The purpose of this filtering is to ensure that query images are compared only against reference images from the same biologically relevant developmental window.

Time Window Parameters

The method should expose the following parameters:

time_col = "predicted_stage_hpf"
min_hpf = 30
max_hpf = 48

Rows should be kept if:

min_hpf <= predicted_stage_hpf <= max_hpf

Reference Dataframe Filtering

For the reference dataframe, keep only rows that satisfy both conditions:

1. predicted_stage_hpf is within the selected time window
2. cluster_categories is not NaN

This removes unlabeled or outlier rows from the reference data before label transfer.

Query Dataframe Filtering

For the query dataframe, keep only rows where:

predicted_stage_hpf is within the selected time window

The query dataframe does not need to contain labels.

Handling Embryos With No Images in the Time Window

Some query embryos may have no images within the selected time window.

These embryos should not be passed into the KNN label-transfer step.

Instead, they should be recorded in the final output with a status such as:

outside_time_window

Their predicted label should be set to:

NaN

or:

not_assigned

Handling Embryos With One Image in the Time Window

If a query embryo has exactly one image in the selected time window, the method should still run.

The single image will produce the embryo-level prediction.

However, the final output should record:

n_images = 1

and the embryo status should indicate that the prediction is based on limited evidence.

Possible status:

single_image

Handling Embryos With Multiple Images in the Time Window

If a query embryo has multiple images in the selected time window, the method should run label transfer for each image separately.

The image-level evidence will later be averaged into a single embryo-level prediction.

The final output should record:

n_images
min_query_hpf
max_query_hpf

This allows us to inspect how much developmental time was represented for each query embryo.

Next stage should be **embedding column selection and KNN neighbor search**.

Next stage: image-level label probabilities. This turns the neighbor table into phenotype probabilities for each query image.

## Image-Level Label Probabilities
After constructing the neighbor long table, the next step is to compute phenotype label probabilities for each query image.
Each query image has `K` nearest reference images. Each reference neighbor has:
```text
ref_label
distance
weight

The goal is to use the reference neighbor labels and distance weights to estimate the probability of each phenotype label for the query image.

Distance-Weighted Label Voting

For each query image, compute a weighted vote for each phenotype label.

The probability of a label is:

P(label) = sum(weights of neighbors with that label) / sum(weights of all neighbors)

This means that closer reference images contribute more strongly to the query image’s label probability.

Labels Used

For the first version, use only the main phenotype categories:

Low_to_High
High_to_Low
Intermediate
Not Penetrant

Subcategories should be ignored for now.

Image-Level Probability Table

The output should be a long-format table with one row per query image per label.

Suggested output file:

image_label_probabilities.csv

Suggested columns:

query_embryo_id
query_snip_id
query_hpf
label
image_label_probability

Example:

query_embryo_id	query_snip_id	query_hpf	label	image_label_probability
emb_001	img_001	34.2	Low_to_High	0.72
emb_001	img_001	34.2	High_to_Low	0.08
emb_001	img_001	34.2	Intermediate	0.15
emb_001	img_001	34.2	Not Penetrant	0.05

For each query image, the probabilities across all labels should sum to 1.

Image-Level Predicted Label

The image-level predicted label is the label with the highest image-level probability.

image_pred_label = label with highest image_label_probability

This is not the final embryo-level prediction. It is an intermediate image-level prediction used for inspection and confidence scoring.

Image-Level Neighbor Agreement

The image-level neighbor agreement score measures how strongly the nearest neighbors agree on the predicted label.

For each query image:

image_neighbor_agreement = max(image_label_probability)

Interpretation:

High image_neighbor_agreement:
    Most of the distance-weighted neighbor evidence supports one label.
Low image_neighbor_agreement:
    The nearest neighbors are split across multiple labels.

Example:

Low_to_High: 0.85
High_to_Low: 0.05
Intermediate: 0.07
Not Penetrant: 0.03
image_pred_label = Low_to_High
image_neighbor_agreement = 0.85

This is a high-agreement image.

Example:

Low_to_High: 0.35
High_to_Low: 0.30
Intermediate: 0.25
Not Penetrant: 0.10
image_pred_label = Low_to_High
image_neighbor_agreement = 0.35

This is a low-agreement image.

Image Prediction Summary Table

In addition to the long-format probability table, create a one-row-per-image summary table.

Suggested output file:

image_prediction_summary.csv

Suggested columns:

query_embryo_id
query_snip_id
query_hpf
image_pred_label
image_neighbor_agreement

Later, distance-based confidence scores can be added to this table.

Why Preserve Image-Level Predictions?

Even though the final prediction is made per embryo, image-level predictions are useful for debugging and interpretation.

They allow us to inspect:

whether different images from the same embryo support the same label
whether predictions change across developmental time
whether one image has low agreement while others are confident
whether a single poor-quality image may be driving uncertainty

The image-level prediction table is therefore an intermediate evidence table, not the final output.

Next stage: **embryo-level probability aggregation**, where image-level evidence becomes one prediction per embryo.

Agreed. Let’s keep the method lean and avoid building side-quests until the main path proves it needs them.

Next stage: embryo-level probability aggregation.

## Embryo-Level Label Probabilities
The final prediction should be made at the embryo level.
The image-level label probabilities are intermediate values. They describe the evidence for each query image. To assign one phenotype label to each embryo, image-level probabilities should be pooled across all images from the same query embryo.
## Aggregating Image-Level Evidence
For each query embryo, average the image-level label probabilities across all images belonging to that embryo.
```python
embryo_label_probability = mean(image_label_probabilities)

This should be done separately for each phenotype label.

For example, if an embryo has multiple images, each image contributes one probability distribution over labels. The embryo-level probability is the average of those image-level distributions.

Single-Image Embryos

If a query embryo has only one image, the embryo-level probabilities are equal to the image-level probabilities for that image.

single image embryo:
    embryo_label_probability = image_label_probability

The method should still assign a label to this embryo.

However, the output should preserve that the prediction was based on only one image by recording:

n_images = 1

This allows downstream analysis to treat single-image predictions more cautiously if needed.

Multi-Image Embryos

If a query embryo has multiple images, the embryo-level probability for each label is the average probability across images.

multi-image embryo:
    embryo_label_probability = mean probability across images

This allows the method to pool evidence across the embryo.

Images that strongly support the same label will increase the embryo-level probability for that label. Images that disagree will produce a more mixed embryo-level probability distribution.

Embryo-Level Probability Table

The output should be a long-format table with one row per query embryo per label.

Suggested output file:

embryo_label_probabilities.csv

Suggested columns:

query_embryo_id
label
embryo_label_probability

Example:

query_embryo_id	label	embryo_label_probability
emb_001	Low_to_High	0.81
emb_001	High_to_Low	0.04
emb_001	Intermediate	0.10
emb_001	Not Penetrant	0.05

For each query embryo, the probabilities across all labels should sum to 1.

Embryo-Level Predicted Label

The embryo-level predicted label is the label with the highest embryo-level probability.

embryo_pred_label = label with highest embryo_label_probability

This is the main phenotype assignment produced by the method.

Top Label Probability

The top label probability should also be saved.

top_label_probability = max(embryo_label_probability)

This captures how strongly the embryo-level pooled evidence supports the predicted label.

Example:

Low_to_High: 0.81
High_to_Low: 0.04
Intermediate: 0.10
Not Penetrant: 0.05
embryo_pred_label = Low_to_High
top_label_probability = 0.81

Embryo-Level Summary Table

Create a one-row-per-embryo summary table.

Suggested output file:

embryo_label_transfer_summary.csv

Suggested columns at this stage:

query_embryo_id
n_images
min_query_hpf
max_query_hpf
predicted_label
top_label_probability

Additional confidence scores will be added in the next stage.

Why Average Image Probabilities?

Averaging image probabilities keeps the method simple and interpretable.

It also allows the same method to work for both single-image and multi-image embryos.

For single-image embryos, the prediction comes from one image.

For multi-image embryos, the prediction reflects the average evidence across available images.

This keeps the final prediction embryo-level while preserving the image-level evidence for later inspection.

Next we should add **confidence components**, starting with neighbor agreement, distance confidence, and embryo consistency.

Next stage: confidence scoring. This keeps the prediction interpretable instead of turning it into a little black-box gremlin.

## Confidence Scoring
After assigning embryo-level label probabilities, the method should compute confidence scores for each query embryo.
The goal is not only to assign a phenotype label, but also to estimate how reliable that assignment is.
The final confidence score should be an aggregate score, but each component should also be preserved separately.
The three confidence components are:
```text
1. Neighbor agreement confidence
2. Distance confidence
3. Embryo consistency confidence

The aggregate score is:

embryo_confidence = (
    mean_image_neighbor_agreement
    * distance_in_distribution_score
    * embryo_consistency_score
)

Each component captures a different reason why a prediction may or may not be reliable.

⸻

1. Neighbor Agreement Confidence

Neighbor agreement confidence measures whether the nearest reference images agree on the same label.

This is computed first at the image level.

For each query image:

image_neighbor_agreement = max(image_label_probability)

This is the highest probability assigned to any label for that image.

Example high-agreement image:

Low_to_High: 0.88
High_to_Low: 0.04
Intermediate: 0.06
Not Penetrant: 0.02
image_neighbor_agreement = 0.88

Example low-agreement image:

Low_to_High: 0.36
High_to_Low: 0.31
Intermediate: 0.24
Not Penetrant: 0.09
image_neighbor_agreement = 0.36

For each query embryo, average the image-level neighbor agreement scores:

mean_image_neighbor_agreement = mean(image_neighbor_agreement across images)

Interpretation:

High value:
    The nearest reference images mostly agree on one label.
Low value:
    The nearest reference images are split across several labels.

This score tells us whether the local neighborhood around the query images has a clear label signal.

⸻

2. Distance Confidence

Distance confidence measures whether the query images are close to the labeled reference data.

The method should always assign the closest known label, but it should also detect when a query embryo is far from the reference distribution.

For each query image, compute:

mean_knn_distance = mean(distance to K nearest reference images)

Then compare that value to a reference distance distribution.

The reference distance distribution is computed from the reference dataframe itself:

For each reference image:
    find its K nearest neighboring reference images, excluding itself
    compute the mean distance to those K neighbors

This produces a distribution of typical within-reference KNN distances.

Each query image can then be compared against this distribution.

For each query image:

distance_percentile = percentile of query mean_knn_distance relative to reference mean_knn_distances

Then convert this into an in-distribution score:

distance_in_distribution_score = 1 - (distance_percentile / 100)

Interpretation:

High distance_in_distribution_score:
    The query image is close to the reference data.
Low distance_in_distribution_score:
    The query image is far from the reference data and may be an outlier.

For each query embryo, average this score across images:

embryo_distance_score = mean(distance_in_distribution_score across images)

This gives an embryo-level measure of whether the embryo is close to the labeled reference data.

⸻

3. Embryo Consistency Confidence

Embryo consistency confidence measures whether images from the same embryo support the same final embryo-level label.

For each query embryo:

embryo_consistency_score = fraction of images whose image_pred_label equals embryo_pred_label

Example:

5 images total
4 images have image_pred_label = Low_to_High
1 image has image_pred_label = Intermediate
embryo_pred_label = Low_to_High
embryo_consistency_score = 4 / 5 = 0.80

Interpretation:

High value:
    Images from the same embryo agree with the final embryo-level prediction.
Low value:
    Different images from the same embryo support different labels.

For embryos with only one image:

embryo_consistency_score = 1.0

However, this should not be interpreted as strong trajectory-level evidence. It only means there was no within-embryo disagreement because there was only one image.

The n_images column should be used to distinguish single-image embryos from embryos with pooled evidence across multiple images.

⸻

Aggregate Embryo Confidence

The aggregate embryo confidence score combines the three confidence components:

embryo_confidence = (
    mean_image_neighbor_agreement
    * embryo_distance_score
    * embryo_consistency_score
)

This score will be low if any one of the confidence components is low.

For example:

High neighbor agreement + high distance score + high consistency:
    confident prediction
Low neighbor agreement:
    nearby reference labels disagree
Low distance score:
    query embryo may be far from the reference data
Low consistency:
    different images from the same embryo support different labels

The aggregate confidence score should be saved, but the individual components should always be preserved.

⸻

Confidence Columns in Embryo Summary

The embryo-level summary table should include:

query_embryo_id
n_images
min_query_hpf
max_query_hpf
predicted_label
top_label_probability
mean_image_neighbor_agreement
embryo_distance_score
embryo_consistency_score
embryo_confidence

These values allow the prediction to be interpreted and debugged.

The aggregate score gives a quick summary, while the component scores explain why a prediction is confident or uncertain.

Next stage should be **status labels / flagging uncertainty**, where we translate these confidence components into things like `assigned`, `single_image`, `low_density`, and `ambiguous`.

Agreed. Minimal set is better. Keep n_images as metadata, not a status. Then the status column only needs:

assigned
ambiguous
low_density
not_evaluated

Here’s the revised Markdown chunk.

## Status Labels and Uncertainty Flags
After computing embryo-level predictions and confidence scores, the method should assign a compact status label to each query embryo.
The status label should describe how the prediction should be interpreted.
The method should not create separate status labels for the number of images available. Instead, the number of images should be stored as a metadata column:
```text
n_images

This keeps the status labels simple while still preserving information about how much image-level evidence was available for each embryo.

Minimal Status Set

Use the following status labels:

assigned
ambiguous
low_density
not_evaluated

assigned

Use assigned when the embryo has a predicted label and no major warning flags.

This means:

The embryo was evaluated successfully.
The nearest-neighbor evidence supports a phenotype label.
The embryo is not unusually far from the labeled reference data.

An embryo can be assigned whether it has one image or multiple images.

The amount of image evidence should be interpreted using:

n_images
min_query_hpf
max_query_hpf
query_hpf_range

ambiguous

Use ambiguous when the embryo is close enough to the reference data, but the label evidence is mixed.

This can happen when:

nearest reference neighbors are split across phenotype labels
different images from the same embryo support different labels
top label probability is low

This status means the embryo was evaluated, but the phenotype assignment is uncertain.

Possible rules:

if mean_image_neighbor_agreement < agreement_threshold:
    status = "ambiguous"
if embryo_consistency_score < consistency_threshold:
    status = "ambiguous"
if top_label_probability < top_probability_threshold:
    status = "ambiguous"

low_density

Use low_density when the query embryo is far from the labeled reference data.

The method should still assign the closest known phenotype label when possible, but this status indicates that the embryo may not be well represented by the reference dataset.

This is the outlier-like flag.

The term low_density is preferred because it describes the measurement directly:

The query embryo is in a low-density region relative to the labeled reference embeddings.

Possible rule:

if embryo_distance_score < distance_score_threshold:
    status = "low_density"

not_evaluated

Use not_evaluated when the method could not assign a label.

This can happen when:

the embryo has no images in the selected time window
embedding values are missing
the query dataframe fails input checks
there are not enough reference examples to run KNN

For not_evaluated embryos:

predicted_label = not_assigned
embryo_confidence = NaN

The specific reason should be recorded in:

status_reason

Priority Order for Status Assignment

Some embryos may satisfy multiple warning conditions. To make status assignment consistent, apply a priority order.

Suggested priority:

if cannot_evaluate:
    status = "not_evaluated"
elif embryo_distance_score < distance_score_threshold:
    status = "low_density"
elif mean_image_neighbor_agreement < agreement_threshold:
    status = "ambiguous"
elif embryo_consistency_score < consistency_threshold:
    status = "ambiguous"
elif top_label_probability < top_probability_threshold:
    status = "ambiguous"
else:
    status = "assigned"

This means low_density takes priority over ambiguous.

The reasoning is that if an embryo is far from the reference data, the main issue is that the closest label may not be well supported by the known reference distribution.

Final Output Columns

The embryo-level summary table should include:

query_embryo_id
n_images
min_query_hpf
max_query_hpf
query_hpf_range
predicted_label
top_label_probability
mean_image_neighbor_agreement
embryo_distance_score
embryo_consistency_score
embryo_confidence
status
status_reason

The compact status label gives a quick interpretation, while the individual columns preserve the evidence behind that status.

This is the cleaner version: **four statuses, with `n_images` kept as evidence rather than becoming its own category.**

Next should be validation strategy, because now the method is defined enough to test.

We want to answer:

If we pretend one experiment is new query data, can the labeled reference data from the other experiments recover its embryo-level labels?

Here’s the next Markdown chunk:

## Validation Strategy
Because the CEP290 dataset already contains labeled embryos from multiple experiments, the method should be validated before being applied to truly new data.
The main validation strategy will be leave-one-experiment-out validation.
This tests whether phenotype labels can transfer across experiments, rather than only working within the same experimental batch.
## Leave-One-Experiment-Out Validation
For each experiment, hold out that experiment as the query dataframe.
All remaining experiments are used as the labeled reference dataframe.
For each held-out experiment:
```text
1. Set the held-out experiment as the query dataframe.
2. Set all other experiments as the reference dataframe.
3. Run the full label-transfer method.
4. Compare predicted embryo labels to known embryo labels.
5. Save embryo-level predictions and summary metrics.

Experiments to evaluate:

20250512
20251017_combined
20251106
20251112
20251113
20251205
20251212

Why Leave-One-Experiment-Out?

This validation is important because new data will likely come from a new experiment.

The method should therefore be able to transfer labels across experimental batches.

If performance is strong in leave-one-experiment-out validation, that supports the idea that the KNN label transfer is capturing phenotype structure rather than experiment-specific artifacts.

If performance is weak for a specific held-out experiment, that may indicate batch effects, differences in image quality, differences in developmental timing, or label distributions that are not well represented in the reference dataframe.

Validation Outputs

For each held-out experiment, save embryo-level predictions.

Suggested file:

leave_one_experiment_out_predictions.csv

Suggested columns:

heldout_experiment_id
query_embryo_id
true_label
predicted_label
top_label_probability
n_images
min_query_hpf
max_query_hpf
query_hpf_range
mean_image_neighbor_agreement
embryo_distance_score
embryo_consistency_score
embryo_confidence
status
status_reason

Also save a summary table across experiments.

Suggested file:

leave_one_experiment_out_summary.csv

Suggested columns:

heldout_experiment_id
n_query_embryos
n_assigned
n_ambiguous
n_low_density
n_not_evaluated
accuracy_all_evaluated
accuracy_assigned_only
mean_embryo_confidence
mean_neighbor_agreement
mean_distance_score
mean_consistency_score

Metrics

The main validation metrics should include:

embryo-level accuracy
confusion matrix
per-label precision
per-label recall
number of embryos assigned
number of embryos flagged ambiguous
number of embryos flagged low_density
number of embryos not_evaluated

Accuracy should be computed at the embryo level.

The method should report accuracy in at least two ways:

accuracy_all_evaluated:
    Accuracy across all embryos that received a predicted label.
accuracy_assigned_only:
    Accuracy only among embryos with status = assigned.

This allows us to check whether confidence/status flags are useful.

For example, if accuracy_assigned_only is higher than accuracy_all_evaluated, then the status labels are helping identify less reliable predictions.

Confusion Matrix

A confusion matrix should be generated for the main phenotype categories:

Low_to_High
High_to_Low
Intermediate
Not Penetrant

This will show which phenotype categories are most often confused.

The confusion matrix should be generated for:

all evaluated embryos
assigned embryos only

This helps determine whether ambiguous and low-density flags are capturing difficult cases.

Confidence Calibration Check

The validation should also test whether confidence scores are meaningful.

Questions to ask:

Are correct predictions higher confidence than incorrect predictions?
Are low-density embryos less accurate?
Are ambiguous embryos less accurate?
Does accuracy increase as embryo_confidence increases?

Useful summaries:

mean confidence for correct predictions
mean confidence for incorrect predictions
accuracy by confidence bin
accuracy by status

This will help decide whether the confidence components and thresholds are useful.

After this, the next chunk should be **development directory and Python script organization**.

Here’s the updated Markdown chunk with feature_cols and the simple label-transfer output folded in cleanly.

## Proposed Label Transfer API
The final method should be exposed as a high-level function that takes a labeled reference dataframe and a query dataframe.
The method should be flexible enough to work with any numeric feature columns, not only embedding columns.
Example usage:
```python
from src.analyze.label_transfer import run_label_transfer
results = run_label_transfer(
    reference_df=reference_df,
    query_df=query_df,
    feature_cols=z_mu_columns,
    label_col="cluster_categories",
    embryo_col="embryo_id",
    snip_col="snip_id",
    time_col="predicted_stage_hpf",
    experiment_col="experiment_id",
    min_hpf=30,
    max_hpf=48,
    k=15,
    metric="euclidean",
    epsilon=1e-8,
)

Why feature_cols Instead of embedding_cols

The method should operate on a set of numeric features.

In the first version, these features will likely be embedding values, such as VAE latent dimensions. However, the method should not be restricted to embeddings.

Possible feature inputs could include:

VAE latent embeddings
PCA coordinates
morphology measurements
phenotype scores across time
engineered image features
combined feature sets

Therefore, the argument should be called:

feature_cols

rather than:

embedding_cols

This makes the method more general without changing the core logic.

Required Inputs

reference_df

The labeled dataframe used as the source for label transfer.

Each row should correspond to one image or observation from a labeled embryo.

The reference dataframe must contain:

embryo identifier column
image/snip identifier column
time column
label column
feature columns

query_df

The dataframe containing embryos/images that need labels assigned.

Each row should correspond to one image or observation from a query embryo.

The query dataframe must contain:

embryo identifier column
image/snip identifier column
time column
feature columns

The query dataframe does not need to contain labels.

feature_cols

A list of numeric columns used to compute distances.

The same feature columns must exist in both reference_df and query_df.

label_col

The column in reference_df containing the labels to transfer.

For the CEP290 analysis, this will initially be:

label_col = "cluster_categories"

embryo_col

The column identifying embryos.

For the current data:

embryo_col = "embryo_id"

snip_col

The column identifying individual images or observations.

For the current data:

snip_col = "snip_id"

time_col

The column containing developmental time.

For the current data:

time_col = "predicted_stage_hpf"

experiment_col

Optional column identifying the experiment or batch.

This is useful for validation and diagnostics.

For the current data:

experiment_col = "experiment_id"

This argument should be optional because future query data may not always have an experiment column.

Time Window Arguments

The method should allow filtering by developmental time.

min_hpf = 30
max_hpf = 48

The time window should be applied to both the reference dataframe and the query dataframe.

If no time filtering is desired, these arguments could be set to None.

Example:

min_hpf = None
max_hpf = None

KNN Arguments

The method should expose the main nearest-neighbor parameters.

k = 15
metric = "euclidean"
epsilon = 1e-8

k

Number of nearest reference images used for label transfer.

metric

Distance metric used to compare feature vectors.

The first version should use:

metric = "euclidean"

Other metrics could be supported later if needed.

epsilon

Small value added to distances before computing inverse-distance weights.

weight = 1 / (distance + epsilon)

This prevents division by zero when a query image exactly matches a reference image.

Suggested Return Object

The function should return a dictionary of outputs.

The full results should include intermediate tables for inspection and debugging, as well as simple label mappings for easy downstream use.

Example:

results = {
    "neighbor_long_table": neighbor_df,
    "image_label_probabilities": image_prob_df,
    "image_prediction_summary": image_summary_df,
    "embryo_label_probabilities": embryo_prob_df,
    "embryo_label_transfer_summary": embryo_summary_df,
    "embryo_pred_label_dict": embryo_pred_label_dict,
    "embryo_confidence_dict": embryo_confidence_dict,
    "embryo_status_dict": embryo_status_dict,
}

Returned Tables

neighbor_long_table

Long-format table containing the K nearest reference images for each query image.

This is the lowest-level evidence table.

image_label_probabilities

Long-format table containing distance-weighted label probabilities for each query image.

image_prediction_summary

One-row-per-image table containing the predicted image label and image-level confidence components.

embryo_label_probabilities

Long-format table containing embryo-level label probabilities after averaging image-level probabilities across each embryo.

embryo_label_transfer_summary

One-row-per-embryo table containing the final prediction, confidence scores, status, and diagnostic metadata.

This should be the main table used for downstream analysis and inspection.

Suggested columns:

query_embryo_id
n_images
min_query_hpf
max_query_hpf
query_hpf_range
predicted_label
top_label_probability
mean_image_neighbor_agreement
embryo_distance_score
embryo_consistency_score
embryo_confidence
status
status_reason

Simple Embryo Label Output

In addition to the full intermediate outputs, the method should return a simple embryo-level label mapping.

This makes it easy to transfer predicted labels back onto the query dataframe.

The full output tables are useful for inspection and debugging, but many downstream steps only need:

embryo_id -> predicted_label

Therefore, the returned results should include a simple dictionary:

embryo_pred_label_dict = {
    "embryo_001": "Low_to_High",
    "embryo_002": "Intermediate",
    "embryo_003": "Not Penetrant",
}

This dictionary should be generated from embryo_label_transfer_summary.

Adding Predicted Labels Back to the Query Dataframe

The dictionary should allow labels to be added back to the query dataframe directly:

query_df["predicted_label"] = query_df[embryo_col].map(
    results["embryo_pred_label_dict"]
)

This makes label transfer seamless for downstream analysis.

Optional Simple Confidence and Status Dictionaries

It may also be useful to return dictionaries for confidence scores and status labels.

embryo_confidence_dict = {
    "embryo_001": 0.82,
    "embryo_002": 0.64,
    "embryo_003": 0.41,
}
embryo_status_dict = {
    "embryo_001": "assigned",
    "embryo_002": "ambiguous",
    "embryo_003": "low_density",
}

These can also be added back to the query dataframe:

query_df["label_transfer_confidence"] = query_df[embryo_col].map(
    results["embryo_confidence_dict"]
)
query_df["label_transfer_status"] = query_df[embryo_col].map(
    results["embryo_status_dict"]
)

Optional Helper Function

To make the transfer even easier, the module can include a helper function:

query_df_labeled = add_label_transfer_predictions(
    query_df=query_df,
    embryo_summary_df=results["embryo_label_transfer_summary"],
    embryo_col="embryo_id",
    label_col_out="predicted_label",
    confidence_col_out="label_transfer_confidence",
    status_col_out="label_transfer_status",
)

This function would return a copy of the query dataframe with embryo-level predictions merged onto every row.

Suggested added columns:

predicted_label
label_transfer_confidence
label_transfer_status

This keeps the main method useful for both detailed inspection and simple label transfer.

Minimal Use Case

The high-level function should make the common case simple:

results = run_label_transfer(
    reference_df=reference_df,
    query_df=query_df,
    feature_cols=z_mu_columns,
    label_col="cluster_categories",
)

Then labels can be added back to the query dataframe with:

query_df["predicted_label"] = query_df["embryo_id"].map(
    results["embryo_pred_label_dict"]
)

Design Principle

The method should hide repetitive implementation details, but it should not hide the evidence.

The simple dictionaries make label transfer easy.

The intermediate tables make the predictions inspectable.

This lets the method support both quick downstream use and detailed debugging when a prediction looks strange.