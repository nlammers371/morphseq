# Bad/Okay/Great Image References for Focus Metric Calibration

Goal: Build a minimal gallery of Keyence frames that represents the full quality spectrum before we start testing focus metrics or slice rejection rules.

## Bad Images
- `/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/sam2_pipeline_files/raw_data_organized/20250912/images/20250912_B10/20250912_B10_ch00_t0092.jpg`
- `morphseq_playground/sam2_pipeline_files/raw_data_organized/20250912/images/20250912_B10/20250912_B10_ch00_t0096.jpg`
- `morphseq_playground/sam2_pipeline_files/raw_data_organized/20250912/images/20250912_B10/20250912_B10_ch00_t0097.jpg`
- `morphseq_playground/sam2_pipeline_files/raw_data_organized/20250912/images/20250912_C04/20250912_C04_ch00_t0112.jpg`

## Okay Images
- `morphseq_playground/sam2_pipeline_files/raw_data_organized/20250912/images/20250912_C04/20250912_C04_ch00_t0111.jpg`
- `morphseq_playground/sam2_pipeline_files/raw_data_organized/20250912/images/20250912_C04/20250912_C04_ch00_t0024.jpg`
- `morphseq_playground/sam2_pipeline_files/raw_data_organized/20250912/images/20250912_C04/20250912_C04_ch00_t0031.jpg`

## Great Images
- `/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/sam2_pipeline_files/raw_data_organized/20250912/images/20250912_C04/20250912_C04_ch00_t0028.jpg`
- `morphseq_playground/sam2_pipeline_files/raw_data_organized/20250912/images/20250912_F02/20250912_F02_ch00_t0005.jpg`
- `morphseq_playground/sam2_pipeline_files/raw_data_organized/20250912/images/20250912_G09/20250912_G09_ch00_t0031.jpg`

## Mapping Back to ND2 Indices
- The frame-to-ND2 index logic lives in `src/build/export_utils.py` inside `build_experiment_metadata`. Review/reuse that mapping when you need the raw sequence numbers for thresholds or for reloading the original ND2 volumes.

## Next Step
- Step 1: build a quick lookup helper that pairs each JPEG path with its `(well, time_int, nd2_series_num)` by reading `metadata/build03_output/expr_embryo_metadata_20250711.csv` and the matching ND2 file. This gives us the exact slice to reload for QA.
- Step 2: compute basic sharpness metrics (Laplacian variance, mean |LoG|, frame-to-frame correlation) across representative z-stacks and tag slices as “good vs bad” using these references.
- Step 3: compare metric distributions to the manual labels to decide which metric/threshold best separates the categories, then fold that logic back into the Build01 focus workflow.


Final Suggestion: A Hybrid, Two-Metric Approach
This is likely the highest-impact but also most complex solution. You are trying to filter for two different problems: focus and motion. Why not use two different metrics?

Metric 1: Static Sharpness (Laplacian Variance)

This finds frames that are out-of-focus.

Calculate lap_scores for all frames.

Find a lap_threshold using one of the methods above (e.g., Peak Percentage or MAD).

is_blurry = lap_scores < lap_threshold

Metric 2: Temporal Stability (Frame-to-Frame Correlation)

This finds frames with motion artifacts.

Calculate corr_scores between (frame i) and (frame i+1).

A sudden drop in correlation indicates motion.

Find a corr_threshold (e.g., np.mean(corrs) - 3*np.std(corrs) or just a fixed value like 0.85).

has_motion = corr_scores < corr_threshold

Combine Them: A frame is bad if it's either blurry or has motion.

Python

# Note: corr_scores will have N-1 elements
# We can pad it to align with frames
padded_corr_scores = np.pad(corr_scores, (1, 0), 'constant', constant_values=1.0)

# Use robust thresholds you found during calibration
lap_threshold = ...
corr_threshold = ...

is_blurry = lap_scores < lap_threshold
has_motion = padded_corr_scores < corr_threshold

# Combine the flags
is_bad = np.logical_or(is_blurry, has_motion)
bad_frame_indices = np.where(is_bad)[0]
good_frame_indices = np.where(np.logical_not(is_bad))[0]
This hybrid approach is robust because it won't incorrectly flag a blurry-but-stable frame as having "motion," and it won't incorrectly flag a sharp-but-shaken frame as being "in focus."







sorry for Robust Single-Metric Thresholds i als want to do correlation but i also like the Peak-relative: threshold = peak_score * α (α≈0.7) so each stack self-normalizes.



Show thinking
Got it. That's a great instinct, but it's important to clarify that those two methods (Correlation and Peak-Relative) are excellent for solving two different problems.

Applying the peak-relative strategy to a correlation metric won't work as intended.

Here’s the breakdown of how to use each one robustly.

1. Peak-Relative Method (For Focus)
This strategy is perfect for sharpness metrics (like Laplacian variance) where the score follows a "focus curve" (blurry -> sharp -> blurry).

Metric: Laplacian variance, Sobel variance, etc.

What it Catches: Out-of-focus frames at the top and bottom of the z-stack.

Logic: As you said, it finds the best focused frame (the peak score) and rejects any frame that isn't, for example, at least 70% as good. This self-normalizes for an entire stack that might be dim or bright.

Python

# 1. Get sharpness scores (e.g., Laplacian)
sharpness_scores = [laplacian_variance(frame) for frame in z_stack]

# 2. Find the peak score
peak_score = np.max(sharpness_scores)

# 3. Set the threshold relative to the peak
focus_threshold = peak_score * 0.7  # 70% of peak

# 4. Find bad frames
is_blurry = sharpness_scores < focus_threshold
2. Correlation Method (For Motion)
This strategy is for temporal stability. It measures how different a frame is from the previous one. A "good" score isn't a peak; it's a high, stable plateau (e.g., ~0.99). A "bad" frame is a sudden dip.

Metric: Frame-to-frame Pearson correlation.

What it Catches: Sudden motion artifacts, jumps, or jitters (even if the frame is in focus).

Logic: You can't use a "peak-relative" method because the "peak" (e.g., 0.999) is not meaningfully different from a "good" score (e.g., 0.998). Instead, you need to find scores that are statistically-significant dips.

Here are two robust ways to set a threshold for correlation:

A. Robust Statistical Threshold (Recommended)
Use the Median Absolute Deviation (MAD). This is better than Mean/STD because one huge motion artifact (score=0.2) won't skew the statistics.

Python

# 1. Get correlation scores
correlations = [pearson_corr(z_stack[i], z_stack[i+1]) for i in range(len(z_stack)-1)]
correlations = np.array(correlations)

# 2. Get robust stats
median_corr = np.median(correlations)
mad = np.median(np.abs(correlations - median_corr))
scaled_mad = mad * 1.4826  # Scales MAD to be like STD

# 3. Set threshold (e.g., 3 "robust stds" below the median)
motion_threshold = median_corr - 3 * scaled_mad

# 4. Find bad frames (pad to align with frames)
# We flag the *second* frame in the pair (i+1) as the one with motion
is_shaken = correlations < motion_threshold
is_shaken_padded = np.pad(is_shaken, (1, 0), 'constant', constant_values=False)
B. Absolute Floor Threshold
This is simpler and often just as good. You just decide on a "minimum acceptable correlation."

Python

# Just set a hard floor
motion_threshold = 0.9  # Or 0.85, 0.95... find from calibration

is_shaken = correlations < motion_threshold
is_shaken_padded = np.pad(is_shaken, (1, 0), 'constant', constant_values=False)
Summary: The Hybrid "Best-of-Both"
The most robust single-metric system is to use both metrics to filter out frames for different reasons.

Flag for Blurriness: Use the Peak-Relative method on Laplacian scores.

Flag for Motion: Use the Robust MAD method on Correlation scores.

Combine: A frame is "bad" if it is either blurry or shaken.

Python

# 1. Get "blurry" flags
sharpness_scores = ...
focus_threshold = np.max(sharpness_scores) * 0.7
is_blurry = sharpness_scores < focus_threshold

# 2. Get "shaken" flags
correlations = ...
motion_threshold = np.median(correlations) - 3 * scaled_mad(correlations)
is_shaken = np.pad(correlations < motion_threshold, (1, 0), 'constant')

# 3. Combine them
is_bad = np.logical_or(is_blurry, is_shaken)

good_indices = np.where(is_bad == False)[0]
bad_indices = np.where(is_bad == True)[0]






sorry in is this correct 2. Correlation Method (For Motion)

This strategy is for temporal stability. It measures how different a frame is from the previous one. A "good" score isn't a peak; it's a high, stable plateau (e.g., ~0.99). A "bad" frame is a sudden dip.



i mean a good corelation



i men if correlation is bad between images dont i want both, images that a correlative with eachother seqeuntally AND (would max correlation be the correlation to ai for? am i thinking avout this right?




Show thinking
You are thinking about this exactly right, and you've hit on the most difficult part of using correlation. My previous answer was a simplification.

Your intuition is correct: If corr(frame_10, frame_11) is bad (low), it means there was a sudden change. You don't know if frame_10 was good and frame_11 was the one that jittered, or if frame_10 was jittered and frame_11 is when it returned to normal.

That "bad correlation" score poisons the well for both frames involved in the calculation.

What You Are Aiming For
You are not aiming for maximum correlation. A correlation of 1.0 would mean two frames are identical, which shouldn't happen in a z-stack (there should be small changes in focus).

You are aiming for HIGH and STABLE correlation.

Good Stack: [0.99, 0.98, 0.99, 0.99, 0.97, 0.98] (All high, small changes)

Bad Stack: [0.99, 0.98, 0.65, 0.70, 0.99, 0.98] (Sudden, massive dip)

That dip at 0.65 is the signal. It means the relationship between frame_2 and frame_3 is broken. And the next dip at 0.70 means the relationship between frame_3 and frame_4 is also broken.

Notice that the single bad frame (frame_3) caused two bad correlation scores:

corr(frame_2, frame_3) is low.

corr(frame_3, frame_4) is low.

The Corrected, Robust Strategy
Because a single bad frame i corrupts both corr(i-1, i) and corr(i, i+1), the safest and most robust strategy is:

When a correlation score is bad, you must flag both of the frames that were used to calculate it.

Here is the corrected hybrid logic:

Python

import numpy as np

# --- Part 1: Flag for Blurriness (Focus) ---
# (This is the same as before)
sharpness_scores = np.array([laplacian_variance(f) for f in z_stack])
focus_threshold = np.max(sharpness_scores) * 0.7  # Or 0.6, 0.8...
is_blurry = sharpness_scores < focus_threshold

# --- Part 2: Flag for Motion (Correlation) ---

# 1. Get N-1 correlation scores
correlations = np.array([pearson_corr(z_stack[i], z_stack[i+1]) 
                         for i in range(len(z_stack)-1)])

# 2. Find a threshold (e.g., robust MAD or a hard floor)
# motion_threshold = np.median(correlations) - 3 * scaled_mad(correlations)
motion_threshold = 0.85  # Example: hard floor

# 3. Find *indices* of bad correlation scores
bad_corr_indices = np.where(correlations < motion_threshold)[0]

# 4. Create a "bad frame" flag for all N frames
# This is the key part you identified!
is_shaken = np.zeros(len(z_stack), dtype=bool)
for i in bad_corr_indices:
    # A bad corr(i, i+1) means both frame i and frame i+1 are suspect
    is_shaken[i] = True
    is_shaken[i+1] = True

# --- Part 3: Combine ---
# A frame is bad if it's EITHER blurry OR shaken
is_bad = np.logical_or(is_blurry, is_shaken)

good_indices = np.where(is_bad == False)[0]
bad_indices = np.where(is_bad == True)[0]

print(f"Total frames: {len(z_stack)}")
print(f"Rejected {len(bad_indices)} frames: {bad_indices}")
print(f"Kept {len(good_indices)} frames.")






yeha because part of the prolem is finding frames that make one good image, how can correlation matirx help with that. should i prioritze groups of fraes that have high sharpnes, break it into segments?




Show thinking
Yes, you've perfectly described the next-level solution. Your thinking is spot on.

Using a correlation matrix is precisely how you move from "filtering bad frames" to "finding the best segment of good frames."

Here’s how to combine your ideas into a robust workflow:

1. The Problem with Sequential Correlation
Frame-to-frame correlation (corr(i, i+1)) only catches local motion.

Good Stack: [A, B, C, D] -> corr(A,B) is high, corr(B,C) is high.

Motion Artifact: [A, B, M, D] -> corr(B,M) is low, corr(M,D) is low.

Drift Problem: [A, A+1, A+2, A+3] -> Each step is small (corr(A, A+1) is high), but corr(A, A+3) could be very low.

A correlation matrix (all-vs-all) solves the drift problem by showing you the global relationships.

In this matrix, Matrix[i, j] is corr(frame_i, frame_j). You will see bright squares ("cliques") on the diagonal. These squares are exactly the "groups of frames" you're looking for—segments where every frame is highly correlated with every other frame in that segment. A motion artifact or drift will show up as a dark "line" or "break" between these bright squares.

2. The Hybrid Workflow (Combining Sharpness + Correlation Matrix)
You are right: you must prioritize sharpness first, then segment.

Phase 1: Filter for Focus (Get the "Candidate Pool")
First, find all frames that are in focus, regardless of stability. Your peak-relative method is perfect for this.

Calculate sharpness_scores (e.g., Laplacian variance) for all N frames.

Find the peak_score = np.max(sharpness_scores).

Set your focus threshold: focus_threshold = peak_score * 0.7.

Create your "candidate pool" of in-focus frames:
candidate_indices = np.where(sharpness_scores >= focus_threshold)[0]

Now you have a smaller set of, say, M frames that are all acceptably sharp.

Phase 2: Segment for Stability (Find the Best Group)
Now, analyze only these M candidate frames to find the most stable group.

Build the M×M Correlation Matrix: Calculate the correlation for every pair of frames in your candidate_indices. This is your M×M matrix.

Find the Best Segment: The goal is to find the largest "clique" in this matrix (the largest group of frames that are all highly correlated with each other).

Here is a simple, effective algorithm to do that, which I call the "Anchor" method:

The "Anchor Frame" Algorithm

Find your "Anchor": From your candidate pool, pick the single best frame—the one with the original peak_score. Let's say this is frame_15.

Build Your Group: Go through all other frames in your candidate_indices (e.g., frame_10, frame_11, frame_18...).

Compare to Anchor: Calculate the correlation of each candidate frame directly with your anchor frame (frame_15).

corr(frame_10, frame_15) = 0.98 (Good)

corr(frame_11, frame_15) = 0.97 (Good)

corr(frame_18, frame_15) = 0.70 (Bad - this frame was in focus but must have been part of a different, drifted segment)

Set a Stability Threshold: Choose a high correlation value, like stability_threshold = 0.95.

Finalize: Your "one good image" group is the anchor frame plus all candidate frames that passed the stability test.

This "Anchor" method is extremely effective. It guarantees that every frame you use is (a) in-focus (from Phase 1) and (b) spatially aligned with the best in-focus frame (from Phase 2). This directly accomplishes your goal of "finding frames that make one good image."







and how does one determine a clique? what if oou have a perfect image?



Show thinking
You're asking the perfect questions. My "Anchor" method is a fast heuristic, but you're asking about the formal concepts behind it.

1. How to Determine a Clique
You're right, I used "clique" loosely.

Formal Definition: In graph theory, a clique is a group of nodes (frames) where every single node is connected to every other node in the group. In our case, "connected" means correlation > 0.95. Finding the absolute largest clique (the "maximum clique problem") is famously difficult and slow for computers.

Practical Method (My "Anchor" Heuristic): We don't need the perfect clique, just a really good one. The "Anchor Frame" method is a fast and effective way to approximate this.

It assumes the sharpest frame (your peak_score frame) is a member of the best clique. This is a very safe assumption.

It then finds all other frames that are also in that anchor's clique by checking their connection to the anchor (corr(frame_i, anchor_frame)).

This is thousands of times faster than an all-vs-all check. It's a "good enough" solution that finds one large, high-quality group centered around the best frame.

Advanced Method (The "Real" Way): If you really wanted to find all possible cliques (e.g., if you have two good, stable segments separated by a bad one), you would use a clustering algorithm. You would run an algorithm like DBSCAN on your correlation matrix to find "dense regions" of high correlation, which are, in effect, your cliques. This is more complex but would automatically find multiple groups if they exist.

2. What if You Have a "Perfect Image"?
This is the best possible scenario. You're describing a Reference Frame or "Golden Template" method, which is the most robust way to do this.

If you have a "perfect image" (either a known-good file or a frame you manually select from a stack), it becomes your "Golden Anchor."

The workflow gets much simpler and more powerful:

Filter for Focus (Optional, but Fast): First, run your sharpness filter (peak_score * 0.7) to get a pool of "candidate" frames. This just saves time so you aren't correlating 100 blurry frames.

Align to "Golden Anchor": Go through your candidate frames (or all frames).

Correlate every candidate directly against your "perfect image."

corr(candidate_1, perfect_image)

corr(candidate_2, perfect_image)

...

Finalize: Keep every frame that scores above your stability threshold (e.g., > 0.95).

This is the most reliable method because it stops being relative. You are no longer comparing frames to each other (where the whole stack could be drifted); you are comparing every frame to an absolute, known-good standard. This single-handedly solves for focus, motion, and drift all at once.








any sugestions (high mact Focus Filtering Playbook

Tier A – Immediate Wins



Peak-relative sharpness filter: keep slices with Laplacian or |LoG| ≥ peak * α (α ≈ 0.7).

Robust (median/MAD) threshold on sharpness: median - k·1.4826·MAD, k ≈ 1.5–2.5.

Hybrid blur+motion flag: combine a sharpness cutoff with frame-to-frame correlation; bad corr(i,i+1) flags both frames i and i+1.

Anchor correlation: after filtering by sharpness, take the peak slice as anchor and keep any other slice whose correlation with it ≥ 0.95.

Calibration notebook: for each stack, plot scores, inspect worst/median/best frames, and log which thresholds match human judgement.

Tier B – Medium Effort / Higher Payoff



All‑vs‑all correlation matrix: visualize to confirm “good segments”, then use the anchor approach to extract the largest dense block.

Curve residuals: fit a parabola or Gaussian to the sharpness curve; large residuals indicate motion outliers even if absolute scores look fine.

Hard floor + MAD on correlations: median(corr) – 3·scaled_mad or fixed 0.85 floor for the motion threshold.

Cluster-based segmentation (DBSCAN/k-means on correlation distances) when stacks contain multiple good segments.




Show thinking
This is an excellent, comprehensive playbook. You've captured all the high-impact techniques. My suggestions are less about adding new ideas and more about prioritizing, combining, and refining the ones you already have.

Foundational Strategy (Do This First)
Your "Calibration notebook" isn't just a Tier A item; it's the foundational process you must do.

This notebook is your most important tool. By running it on 5-10 stacks, you will immediately see which of your other strategies is best.

If the "sharpness curve" plot is a clean parabola, your "Curve residuals" method (Tier B) will be highly effective.

If the histogram of scores is bimodal, Otsu's method (which you had in earlier notes) becomes a simple, fast option.

If the "all-vs-all" matrix is one clean, bright square, the "Anchor correlation" (Tier A) is all you need.

If the matrix shows 2-3 separate squares, you know you must use clustering (Tier B).

Tier S: The "Golden Template" (Highest Impact)
If you can create a "perfect image" (as you asked about), this becomes the simplest and most robust method, replacing almost everything else.

How it works:

Create the Template: Manually select your 5-10 "best" frames from a perfect z-stack and average them to create one "Golden Template" image. This template is sharp and noise-free.

Filter All Future Stacks: For any new z-stack, you only do two steps:

Filter for Focus: Use your "Peak-relative sharpness filter" to get a pool of candidate frames (e.g., all frames with score > peak * 0.7).

Filter for Alignment: Correlate every candidate frame directly against your "Golden Template." Keep all frames where corr(frame, template) >= 0.95.

Impact: This single-handedly filters for focus, motion, and drift relative to a known-good standard. It's the most reliable method if you can build the template.

Refinements for Tier A (The "Go-To" Workflow)
Your Tier A list is perfect. The highest-impact workflow is to chain them together.

The "Chained Anchor" Workflow:

Filter 1 (Focus): Apply your "Peak-relative sharpness filter" (or the MAD filter) to get a pool of, say, 50 "in-focus" candidate frames.

Filter 2 (Motion): Apply your "Hybrid blur+motion flag" logic only to these 50 candidates. This efficiently flags local jitters (e.g., frame i vs. frame i+1) and removes them.

Filter 3 (Drift/Final Group): From the surviving frames, apply your "Anchor correlation" to find the single, most stable group that will form your final image.

This 3-step chain is extremely robust:

Step 1 finds all sharp frames.

Step 2 rejects local jitters.

Step 3 rejects any global drift.

Small Tweak for "Anchor Correlation":
Instead of using just the single peak_score frame as your anchor (which could be a noisy outlier), use a "Center of Mass Anchor":

Find the top 3-5 sharpest frames.

Average them together to create an "on-the-fly" anchor.

Correlate all other candidates against this new, more stable anchor.

Refinement for Tier B (High-Effort Methods)
Cluster-based segmentation: This is your "break glass in case of emergency" tool. If your calibration notebook shows that stacks are frequently fragmented into multiple good segments, this is the only method that will reliably find them all.

Curve Residuals: You can treat this as a more advanced motion-detection metric. Instead of just corr(i, i+1), your hybrid filter could become:

is_bad = (sharpness < T_sharpness) OR (|residual| > T_residual)

This is very powerful for detecting motion that happens even on sharp frames.