# Stream C: Visualization Contract — Decisions

## NaN masking
All viz functions apply NaN masking: non-support pixels get NaN, not zero.
This prevents confusion between "no data" and "zero motion/mass".

## apply_nan_mask utility
Centralizes the NaN masking logic. Takes a field and support_mask, returns field with NaN outside support.

## 4-panel plot_uot_summary
Panel 1: support mask, Panel 2: velocity quiver, Panel 3: creation heatmap, Panel 4: destruction heatmap.
All panels show numeric annotations (total mass, max velocity, etc.).

## Backward compatibility
Existing functions (plot_creation_destruction, plot_velocity_overlay, etc.) are preserved.
New functions are additions, not replacements.
