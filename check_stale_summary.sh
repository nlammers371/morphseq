#!/bin/bash
# Check for stale data across YX1 experiments
# Shows which experiments need data regeneration

cd /net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground

echo "=========================================================================="
echo "STALE DATA CHECK - YX1 EXPERIMENTS"
echo "=========================================================================="
echo ""

STALE_SNIPS=()
STALE_SAM2=()

for exp_dir in raw_image_data/YX1/*/; do
    exp_id=$(basename "$exp_dir")

    stitched_dir="built_image_data/stitched_FF_images/$exp_id"
    sam2_dir="sam2_pipeline_files/exported_masks/$exp_id/masks"
    snips_dir="training_data/bf_embryo_snips/$exp_id"

    [ ! -d "$stitched_dir" ] && continue

    stitched_time=$(stat -c %Y "$stitched_dir")
    sam2_time=$(stat -c %Y "$sam2_dir" 2>/dev/null || echo "0")
    snips_time=$(stat -c %Y "$snips_dir" 2>/dev/null || echo "0")

    # Check if SAM2 masks are stale
    if [ "$sam2_time" -gt 0 ] && [ "$sam2_time" -lt "$stitched_time" ]; then
        STALE_SAM2+=("$exp_id")
    fi

    # Check if snips are stale
    if [ "$snips_time" -gt 0 ] && [ "$snips_time" -lt "$stitched_time" ]; then
        STALE_SNIPS+=("$exp_id")
    fi
done

echo "STALE SNIPS (Need Build03 --force to regenerate):"
echo "=================================================="
if [ ${#STALE_SNIPS[@]} -eq 0 ]; then
    echo "✅ None - all snips are current"
else
    for exp in "${STALE_SNIPS[@]}"; do
        echo "  ⚠️  $exp"
    done
fi
echo ""

echo "STALE SAM2 MASKS (Need SAM2 --force to regenerate):"
echo "===================================================="
if [ ${#STALE_SAM2[@]} -eq 0 ]; then
    echo "✅ None - all SAM2 masks are current"
else
    for exp in "${STALE_SAM2[@]}"; do
        echo "  ⚠️  $exp"
    done
fi
echo ""

echo "REGENERATION COMMANDS:"
echo "======================"
echo ""

if [ ${#STALE_SNIPS[@]} -gt 0 ]; then
    echo "Re-run Build03 (regenerate snips) for ${#STALE_SNIPS[@]} experiments:"
    IFS=','
    echo "python -m src.run_morphseq_pipeline.cli pipeline \\"
    echo "  --data-root morphseq_playground \\"
    echo "  --experiments ${STALE_SNIPS[*]} \\"
    echo "  --action build03 --force"
    echo ""
fi

if [ ${#STALE_SAM2[@]} -gt 0 ]; then
    echo "Re-run SAM2 (regenerate masks) for ${#STALE_SAM2[@]} experiments:"
    IFS=','
    echo "python -m src.run_morphseq_pipeline.cli pipeline \\"
    echo "  --data-root morphseq_playground \\"
    echo "  --experiments ${STALE_SAM2[*]} \\"
    echo "  --action sam2 --force"
    echo ""
fi

echo "SUMMARY:"
echo "========"
TOTAL=$(find raw_image_data/YX1 -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "Total YX1 experiments: $TOTAL"
echo "Stale SAM2 masks:     ${#STALE_SAM2[@]}"
echo "Stale snips:          ${#STALE_SNIPS[@]}"
echo ""
