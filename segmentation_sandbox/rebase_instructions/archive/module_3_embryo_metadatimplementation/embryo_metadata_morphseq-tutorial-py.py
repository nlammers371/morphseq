#!/usr/bin/env python
"""
MorphSeq Annotation Pipeline Tutorial
=====================================
Welcome to the wonderful world of embryo annotation! 
This tutorial will guide you through the EmbryoMetadata (EM) class - your new best friend for managing zebrafish data.
"""

#NOTE for @AIAGENT, this needs to be converted in to a ipynb file. 


# %% [markdown]
# # MorphSeq Annotation Pipeline Tutorial
# 
# ## Prerequisites
# - SAM annotation file (`*_sam_annotations.json`)
# - Python environment with required packages
# - Basic understanding of zebrafish embryo experiments

# %% [markdown]
# ## 1. Setup and Initialization

# %%
# 1.1 Import Required Modules
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Import our heroes
from streamlined_embryo_metadata import EmbryoMetadata
from annotation_batch import AnnotationBatch
from schema_manager import SchemaManager

# For visualization (optional but fun!)
import pandas as pd
import matplotlib.pyplot as plt

print("🐠 Welcome to MorphSeq! Let's annotate some embryos!")

# %%
# 1.2 Load SAM Annotations and Initialize Metadata

# Pro tip: Always use Path objects for file paths - they're cross-platform!
sam_path = Path("data/experiment_001_sam_annotations.json")
metadata_path = Path("data/experiment_001_embryo_metadata.json")

# Initialize metadata - this is where the magic begins
metadata = EmbryoMetadata(
    sam_annotation_path=sam_path,
    embryo_metadata_path=metadata_path,
    gen_if_no_file=True,  # Create fresh metadata if none exists
    verbose=True  # We like to see what's happening!
)

print(f"✅ Loaded metadata with {metadata.embryo_count} embryos and {metadata.snip_count} snips")

# %%
# 1.3 Explore Data Structure

# Let's peek under the hood
entity_counts = metadata.get_entity_counts()
print("📊 Entity breakdown:")
for entity_type, count in entity_counts.items():
    print(f"  - {entity_type}: {count}")

# Show a sample embryo structure
sample_embryo_id = list(metadata.data["embryos"].keys())[0]
print(f"\n🔍 Sample embryo structure ({sample_embryo_id}):")
print(json.dumps(metadata.data["embryos"][sample_embryo_id], indent=2)[:500] + "...")

# List available experiments/videos
print("\n📹 Available experiments:")
# Hint: Entity IDs contain hierarchy info - parse them!
experiments = set()
for embryo_id in metadata.data["embryos"]:
    exp_id = embryo_id.split("_")[0]  # exp001_v001_e001 -> exp001
    experiments.add(exp_id)
print(f"Found experiments: {sorted(experiments)}")

# %% [markdown]
# ## 🎯 Exercise 1: Entity Explorer
# Write a function to extract and count all unique videos from embryo IDs
# Hint: Video ID is the second component (e.g., exp001_v001_e001 -> v001)

# %%
def count_videos_per_experiment(metadata):
    """Your code here - extract video counts per experiment"""
    # TODO: Implement this function
    pass

# Solution (hidden - try first!):
# video_counts = {}
# for embryo_id in metadata.data["embryos"]:
#     parts = embryo_id.split("_")
#     exp_id, video_id = parts[0], parts[1]
#     ...

# %% [markdown]
# ## 2. Understanding the Data Hierarchy

# %%
# 2.1 Entity ID Conventions

# The Rosetta Stone of MorphSeq IDs
sample_ids = {
    "experiment": "exp001",
    "video": "exp001_v001", 
    "embryo": "exp001_v001_e001",
    "snip": "exp001_v001_e001_s0250"  # s0250 = frame 250
}

print("🔤 ID Convention Decoder:")
for entity_type, example_id in sample_ids.items():
    print(f"{entity_type:12} → {example_id}")
    
# Parse a complex ID
complex_id = "exp003_v002_e005_s0450"
parts = complex_id.split("_")
print(f"\n🧩 Parsing '{complex_id}':")
print(f"  Experiment: {parts[0]}")
print(f"  Video:      {parts[1]}")
print(f"  Embryo:     {parts[2]}")
print(f"  Frame:      {int(parts[3][1:])}")  # Remove 's' prefix

# %%
# 2.2 Navigate Embryo-Snip Relationships

# Get snips for an embryo
embryo_id = "exp001_v001_e001"  # Adjust to your data
snips = metadata.get_available_snips(embryo_id)
print(f"📸 Embryo {embryo_id} has {len(snips)} snips:")
print(f"  First 5: {snips[:5]}")

# Get embryo from snip
if snips:
    sample_snip = snips[0]
    parent_embryo = metadata.get_embryo_id_from_snip(sample_snip)
    print(f"\n🔗 Snip {sample_snip} belongs to embryo {parent_embryo}")

# Visualize temporal sequence
if len(snips) > 10:
    # Extract frame numbers
    frame_numbers = []
    for snip in snips[:20]:  # First 20 for visualization
        frame_num = int(snip.split("_s")[1])
        frame_numbers.append(frame_num)
    
    plt.figure(figsize=(10, 3))
    plt.plot(frame_numbers, 'o-')
    plt.xlabel("Snip Index")
    plt.ylabel("Frame Number")
    plt.title(f"Temporal Sequence for {embryo_id}")
    plt.show()

# %% [markdown]
# ## 3. Basic Annotations
# Time to add some data! Remember: Phenotype → Genotype → Treatment

# %%
# 3.1 Adding Phenotypes

# List available phenotypes from schema
phenotypes = metadata.schema_manager.get_phenotypes()
print("🔬 Available phenotypes:")
for name, info in phenotypes.items():
    print(f"  - {name}: {info['description']}")

# Add phenotype to specific frames
embryo_id = list(metadata.data["embryos"].keys())[0]
snips = metadata.get_available_snips(embryo_id)

# Add to frames 200-400
metadata.add_phenotype(
    snips[0],  # Use actual snip ID
    phenotype="EDEMA",
    author="tutorial_user",
    notes="Mild pericardial edema observed",
    confidence=0.85
)
print("✅ Added EDEMA phenotype!")

# Add phenotype to all frames of an embryo
# Pro tip: Use batch operations for this - more efficient!
batch = AnnotationBatch("tutorial_user", "Phenotype all frames demo")
batch.add_phenotype(embryo_id, "BODY_AXIS_DEFECT", frames="all")
results = metadata.apply_batch(batch)
print(f"✅ Applied phenotype to all frames: {results['applied']} annotations")

# %%
# 3.2 The DEAD Phenotype Rule

# The DEAD phenotype is special - it's exclusive and temporal!
print("☠️ Understanding DEAD phenotype rules:")

# Mark embryo as dead from frame 350
dead_embryo = list(metadata.data["embryos"].keys())[1]  # Use second embryo
metadata.mark_dead(dead_embryo, start_frame=350, author="tutorial_user")
print(f"✅ Marked {dead_embryo} as DEAD from frame 350")

# Try to add another phenotype (this will fail!)
try:
    dead_snip = metadata.get_available_snips(dead_embryo)[-1]  # Last snip
    metadata.add_phenotype(
        dead_snip,
        phenotype="EDEMA",
        author="tutorial_user"
    )
except ValueError as e:
    print(f"❌ Expected error: {e}")
    print("   → DEAD embryos can't have other phenotypes!")

# Force override if really needed (use sparingly!)
metadata.add_phenotype(
    dead_snip,
    phenotype="EDEMA",
    author="tutorial_user",
    force_dead=True  # Override DEAD exclusivity
)
print("⚠️  Forced phenotype addition (not recommended in practice!)")

# %%
# 3.3 Single Embryo Genotyping

# List available genotypes
genotypes = metadata.schema_manager.get_genotypes()
print("🧬 Available genotypes:")
for name, info in genotypes.items():
    print(f"  - {name}: {info.get('description', 'No description')}")

# Add genotype to embryo
target_embryo = list(metadata.data["embryos"].keys())[2]
metadata.add_genotype(
    target_embryo,
    gene_name="tmem67",
    allele="sa1423",
    zygosity="homozygous",
    author="tutorial_user"
)
print(f"✅ Added tmem67 genotype to {target_embryo}")

# Try to add another genotype (will fail - single genotype rule!)
try:
    metadata.add_genotype(
        target_embryo,
        gene_name="lmx1b",
        allele="sa2187"
    )
except ValueError as e:
    print(f"❌ Expected error: {e}")
    print("   → Use overwrite_genotype=True to change existing genotype")

# Check genotype coverage
coverage = metadata.validate_genotype_coverage()
print(f"\n📊 Genotype coverage: {coverage['missing_count']} embryos missing genotypes")

# %%
# 3.4 Working with Treatments

# Available treatments
treatments = metadata.schema_manager.get_treatments()
print("💊 Available treatments:")
for name, info in treatments.items():
    print(f"  - {name}: {info['description']} ({info['type']})")

# Single treatment
treated_embryo = list(metadata.data["embryos"].keys())[3]
metadata.add_treatment(
    treated_embryo,
    treatment_name="PTU",
    dosage="200μM",
    timing="24hpf-48hpf",
    author="tutorial_user",
    notes="Standard pigmentation blocking"
)
print(f"✅ Added PTU treatment to {treated_embryo}")

# Multiple treatments (compound experiment)
metadata.add_treatment(
    treated_embryo,
    treatment_name="heat_shock",
    timing="30hpf for 1hr at 37°C",
    author="tutorial_user"
)
print("✅ Added second treatment - compound experiment!")

# Treatment with vehicle control
control_embryo = list(metadata.data["embryos"].keys())[4]
metadata.add_treatment(
    control_embryo,
    treatment_name="DMSO",
    dosage="0.1%",
    timing="24hpf-48hpf",
    author="tutorial_user",
    notes="Vehicle control for PTU experiment"
)
print(f"✅ Added DMSO control to {control_embryo}")

# %% [markdown]
# ## 🎯 Exercise 2: Annotation Detective
# Find all embryos that have both a genotype AND a phenotype
# Bonus: Count how many have treatments too!

# %%
def find_fully_annotated_embryos(metadata):
    """Find embryos with both genotype and phenotype annotations"""
    # TODO: Your code here
    # Hint: Check if embryo has non-None genotype and non-empty phenotypes in any snip
    pass

# Test your function:
# annotated = find_fully_annotated_embryos(metadata)
# print(f"Found {len(annotated)} fully annotated embryos")

# %% [markdown]
# ## 4. Batch Annotations
# The power move - annotate multiple embryos at once!

# %%
# 4.1 Create Annotation Batch

# Batch operations are your friend for large-scale annotation
batch = AnnotationBatch(
    author="tutorial_user",
    description="Demo batch for multiple embryos"
)

# Add multiple annotations efficiently
embryo_list = list(metadata.data["embryos"].keys())[:5]  # First 5 embryos

for i, embryo_id in enumerate(embryo_list):
    # Alternate between two genotypes
    genotype = "tmem67" if i % 2 == 0 else "WT"
    batch.add_genotype(embryo_id, genotype, zygosity="homozygous")
    
    # Add phenotypes based on genotype
    if genotype == "tmem67":
        batch.add_phenotype(embryo_id, "EDEMA", frames="0200:0400")
        batch.add_phenotype(embryo_id, "CONVERGENCE_EXTENSION", frames="0100:0300")
    
    # Add QC flags
    batch.add_flag(embryo_id, "MOTION_BLUR", level="snip", priority="medium")

# Preview batch contents
print("📦 Batch preview:")
print(batch.preview())

# %%
# 4.2 Validate Before Applying

# Always validate first! Save yourself from errors
validation_results = metadata.apply_batch(batch, dry_run=True)

if validation_results['errors']:
    print(f"❌ Found {len(validation_results['errors'])} validation errors:")
    for error in validation_results['errors'][:3]:  # Show first 3
        print(f"  - {error['embryo']}: {error['error']}")
        print(f"    Suggestion: {error['suggestion']}")
else:
    print("✅ Batch validation passed!")

# Frame resolution check
print(f"\n🎯 Frame resolutions: {len(validation_results['frame_resolutions'])} operations")

# %%
# 4.3 Apply Batch

# Apply for real if validation passed
if not validation_results['errors']:
    results = metadata.apply_batch(batch, dry_run=False)
    print(f"✅ Successfully applied {results['applied']} annotations!")
    
    # Save your work!
    metadata.save(backup=True)
    print("💾 Metadata saved with backup")
else:
    # Fix errors interactively
    print("\n🔧 Let's fix those errors...")
    # Example fix: Remove conflicting annotations
    for error in validation_results['errors']:
        if "already has genotype" in error['error']:
            embryo_id = error['embryo']
            batch.remove_annotation(embryo_id, "genotype")
            print(f"  Removed genotype conflict for {embryo_id}")

# %% [markdown]
# ## 5. Advanced Frame Specifications
# Master the art of temporal annotation!

# %%
# 5.1 Frame Range Syntax

embryo_id = list(metadata.data["embryos"].keys())[0]
demo_batch = AnnotationBatch("tutorial_user", "Frame syntax demo")

# Index-based ranges (by position in snip list)
demo_batch.add_phenotype(embryo_id, "EDEMA", frames="[0:5]")  # First 5 snips
print("✅ Added phenotype to snips by index [0:5]")

# Frame-based ranges (by actual frame numbers)
demo_batch.add_phenotype(embryo_id, "BODY_AXIS_DEFECT", frames="0200:0400")
print("✅ Added phenotype to frames 200-400")

# Step syntax (every Nth frame)
demo_batch.add_phenotype(embryo_id, "CONVERGENCE_EXTENSION", frames="0100:0500:50")
print("✅ Added phenotype every 50 frames from 100-500")

# Apply and see results
results = metadata.apply_batch(demo_batch, dry_run=True)
print(f"\n📊 Frame resolution summary:")
for key, snips in results['frame_resolutions'].items():
    print(f"  {key}: {len(snips)} snips")

# %%
# 5.2 Complex Frame Patterns

# Pro tip: You can use list comprehensions for complex patterns!
snips = metadata.get_available_snips(embryo_id)

# Select snips with specific criteria
from parsing_utils import extract_frame_number

# Only early timepoints (frames < 200)
early_snips = [s for s in snips if extract_frame_number(s) < 200]

# Only every 10th frame
periodic_snips = [s for s in snips if extract_frame_number(s) % 10 == 0]

# Custom selection based on your experiment
critical_frames = [100, 150, 200, 300, 400]  # Your important timepoints
critical_snips = [s for s in snips if extract_frame_number(s) in critical_frames]

print(f"🎯 Custom frame selections:")
print(f"  Early development: {len(early_snips)} snips")
print(f"  Periodic sampling: {len(periodic_snips)} snips")
print(f"  Critical timepoints: {len(critical_snips)} snips")

# %% [markdown]
# ## 🎯 Exercise 3: Temporal Annotator
# Create a function that adds different phenotypes to different developmental stages
# Early (0-200): CONVERGENCE_EXTENSION, Late (400+): EDEMA

# %%
def stage_specific_annotation(metadata, embryo_id, author):
    """Add phenotypes based on developmental stage"""
    # TODO: Your code here
    # Hint: Create a batch, use frame ranges for each stage
    pass

# %% [markdown]
# ## 6. Querying and Analysis
# Find the needles in your embryo haystack!

# %%
# 6.1 Simple Queries

# Find embryos by phenotype
edema_snips = metadata.find_embryos_with_phenotype("EDEMA")
print(f"🔍 Found {len(edema_snips)} snips with EDEMA phenotype")

# Find embryos by genotype  
tmem67_embryos = metadata.find_embryos_with_genotype("tmem67")
print(f"🧬 Found {len(tmem67_embryos)} embryos with tmem67 genotype")

# Find flagged data
motion_blur_flags = metadata.find_embryos_with_flag("MOTION_BLUR", level="snip")
print(f"🚩 Found motion blur in: {list(motion_blur_flags.keys())[:3]}...")  # First 3

# %%
# 6.2 Query Builder

# Build complex queries by chaining conditions
query = metadata.query()

# Find tmem67 mutants with edema
results = (query
    .genotype("tmem67")
    .phenotype("EDEMA")
    .execute())

print(f"🎯 Complex query: {len(results)} tmem67 embryos with EDEMA")

# Query with frame constraints
early_affected = (metadata.query()
    .genotype("tmem67")
    .phenotype("CONVERGENCE_EXTENSION")
    .frame_range(start=0, end=300)
    .execute())

print(f"⏰ Early phenotypes: {len(early_affected)} embryos affected before frame 300")

# Convert query to batch for further work
followup_batch = query.to_batch("analysis_user")
print(f"📦 Created batch from query with {len(followup_batch.data)} embryos")

# %%
# 6.3 Query by Treatment

# Find all treated embryos
ptu_embryos = metadata.find_embryos_with_treatment("PTU")
dmso_embryos = metadata.find_embryos_with_treatment("DMSO")

print(f"💊 Treatment groups:")
print(f"  PTU: {len(ptu_embryos)} embryos")
print(f"  DMSO (control): {len(dmso_embryos)} embryos")

# Compare treated vs control
# This is where you'd do statistical analysis!
treated_with_phenotype = set(ptu_embryos) & set(e for e in tmem67_embryos 
                                                if any(metadata.get_phenotypes(s) 
                                                      for s in metadata.get_available_snips(e)))
print(f"📊 PTU-treated with phenotypes: {len(treated_with_phenotype)}")

# %%
# 6.4 Statistics and Summaries

# Get overall statistics
summary = metadata.get_summary()

print("📈 Metadata Summary:")
print(f"  Entities: {summary['entity_counts']}")
print(f"  Top phenotypes: {dict(sorted(summary['phenotype_stats'].items(), 
                                      key=lambda x: x[1], reverse=True)[:5])}")
print(f"  Genotype distribution: {summary['genotype_stats']}")

# Create a simple visualization
if summary['phenotype_stats']:
    phenotypes = list(summary['phenotype_stats'].keys())
    counts = list(summary['phenotype_stats'].values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(phenotypes, counts)
    plt.xlabel("Phenotype")
    plt.ylabel("Count")
    plt.title("Phenotype Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 🎯 Exercise 4: Data Analyzer
# Create a summary report showing phenotype penetrance by genotype
# (What % of each genotype shows each phenotype?)

# %%
def calculate_penetrance(metadata):
    """Calculate phenotype penetrance by genotype"""
    # TODO: Your code here
    # Hint: Group embryos by genotype, count phenotypes, calculate percentages
    pass

# %% [markdown]
# ## 7. Quality Control
# Keep your data clean and trustworthy!

# %%
# 7.1 Adding Flags

# Different flag levels for different problems
sample_embryo = list(metadata.data["embryos"].keys())[0]
sample_snip = metadata.get_available_snips(sample_embryo)[0]

# Snip-level flag (specific frame issue)
metadata.add_flag(
    sample_snip,
    flag_type="MOTION_BLUR",
    level="snip",  # Auto-detected from ID format
    priority="medium",
    description="Tail movement during acquisition"
)

# Video-level flag (affects all embryos in video)
metadata.add_flag(
    "exp001_v001",  # Video ID
    flag_type="NONZERO_SEED_FRAME",
    level="video",
    priority="high",
    description="Recording started late"
)

# Show available flag types
for level in ["snip", "video", "image", "experiment"]:
    flags = metadata.schema_manager.get_flags_for_level(level)
    if flags:
        print(f"\n🚩 {level.title()}-level flags:")
        for flag, info in flags.items():
            print(f"  - {flag} ({info['priority']}): {info['description']}")

# %%
# 7.2 Data Validation

# Check DEAD consistency
dead_conflicts = metadata.validate_data_integrity()['dead_conflicts']
if dead_conflicts['conflicts']:
    print(f"⚠️  Found {dead_conflicts['conflict_count']} DEAD conflicts!")
    for conflict in dead_conflicts['conflicts'][:2]:
        print(f"  - {conflict['embryo_id']}: DEAD + {conflict['conflicting_phenotypes']}")
else:
    print("✅ No DEAD phenotype conflicts found")

# Check temporal violations
if dead_conflicts['temporal_violations']:
    print(f"\n⏰ Found {dead_conflicts['temporal_violation_count']} temporal violations!")
    for violation in dead_conflicts['temporal_violations'][:2]:
        print(f"  - {violation['embryo_id']}: {violation['message']}")

# Find missing genotypes
missing_genos = metadata.validate_genotype_coverage()
if missing_genos['missing_count'] > 0:
    print(f"\n🧬 Missing genotypes: {missing_genos['missing_count']} embryos")
    print(f"  Affected experiments: {missing_genos['affected_experiments']}")

# Validate entity hierarchy
entity_counts = metadata.get_entity_counts()
print(f"\n🏗️ Entity hierarchy validated:")
for entity, count in entity_counts.items():
    print(f"  - {entity}: {count}")

# %% [markdown]
# ## 8. Export and Reporting

# %%
# 8.1 Export Formats

# JSON export (full data)
export_path = Path("exports/tutorial_metadata.json")
export_path.parent.mkdir(exist_ok=True)
metadata.save_json(metadata.data, export_path)
print(f"💾 Exported full metadata to {export_path}")

# CSV summary for analysis
summary_data = []
for embryo_id, embryo_data in metadata.data["embryos"].items():
    phenotypes = []
    for snip_id, snip_data in embryo_data["snips"].items():
        if "phenotype" in snip_data:
            phenotypes.append(snip_data["phenotype"]["value"])
    
    summary_data.append({
        "embryo_id": embryo_id,
        "genotype": embryo_data.get("genotype", {}).get("value", "TBD"),
        "phenotype_count": len(set(phenotypes)),
        "unique_phenotypes": ",".join(set(phenotypes)),
        "treatment": list(embryo_data.get("treatments", {}).values())[0]["value"] 
                    if embryo_data.get("treatments") else "None"
    })

df = pd.DataFrame(summary_data)
csv_path = Path("exports/embryo_summary.csv")
df.to_csv(csv_path, index=False)
print(f"📊 Exported summary CSV to {csv_path}")

# Batch export for reuse
reuse_batch = AnnotationBatch("export_user", "Exported annotations")
# Add current annotations to batch
for embryo_id in list(metadata.data["embryos"].keys())[:3]:  # First 3 as example
    genotype = metadata.get_genotype(embryo_id)
    if genotype:
        reuse_batch.add_genotype(
            embryo_id, 
            genotype["value"],
            allele=genotype.get("allele"),
            zygosity=genotype.get("zygosity", "heterozygous")
        )

batch_path = Path("exports/annotation_batch.json")
reuse_batch.save_json(batch_path)
print(f"📦 Exported reusable batch to {batch_path}")

# %%
# 8.2 Generate Reports

# Experiment summary report
print("\n📋 EXPERIMENT SUMMARY REPORT")
print("=" * 50)

# Group by experiment
experiments = {}
for embryo_id in metadata.data["embryos"]:
    exp_id = embryo_id.split("_")[0]
    if exp_id not in experiments:
        experiments[exp_id] = {"embryos": 0, "genotypes": set(), "phenotypes": set()}
    
    experiments[exp_id]["embryos"] += 1
    
    genotype = metadata.get_genotype(embryo_id)
    if genotype:
        experiments[exp_id]["genotypes"].add(genotype["value"])
    
    for snip in metadata.get_available_snips(embryo_id):
        phenos = metadata.get_phenotypes(snip)
        for p in phenos:
            experiments[exp_id]["phenotypes"].add(p["value"])

for exp_id, data in experiments.items():
    print(f"\n{exp_id}:")
    print(f"  Embryos: {data['embryos']}")
    print(f"  Genotypes: {', '.join(data['genotypes']) or 'None'}")
    print(f"  Phenotypes: {', '.join(data['phenotypes']) or 'None'}")

# Phenotype report by genotype
print("\n📊 PHENOTYPE PENETRANCE BY GENOTYPE")
print("=" * 50)

genotype_phenotype_map = {}
for embryo_id, embryo_data in metadata.data["embryos"].items():
    genotype = embryo_data.get("genotype", {}).get("value", "TBD")
    
    if genotype not in genotype_phenotype_map:
        genotype_phenotype_map[genotype] = {"total": 0, "phenotypes": {}}
    
    genotype_phenotype_map[genotype]["total"] += 1
    
    # Collect phenotypes
    embryo_phenotypes = set()
    for snip_id, snip_data in embryo_data["snips"].items():
        if "phenotype" in snip_data:
            embryo_phenotypes.add(snip_data["phenotype"]["value"])
    
    for phenotype in embryo_phenotypes:
        if phenotype not in genotype_phenotype_map[genotype]["phenotypes"]:
            genotype_phenotype_map[genotype]["phenotypes"][phenotype] = 0
        genotype_phenotype_map[genotype]["phenotypes"][phenotype] += 1

for genotype, data in genotype_phenotype_map.items():
    print(f"\n{genotype} (n={data['total']}):")
    for phenotype, count in data["phenotypes"].items():
        penetrance = (count / data["total"]) * 100
        print(f"  - {phenotype}: {count}/{data['total']} ({penetrance:.1f}%)")

# %% [markdown]
# ## 9. Common Workflows

# %%
# 9.1 Initial Experiment Annotation

def initial_annotation_workflow(sam_path, author):
    """Complete workflow from SAM to annotated metadata"""
    
    print("🚀 Starting initial annotation workflow...")
    
    # Step 1: Initialize metadata
    metadata = EmbryoMetadata(
        sam_annotation_path=sam_path,
        gen_if_no_file=True,
        verbose=True
    )
    print(f"✅ Loaded {metadata.embryo_count} embryos")
    
    # Step 2: Create annotation batch
    batch = AnnotationBatch(author, "Initial experiment annotation")
    
    # Step 3: Assign genotypes (usually from crossing scheme)
    embryo_ids = list(metadata.data["embryos"].keys())
    
    # Example: Alternating WT and mutant (adjust to your experiment)
    for i, embryo_id in enumerate(embryo_ids):
        genotype = "WT" if i % 3 == 0 else "tmem67"  # 1:2 ratio
        batch.add_genotype(embryo_id, genotype)
    
    # Step 4: Add known treatments
    # Example: First video is treated
    treated_embryos = [e for e in embryo_ids if "_v001_" in e]
    for embryo_id in treated_embryos:
        batch.add_treatment(embryo_id, "PTU", dosage="200μM", timing="24-48hpf")
    
    # Step 5: Apply and save
    results = metadata.apply_batch(batch)
    metadata.save()
    
    print(f"✅ Workflow complete: {results['applied']} annotations applied")
    return metadata

# Demo the workflow
# metadata = initial_annotation_workflow(sam_path, "tutorial_user")

# %%
# 9.2 Phenotype Screening

def phenotype_screening_workflow(metadata, embryo_list, author):
    """Systematic phenotyping workflow"""
    
    print("🔬 Starting phenotype screening workflow...")
    
    # Step 1: Quality check
    for embryo_id in embryo_list:
        snips = metadata.get_available_snips(embryo_id)
        if len(snips) < 50:
            print(f"⚠️  Warning: {embryo_id} has only {len(snips)} snips")
    
    # Step 2: Stage-specific screening
    screening_batch = AnnotationBatch(author, "Phenotype screening")
    
    for embryo_id in embryo_list:
        # Early development (0-300 frames)
        screening_batch.add_phenotype(
            embryo_id, 
            "CONVERGENCE_EXTENSION",
            frames="0000:0300",
            notes="Check for C&E defects"
        )
        
        # Mid development (300-600 frames)
        screening_batch.add_phenotype(
            embryo_id,
            "BODY_AXIS_DEFECT", 
            frames="0300:0600",
            notes="Assess body curvature"
        )
        
        # Late development (600+ frames)
        screening_batch.add_phenotype(
            embryo_id,
            "EDEMA",
            frames="0600:",
            notes="Check for pericardial/yolk edema"
        )
    
    # Step 3: Validate and apply
    validation = metadata.apply_batch(screening_batch, dry_run=True)
    if not validation['errors']:
        results = metadata.apply_batch(screening_batch)
        print(f"✅ Screening complete: {results['applied']} phenotypes recorded")
    else:
        print(f"❌ Validation errors: {len(validation['errors'])}")
    
    return screening_batch

# Example usage
# embryos_to_screen = metadata.find_embryos_with_genotype("tmem67")[:5]
# screening_batch = phenotype_screening_workflow(metadata, embryos_to_screen, "screener")

# %%
# 9.3 Quality Review

def quality_review_workflow(metadata):
    """Review and correct annotations"""
    
    print("🔍 Starting quality review workflow...")
    
    # Step 1: Check for common issues
    issues = {
        "missing_genotypes": [],
        "dead_conflicts": [],
        "suspicious_patterns": [],
        "flagged_data": []
    }
    
    # Find missing genotypes
    for embryo_id in metadata.data["embryos"]:
        if not metadata.get_genotype(embryo_id):
            issues["missing_genotypes"].append(embryo_id)
    
    # Check DEAD conflicts
    integrity = metadata.validate_data_integrity()
    issues["dead_conflicts"] = integrity["dead_conflicts"]["conflicts"]
    
    # Find suspicious patterns (e.g., all embryos dead)
    for exp_id in set(e.split("_")[0] for e in metadata.data["embryos"]):
        exp_embryos = [e for e in metadata.data["embryos"] if e.startswith(exp_id)]
        dead_count = sum(1 for e in exp_embryos 
                        if any("DEAD" in str(metadata.get_phenotypes(s)) 
                              for s in metadata.get_available_snips(e)))
        if len(exp_embryos) > 0 and dead_count / len(exp_embryos) > 0.8:
            issues["suspicious_patterns"].append({
                "experiment": exp_id,
                "issue": f"{dead_count}/{len(exp_embryos)} embryos dead (>80%)"
            })
    
    # Find high-priority flags
    for embryo_id in metadata.data["embryos"]:
        flags = metadata.data["embryos"][embryo_id].get("flags", {})
        high_priority = [f for f in flags if f.get("priority") in ["high", "critical"]]
        if high_priority:
            issues["flagged_data"].append({
                "embryo": embryo_id,
                "flags": high_priority
            })
    
    # Step 2: Generate review report
    print("\n📋 QUALITY REVIEW REPORT")
    print("=" * 50)
    
    total_issues = sum(len(v) if isinstance(v, list) else 0 for v in issues.values())
    print(f"Total issues found: {total_issues}")
    
    if issues["missing_genotypes"]:
        print(f"\n❌ Missing genotypes ({len(issues['missing_genotypes'])}):")
        for embryo in issues["missing_genotypes"][:5]:
            print(f"  - {embryo}")
        if len(issues["missing_genotypes"]) > 5:
            print(f"  ... and {len(issues['missing_genotypes']) - 5} more")
    
    if issues["dead_conflicts"]:
        print(f"\n⚠️  DEAD conflicts ({len(issues['dead_conflicts'])}):")
        for conflict in issues["dead_conflicts"][:3]:
            print(f"  - {conflict['embryo_id']}: DEAD + {conflict['conflicting_phenotypes']}")
    
    if issues["suspicious_patterns"]:
        print(f"\n🚨 Suspicious patterns:")
        for pattern in issues["suspicious_patterns"]:
            print(f"  - {pattern['experiment']}: {pattern['issue']}")
    
    return issues

# Run quality review
# issues = quality_review_workflow(metadata)

# %% [markdown]
# ## 🎯 Exercise 5: Workflow Master
# Create a custom workflow that:
# 1. Finds all WT embryos
# 2. Checks if any show phenotypes
# 3. Flags unexpected phenotypes in WT
# Hint: WT embryos shouldn't show strong phenotypes!

# %%
def wild_type_phenotype_check(metadata, author):
    """Check for unexpected phenotypes in WT embryos"""
    # TODO: Your code here
    # Steps:
    # 1. Find all WT embryos
    # 2. Check each for phenotypes
    # 3. Flag any with unexpected phenotypes
    # 4. Generate summary report
    pass

# %% [markdown]
# ## 10. Troubleshooting

# %%
# 10.1 Common Errors and Solutions

print("🔧 Common Errors and Solutions")
print("=" * 50)

# Error 1: Frame resolution failures
print("\n1️⃣ Frame Resolution Failures")
try:
    batch = AnnotationBatch("demo", "Frame error demo")
    batch.add_phenotype("fake_embryo_id", "EDEMA", frames="9999:")  # Non-existent frame
    metadata.apply_batch(batch)
except Exception as e:
    print(f"❌ Error: {e}")
    print("✅ Solution: Check available frames with get_available_snips()")
    print("   Use 'all' for all frames or valid frame numbers")

# Error 2: DEAD phenotype conflicts
print("\n2️⃣ DEAD Phenotype Conflicts")
print("❌ Error: Cannot add 'EDEMA' - embryo already marked DEAD")
print("✅ Solutions:")
print("   1. Use force_dead=True to override")
print("   2. Remove DEAD phenotype first")
print("   3. Check temporal logic - was embryo marked dead too early?")

# Error 3: Missing entity IDs
print("\n3️⃣ Missing Entity IDs")
print("❌ Error: Embryo exp001_v999_e001 not found")
print("✅ Solutions:")
print("   1. Check ID format: experiment_video_embryo_snip")
print("   2. List available IDs: list(metadata.data['embryos'].keys())")
print("   3. Verify SAM file has expected data")

# Error 4: Schema validation errors
print("\n4️⃣ Schema Validation Errors")
print("❌ Error: Invalid phenotype 'CUSTOM_PHENOTYPE'")
print("✅ Solutions:")
print("   1. Check available values: metadata.schema_manager.get_phenotypes()")
print("   2. Add to schema: metadata.schema_manager.add_phenotype()")
print("   3. Use existing phenotype with notes for specifics")

# %%
# 10.2 Data Recovery

# Restore from backup
backup_dir = Path(metadata.filepath).parent / "backups"
if backup_dir.exists():
    backups = sorted(backup_dir.glob("*.json"))
    if backups:
        print(f"📁 Found {len(backups)} backups:")
        for backup in backups[-3:]:  # Last 3
            print(f"  - {backup.name}")
        
        # To restore:
        # metadata.data = metadata.load_json(backups[-1])
        # metadata._build_caches()
        print("\n💡 To restore: metadata.data = metadata.load_json(backup_path)")

# Fix corrupted data
def fix_orphaned_snips(metadata):
    """Find and fix snips without parent embryos"""
    orphans = []
    for embryo_data in metadata.data["embryos"].values():
        for snip_id in embryo_data.get("snips", {}):
            if snip_id not in metadata._snip_to_embryo:
                orphans.append(snip_id)
    
    if orphans:
        print(f"🔧 Found {len(orphans)} orphaned snips")
        # Rebuild cache
        metadata._build_caches()
        print("✅ Cache rebuilt")
    
    return orphans

# Check for data corruption
# orphans = fix_orphaned_snips(metadata)

# Merge conflicting changes
def merge_metadata_files(file1, file2, output_path):
    """Merge two metadata files intelligently"""
    data1 = EmbryoMetadata.load_json(file1)
    data2 = EmbryoMetadata.load_json(file2)
    
    merged = data1.copy()
    conflicts = []
    
    # Merge embryos
    for embryo_id, embryo_data in data2["embryos"].items():
        if embryo_id not in merged["embryos"]:
            merged["embryos"][embryo_id] = embryo_data
        else:
            # Check for conflicts
            if embryo_data != merged["embryos"][embryo_id]:
                conflicts.append(embryo_id)
    
    print(f"📊 Merge summary:")
    print(f"  File 1: {len(data1['embryos'])} embryos")
    print(f"  File 2: {len(data2['embryos'])} embryos")
    print(f"  Merged: {len(merged['embryos'])} embryos")
    print(f"  Conflicts: {len(conflicts)}")
    
    return merged, conflicts

# %% [markdown]
# ## 11. Best Practices

# %%
# 11.1 Annotation Guidelines

print("📚 ANNOTATION BEST PRACTICES")
print("=" * 50)

print("\n✅ Consistent Phenotype Terminology:")
print("  - Use schema-defined phenotypes")
print("  - Add specific details in notes field")
print("  - Be consistent across experiments")

print("\n✅ Temporal Annotation Strategies:")
print("  - Annotate at first appearance of phenotype")
print("  - Use frame ranges for transient phenotypes")
print("  - Mark DEAD only when certain")

print("\n✅ Author Attribution:")
print("  - Always include author in annotations")
print("  - Use consistent author IDs")
print("  - Add timestamps for tracking")

# Example of good practice
good_batch = AnnotationBatch(
    author="researcher_001",  # Consistent ID
    description="2024-01-15 screening - Experiment 42"  # Detailed description
)

# Specific phenotype with detailed notes
good_batch.add_phenotype(
    "exp042_v001_e001",
    "EDEMA",
    frames="0250:0350",  # Specific frame range
    notes="Mild pericardial edema, ~10% of heart volume",
    confidence=0.90  # Include confidence when relevant
)

print("\n📝 Example annotation with best practices applied ✓")

# %%
# 11.2 Performance Tips

print("\n⚡ PERFORMANCE OPTIMIZATION TIPS")
print("=" * 50)

# Batch size optimization
print("\n📦 Batch Size Optimization:")
print("  - Ideal batch size: 50-100 embryos")
print("  - Larger batches = more memory, but fewer I/O operations")
print("  - Use dry_run=True for large batches first")

# Demonstrate batch size impact
import time

# Small batches (inefficient)
start = time.time()
for embryo_id in list(metadata.data["embryos"].keys())[:10]:
    single_batch = AnnotationBatch("test", "Single")
    single_batch.add_genotype(embryo_id, "WT")
    # Would apply each one separately (don't actually do this!)
print(f"  10 individual operations: {time.time() - start:.3f}s (simulated)")

# One large batch (efficient)
start = time.time()
large_batch = AnnotationBatch("test", "Large")
for embryo_id in list(metadata.data["embryos"].keys())[:10]:
    large_batch.add_genotype(embryo_id, "WT")
# Single apply operation
print(f"  1 batch operation: {time.time() - start:.3f}s")

print("\n🔍 Validation Frequency:")
print("  - Turn off auto-validation for bulk operations:")
print("    metadata.auto_validate(False)")
print("  - Run validation after batch completion:")
print("    metadata.validate_data_integrity()")
print("  - Re-enable for interactive work")

print("\n💾 Memory Management:")
print("  - For huge datasets (>10k embryos):")
print("    - Process in chunks")
print("    - Clear caches periodically: metadata._build_caches()")
print("    - Save frequently: metadata.save()")

# %% [markdown]
# ## Appendix

# %%
# A. Schema Customization

print("🎨 SCHEMA CUSTOMIZATION")
print("=" * 50)

# Add custom phenotypes
metadata.schema_manager.add_phenotype(
    "HEART_DEFECT",
    description="Cardiac abnormalities including looping defects",
    exclusive=False,
    terminal=False
)
print("✅ Added custom phenotype: HEART_DEFECT")

# Add custom flag types
metadata.schema_manager.add_flag(
    "LOW_QUALITY",
    level="snip",
    description="Image quality below threshold",
    priority="medium"
)
print("✅ Added custom flag: LOW_QUALITY")

# Export schema for sharing
schema_export_path = Path("exports/custom_schema.json")
metadata.schema_manager.save_schema()
print(f"💾 Exported schema to {metadata.schema_manager.schema_path}")

# Import schema from collaborator
# metadata.schema_manager.import_schema("collaborator_schema.json", merge=True)

# %%
# B. Integration Examples

print("🔗 INTEGRATION EXAMPLES")
print("=" * 50)

# Connect to image viewer (pseudocode)
def open_in_viewer(snip_id):
    """Open snip in external image viewer"""
    # Get image path from SAM annotations
    embryo_id = metadata.get_embryo_id_from_snip(snip_id)
    
    # Parse video and frame info
    parts = snip_id.split("_")
    video_id = f"{parts[0]}_{parts[1]}"
    frame_num = int(parts[-1][1:])
    
    print(f"🖼️ Opening {snip_id}:")
    print(f"  Video: {video_id}")
    print(f"  Frame: {frame_num}")
    print(f"  [Would launch viewer here]")
    
    return {"video": video_id, "frame": frame_num}

# Example
# viewer_info = open_in_viewer("exp001_v001_e001_s0250")

# Export for statistical analysis
def export_for_R(metadata, output_path):
    """Export data in R-friendly format"""
    data_for_r = []
    
    for embryo_id, embryo_data in metadata.data["embryos"].items():
        # Parse metadata
        parts = embryo_id.split("_")
        
        # Count phenotypes
        phenotype_counts = {}
        for snip_data in embryo_data["snips"].values():
            if "phenotype" in snip_data:
                pheno = snip_data["phenotype"]["value"]
                phenotype_counts[pheno] = phenotype_counts.get(pheno, 0) + 1
        
        row = {
            "embryo_id": embryo_id,
            "experiment": parts[0],
            "video": parts[1],
            "embryo_num": parts[2],
            "genotype": embryo_data.get("genotype", {}).get("value", "TBD"),
            **{f"pheno_{k}": v for k, v in phenotype_counts.items()}
        }
        data_for_r.append(row)
    
    df = pd.DataFrame(data_for_r)
    df.to_csv(output_path, index=False)
    print(f"📊 Exported {len(df)} rows for R analysis")
    
    return df

# Pipeline automation
def automated_pipeline(sam_path, config):
    """Fully automated annotation pipeline"""
    print("🤖 Starting automated pipeline...")
    
    # Load configuration
    author = config.get("author", "auto_pipeline")
    genotype_map = config.get("genotype_map", {})  # video -> genotype
    treatment_map = config.get("treatment_map", {})  # video -> treatment
    
    # Initialize
    metadata = EmbryoMetadata(sam_path, gen_if_no_file=True)
    batch = AnnotationBatch(author, "Automated annotation")
    
    # Apply mappings
    for embryo_id in metadata.data["embryos"]:
        video_id = "_".join(embryo_id.split("_")[:2])
        
        # Assign genotype
        if video_id in genotype_map:
            batch.add_genotype(embryo_id, genotype_map[video_id])
        
        # Assign treatment
        if video_id in treatment_map:
            treatment_info = treatment_map[video_id]
            batch.add_treatment(embryo_id, **treatment_info)
    
    # Apply and save
    results = metadata.apply_batch(batch)
    metadata.save()
    
    print(f"✅ Pipeline complete: {results['applied']} annotations")
    return metadata

# %%
# C. Quick Reference

print("📋 QUICK REFERENCE")
print("=" * 50)

print("\n🆔 ID Format Patterns:")
print("  experiment:     exp001")
print("  video:          exp001_v001")
print("  embryo:         exp001_v001_e001")
print("  snip:           exp001_v001_e001_s0250")

print("\n🎯 Frame Specification Syntax:")
print("  all frames:     'all'")
print("  index range:    '[0:10]'        # First 10 snips")
print("  frame range:    '0200:0400'     # Frames 200-400")
print("  with step:      '0100:0500:50'  # Every 50th frame")
print("  from frame:     '0300:'         # Frame 300 onward")
print("  list:           ['snip1', 'snip2']")

print("\n🔬 Common Phenotype Codes:")
for pheno, info in metadata.schema_manager.get_phenotypes().items():
    print(f"  {pheno:20} - {info['description']}")

print("\n🚩 Flag Priority Meanings:")
print("  low:      Minor issue, does not affect analysis")
print("  medium:   May affect some analyses, use caution")
print("  high:     Significant issue, exclude from most analyses")
print("  critical: Severe issue, exclude from all analyses")

# %% [markdown]
# ## 🎉 Congratulations!
# 
# You've mastered the EmbryoMetadata system! 
# 
# Key takeaways:
# - Always validate before applying batches
# - Use frame ranges for temporal annotations
# - Respect the DEAD phenotype exclusivity
# - Leverage batch operations for efficiency
# - Keep your schema organized and shared
# 
# Happy annotating! 🐠✨

# %%
print("\n🏁 Tutorial Complete!")
print("You're now ready to annotate embryos like a pro!")
print("\nNext steps:")
print("  1. Try the exercises throughout this notebook")
print("  2. Annotate your own data")
print("  3. Share your schemas with collaborators")
print("  4. Automate repetitive tasks")
print("\nGood luck with your research! 🔬🐟")