#!/usr/bin/env python3
"""
Percentile Threshold Analysis and Video Generation

1. Fix the D03 plot with correct axis orientation
2. Create CV vs percentile analysis plot to find good thresholds
3. Rank embryos by different percentiles and generate evaluation videos
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import subprocess

def load_data():
    """Load the trimmed percentile analysis results."""
    with open('trimmed_percentile_plots/trimmed_percentile_results.json', 'r') as f:
        data = json.load(f)
    
    d03_embryo = data['d03_case_study']
    all_embryos = data['detailed_embryo_data']
    
    return d03_embryo, all_embryos

def create_fixed_d03_plot(d03_embryo: dict):
    """Create corrected D03 plot with proper axis orientation."""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    areas = np.array(d03_embryo["areas"])
    frames = np.arange(len(areas))
    local_diffs = d03_embryo.get("local_diffs", [])
    
    # Top plot: Area over time (UNCHANGED)
    ax1.plot(frames, areas, 'b-', linewidth=1.5, alpha=0.8, label='Area')
    
    # Add trend line
    z = np.polyfit(frames, areas, 1)
    p = np.poly1d(z)
    ax1.plot(frames, p(frames), 'r--', alpha=0.8, linewidth=2, label='Linear trend')
    
    # Mark periods
    early_third = len(areas) // 3
    middle_third = 2 * len(areas) // 3
    
    ax1.axvline(early_third, color='orange', linestyle=':', alpha=0.7, label='Period boundaries')
    ax1.axvline(middle_third, color='orange', linestyle=':', alpha=0.7)
    
    ax1.set_title(f'20250305_D03_e01: The Death/Plateau Problem Example\n'
                  f'CV: {d03_embryo["cv"]:.3f} | Local Median: {d03_embryo["local_var_median"]:.3f} | '
                  f'Local 95th: {d03_embryo["local_var_p95"]:.3f}', 
                  fontsize=16, fontweight='bold')
    ax1.set_xlabel('Frame Number', fontsize=12)
    ax1.set_ylabel('Area (pixels)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Local variation over time (CORRECTED - only variation on x-axis)
    if local_diffs:
        # Create frame indices for local_diffs (accounting for edge effects)
        diff_frames = frames[1:-1]  # Approximate alignment
        if len(local_diffs) == len(diff_frames):
            # CORRECTED: Local variation on x-axis, frame number on y-axis
            ax2.plot(local_diffs, diff_frames, 'g-', linewidth=1.5, alpha=0.8, label='Local variation per frame')
        else:
            # If lengths don't match, interpolate
            diff_frames = np.linspace(0, len(frames)-1, len(local_diffs))
            ax2.plot(local_diffs, diff_frames, 'g-', linewidth=1.5, alpha=0.8, label='Local variation per frame')
        
        # Add threshold lines (vertical since variation is on x-axis)
        median_val = d03_embryo["local_var_median"]
        p95_val = d03_embryo["local_var_p95"]
        
        ax2.axvline(median_val, color='blue', linestyle='--', alpha=0.8, linewidth=3,
                   label=f'Median: {median_val:.3f} (FAILS)')
        ax2.axvline(p95_val, color='red', linestyle='--', alpha=0.8, linewidth=3,
                   label=f'95th percentile: {p95_val:.3f} (SUCCEEDS)')
        
        # Mark periods (horizontal since frame is on y-axis)
        ax2.axhline(early_third, color='orange', linestyle=':', alpha=0.7)
        ax2.axhline(middle_third, color='orange', linestyle=':', alpha=0.7)
        
        ax2.set_title('Local Variation vs Frame: Why 95th Percentile Fixes Death/Plateau', 
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('Local Variation (% difference)', fontsize=12)
        ax2.set_ylabel('Frame Number', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add text annotations
        ax2.text(0.02, 0.98, 'HIGH VARIATION\n(early frames)\nCaptured by 95th %ile', 
                transform=ax2.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7),
                verticalalignment='top', fontsize=11, fontweight='bold')
        
        ax2.text(0.02, 0.02, 'LOW VARIATION\n(death plateau)\nIgnored by 95th %ile', 
                transform=ax2.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.7),
                verticalalignment='bottom', fontsize=11, fontweight='bold')
        
        # Highlight the key insight
        ax2.text(0.6, 0.5, '95th Percentile = Peak Instability\nMedian = Death Plateau Pollution', 
                transform=ax2.transAxes, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.9),
                fontsize=12, fontweight='bold', ha='center')
    
    plt.tight_layout()
    return fig

def create_cv_vs_percentiles_plot(all_embryos):
    """Create plot showing CV vs different percentiles, colored by experiment."""
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Extract data
    cvs = [e['cv'] for e in all_embryos]
    p80s = [e['local_var_p90'] for e in all_embryos]  # Using p90 as proxy for p80
    p90s = [e['local_var_p90'] for e in all_embryos]
    p95s = [e['local_var_p95'] for e in all_embryos]
    experiments = [e['experiment_id'] for e in all_embryos]
    
    percentiles = [
        (p80s, "80th Percentile (approx)", axes[0]),
        (p90s, "90th Percentile", axes[1]),
        (p95s, "95th Percentile", axes[2])
    ]
    
    colors = {'20240418': 'blue', '20250305': 'red'}
    
    for percentile_vals, title, ax in percentiles:
        # Plot by experiment
        for exp in ['20240418', '20250305']:
            exp_cvs = [cvs[i] for i, e in enumerate(experiments) if e == exp]
            exp_percentiles = [percentile_vals[i] for i, e in enumerate(experiments) if e == exp]
            ax.scatter(exp_cvs, exp_percentiles, c=colors[exp], label=exp, alpha=0.7, s=50)
        
        # Add threshold lines
        cv_threshold = 0.15
        ax.axvline(cv_threshold, color='gray', linestyle='--', alpha=0.8, 
                  label=f'CV threshold ({cv_threshold*100:.0f}%)')
        
        # Calculate and show correlation
        correlation = np.corrcoef(cvs, percentile_vals)[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax.set_xlabel('Coefficient of Variation (CV)', fontsize=12)
        ax.set_ylabel(f'Local Variation ({title})', fontsize=12)
        ax.set_title(f'CV vs {title}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('CV vs Different Percentiles: Finding Good Thresholds\n'
                 'Lower Bound (20240418) vs Upper Bound (20250305) Experiments', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

def rank_embryos_by_percentiles(all_embryos):
    """Rank embryos by different percentiles and get top candidates for video generation."""
    
    # Group by experiment
    by_experiment = defaultdict(list)
    for embryo in all_embryos:
        by_experiment[embryo['experiment_id']].append(embryo)
    
    rankings = {}
    
    percentile_keys = ['local_var_p90', 'local_var_p95']  # Using p90 as proxy for both 80th and 90th
    percentile_names = ['80th_90th', '95th']
    
    for percentile_key, percentile_name in zip(percentile_keys, percentile_names):
        rankings[percentile_name] = {}
        
        for exp_id, exp_embryos in by_experiment.items():
            # Sort by percentile value (highest first)
            sorted_embryos = sorted(exp_embryos, key=lambda x: x[percentile_key], reverse=True)
            
            # Get top 2
            top_2 = sorted_embryos[:2]
            rankings[percentile_name][exp_id] = top_2
            
            print(f"\n{percentile_name} percentile - {exp_id} top 2:")
            for i, embryo in enumerate(top_2):
                print(f"  {i+1}. {embryo['embryo_id']}: {embryo[percentile_key]:.4f} "
                      f"(CV: {embryo['cv']:.3f}, frames: {embryo['frame_count']})")
    
    return rankings

def create_video_generation_script(rankings):
    """Create a script to generate evaluation videos for top-ranked embryos."""
    
    script_content = '''#!/usr/bin/env python3
"""
Generate evaluation videos for top-ranked embryos by percentile analysis.
"""

import subprocess
import sys
from pathlib import Path

def run_video_generation(experiment_id, video_id, embryo_id, percentile_type, rank, percentile_value):
    """Generate a single evaluation video."""
    
    output_dir = Path("results/percentile_threshold_videos")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_name = f"{percentile_type}_{experiment_id}_{embryo_id}_rank{rank}_p{percentile_value:.3f}.mp4"
    output_path = output_dir / output_name
    
    # Command to run the video generation
    cmd = [
        sys.executable, 
        "tests/create_sam2_eval_video.py",
        "--experiment", experiment_id,
        "--video", video_id,
        "--output", str(output_path)
    ]
    
    print(f"🎬 Generating: {output_name}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"✅ Success: {output_name}")
            return True
        else:
            print(f"❌ Failed: {output_name}")
            print(f"Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout: {output_name}")
        return False
    except Exception as e:
        print(f"💥 Exception: {output_name} - {e}")
        return False

def main():
    """Generate all videos for ranked embryos."""
    
    print("🎯 Generating evaluation videos for top-ranked embryos...")
    
    # Video generation tasks (from ranking analysis)
    video_tasks = [
'''
    
    # Add the ranked embryos to the script
    for percentile_name, exp_data in rankings.items():
        for exp_id, embryos in exp_data.items():
            for rank, embryo in enumerate(embryos, 1):
                embryo_id = embryo['embryo_id']
                video_id = embryo['video_id']
                percentile_key = 'local_var_p90' if percentile_name == '80th_90th' else 'local_var_p95'
                percentile_value = embryo[percentile_key]
                
                script_content += f'''        ("{exp_id}", "{video_id}", "{embryo_id}", "{percentile_name}", {rank}, {percentile_value}),
'''
    
    script_content += '''    ]
    
    successful = 0
    total = len(video_tasks)
    
    for experiment_id, video_id, embryo_id, percentile_type, rank, percentile_value in video_tasks:
        success = run_video_generation(experiment_id, video_id, embryo_id, percentile_type, rank, percentile_value)
        if success:
            successful += 1
    
    print(f"\\n📊 Summary: {successful}/{total} videos generated successfully")
    
    if successful > 0:
        print(f"📁 Videos saved to: results/percentile_threshold_videos/")
        print(f"🔍 Use vlc or other video player to review the results")

if __name__ == "__main__":
    main()
'''
    
    return script_content

def main():
    """Run the complete percentile threshold analysis."""
    
    print("🎯 Percentile Threshold Analysis and Video Generation")
    print("="*60)
    
    # Load data
    d03_embryo, all_embryos = load_data()
    
    # Create output directory
    output_dir = Path('percentile_threshold_analysis')
    output_dir.mkdir(exist_ok=True)
    
    print("\n1️⃣ Creating fixed D03 plot...")
    d03_fig = create_fixed_d03_plot(d03_embryo)
    d03_fig.savefig(output_dir / "d03_death_plateau_FIXED.png", dpi=300, bbox_inches='tight')
    d03_fig.savefig(output_dir / "d03_death_plateau_FIXED.pdf", bbox_inches='tight')
    print("✅ Fixed D03 plot saved")
    
    print("\n2️⃣ Creating CV vs percentiles analysis plot...")
    cv_percentiles_fig = create_cv_vs_percentiles_plot(all_embryos)
    cv_percentiles_fig.savefig(output_dir / "cv_vs_percentiles_analysis.png", dpi=300, bbox_inches='tight')
    cv_percentiles_fig.savefig(output_dir / "cv_vs_percentiles_analysis.pdf", bbox_inches='tight')
    print("✅ CV vs percentiles plot saved")
    
    print("\n3️⃣ Ranking embryos by percentiles...")
    rankings = rank_embryos_by_percentiles(all_embryos)
    
    # Save rankings to JSON
    rankings_serializable = {}
    for percentile_name, exp_data in rankings.items():
        rankings_serializable[percentile_name] = {}
        for exp_id, embryos in exp_data.items():
            rankings_serializable[percentile_name][exp_id] = [
                {
                    'embryo_id': e['embryo_id'],
                    'video_id': e['video_id'],
                    'cv': e['cv'],
                    'local_var_p90': e['local_var_p90'],
                    'local_var_p95': e['local_var_p95'],
                    'frame_count': e['frame_count']
                } for e in embryos
            ]
    
    with open(output_dir / "embryo_rankings.json", 'w') as f:
        json.dump(rankings_serializable, f, indent=2)
    
    print("\n4️⃣ Creating video generation script...")
    video_script = create_video_generation_script(rankings)
    
    with open(output_dir / "generate_evaluation_videos.py", 'w') as f:
        f.write(video_script)
    
    # Make it executable
    subprocess.run(['chmod', '+x', str(output_dir / "generate_evaluation_videos.py")])
    
    print(f"\n✅ Analysis complete! Results saved to: {output_dir}")
    print("\n📋 Generated files:")
    print(f"  📊 d03_death_plateau_FIXED.png - Corrected D03 plot")
    print(f"  📈 cv_vs_percentiles_analysis.png - CV vs percentiles threshold analysis")
    print(f"  📄 embryo_rankings.json - Top-ranked embryos by percentile")
    print(f"  🐍 generate_evaluation_videos.py - Script to create evaluation videos")
    
    print(f"\n🎬 To generate evaluation videos, run:")
    print(f"     python {output_dir}/generate_evaluation_videos.py")
    
    total_videos = sum(len(embryos) for exp_data in rankings.values() for embryos in exp_data.values())
    print(f"\n📊 Will generate {total_videos} evaluation videos for visual threshold assessment")

if __name__ == "__main__":
    main()