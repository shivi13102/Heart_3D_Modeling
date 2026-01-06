"""
step6_quantitative_metrics.py
Comprehensive quantitative cardiac metrics calculation
"""
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import pandas as pd
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

# Paths
img_path = r"E:/SEMESTER SUBJECTS/6th SEMESTER/MINOR PROJECT/Dataset/imagesTr/la_003.nii.gz"
mask_path = r"E:/SEMESTER SUBJECTS/6th SEMESTER/MINOR PROJECT/Dataset/labelsTr/la_003.nii.gz"

print("Loading data...")
img_nib = nib.load(img_path)
mask_nib = nib.load(mask_path)
img_data = img_nib.get_fdata()
mask_data = mask_nib.get_fdata()

print(f"Image shape: {img_data.shape}")
print(f"Mask shape: {mask_data.shape}")

# Get voxel dimensions
voxel_dims = img_nib.header.get_zooms()
voxel_volume_mm3 = voxel_dims[0] * voxel_dims[1] * voxel_dims[2]
print(f"\nVoxel dimensions (mm): {voxel_dims}")
print(f"Voxel volume: {voxel_volume_mm3:.3f} mm^3")

# ========== IDENTIFY STRUCTURES ==========
unique_labels = np.unique(mask_data)
print(f"\nUnique labels in mask: {unique_labels}")

# Map labels to cardiac structures
structure_map = {}
for label in unique_labels:
    if label == 0:
        structure_map[label] = {'name': 'Background', 'abbr': 'BG'}
    elif label == 1:
        structure_map[label] = {'name': 'Left Atrium', 'abbr': 'LA'}
    elif label == 2:
        structure_map[label] = {'name': 'Left Ventricle', 'abbr': 'LV'}
    elif label == 3:
        structure_map[label] = {'name': 'Right Ventricle', 'abbr': 'RV'}
    elif label == 4:
        structure_map[label] = {'name': 'Myocardium', 'abbr': 'MYO'}
    else:
        structure_map[label] = {'name': f'Unknown_{label}', 'abbr': f'U{label}'}

print("\nStructure mapping:")
for label, info in structure_map.items():
    print(f"  Label {label}: {info['name']} ({info['abbr']})")

# ========== VOLUME CALCULATION ==========
def calculate_structure_metrics(label_value, name):
    """Calculate volume and basic metrics for a structure"""
    mask_binary = (mask_data == label_value).astype(np.uint8)
    voxel_count = np.sum(mask_binary)
    
    if voxel_count == 0:
        return None
    
    volume_mm3 = voxel_count * voxel_volume_mm3
    volume_ml = volume_mm3 / 1000
    
    # Calculate center of mass
    com = ndimage.center_of_mass(mask_binary)
    
    # Calculate bounding box
    nonzero_indices = np.nonzero(mask_binary)
    if len(nonzero_indices[0]) > 0:
        bbox_min = [np.min(idx) for idx in nonzero_indices]
        bbox_max = [np.max(idx) for idx in nonzero_indices]
        bbox_size = [bbox_max[i] - bbox_min[i] for i in range(3)]
    else:
        bbox_min = bbox_max = bbox_size = [0, 0, 0]
    
    return {
        'name': name,
        'label': label_value,
        'voxel_count': voxel_count,
        'volume_mm3': volume_mm3,
        'volume_ml': volume_ml,
        'center_of_mass': com,
        'bbox_min': bbox_min,
        'bbox_max': bbox_max,
        'bbox_size': bbox_size
    }

# Calculate metrics for all non-background structures
structures_metrics = []
for label in unique_labels:
    if label != 0:  # Skip background
        metrics = calculate_structure_metrics(label, structure_map[label]['name'])
        if metrics:
            structures_metrics.append(metrics)

print("\n" + "="*60)
print("BASIC VOLUME METRICS")
print("="*60)

for struct in structures_metrics:
    print(f"\n{struct['name']}:")
    print(f"  Volume: {struct['volume_ml']:.2f} mL ({struct['volume_mm3']:.0f} mm^3)")
    print(f"  Voxels: {struct['voxel_count']:,}")
    print(f"  Center: ({struct['center_of_mass'][0]:.1f}, {struct['center_of_mass'][1]:.1f}, {struct['center_of_mass'][2]:.1f})")
    print(f"  BBox size: {struct['bbox_size'][0]}x{struct['bbox_size'][1]}x{struct['bbox_size'][2]} voxels")

# ========== EJECTION FRACTION ESTIMATION ==========
def estimate_ejection_fraction(edv_ml, structure_name="LV"):
    """Estimate ejection fraction from end-diastolic volume"""
    # Typical values for simulation
    typical_ef_ranges = {
        'LV': {'normal': (55, 70), 'mild': (45, 54), 'moderate': (30, 44), 'severe': (0, 29)},
        'RV': {'normal': (45, 65), 'mild': (35, 44), 'moderate': (25, 34), 'severe': (0, 24)}
    }
    
    # Use middle of normal range for simulation
    normal_ef = np.mean(typical_ef_ranges.get(structure_name, {'normal': (55, 70)})['normal'])
    
    # Calculate ESV from EF
    ef_percent = normal_ef
    esv_ml = edv_ml * (1 - ef_percent/100)
    stroke_volume = edv_ml - esv_ml
    
    return {
        'ef_percent': ef_percent,
        'esv_ml': esv_ml,
        'sv_ml': stroke_volume,
        'status': 'Normal (simulated)'
    }

# Calculate EF for LV and RV if present
ef_results = {}
for struct in structures_metrics:
    if 'ventricle' in struct['name'].lower():
        abbr = 'LV' if 'left' in struct['name'].lower() else 'RV'
        ef_results[abbr] = estimate_ejection_fraction(struct['volume_ml'], abbr)

# ========== WALL THICKNESS ESTIMATION ==========
def estimate_myocardial_thickness(myo_mask, voxel_dims):
    """Estimate myocardial wall thickness"""
    if np.sum(myo_mask) == 0:
        return {'avg_thickness_mm': 0, 'min_thickness_mm': 0, 'max_thickness_mm': 0}
    
    # Simplified thickness estimation using distance transform
    from scipy.ndimage import distance_transform_edt
    
    # Get binary mask
    myo_binary = myo_mask > 0
    
    # Calculate distance from myocardium boundary
    distance = distance_transform_edt(myo_binary)
    
    # Only consider voxels that are part of the myocardium
    thickness_voxels = distance[myo_binary]
    
    # Convert to mm using average in-plane resolution
    avg_resolution = (voxel_dims[0] + voxel_dims[1]) / 2
    thickness_mm = thickness_voxels * avg_resolution * 2  # Multiply by 2 for full thickness
    
    return {
        'avg_thickness_mm': np.mean(thickness_mm),
        'min_thickness_mm': np.min(thickness_mm),
        'max_thickness_mm': np.max(thickness_mm),
        'std_thickness_mm': np.std(thickness_mm)
    }

# Find myocardium
myo_metrics = None
for struct in structures_metrics:
    if 'myocardium' in struct['name'].lower():
        myo_mask = (mask_data == struct['label']).astype(np.uint8)
        thickness_results = estimate_myocardial_thickness(myo_mask, voxel_dims)
        myo_metrics = {**struct, **thickness_results}
        break

# ========== CARDIAC OUTPUT ESTIMATION ==========
def estimate_cardiac_output(lv_sv_ml, heart_rate_bpm=70):
    """Estimate cardiac output"""
    co_l_min = lv_sv_ml * heart_rate_bpm / 1000
    return {
        'co_l_min': co_l_min,
        'heart_rate': heart_rate_bpm,
        'status': 'Normal' if 4.0 <= co_l_min <= 8.0 else 'Outside normal range'
    }

# Get LV stroke volume for CO calculation
if 'LV' in ef_results:
    lv_sv_ml = ef_results['LV']['sv_ml']
else:
    # If no LV found, use typical value for demonstration
    lv_sv_ml = 70  # Default 70 mL
    print("\nNote: No LV found, using typical stroke volume (70 mL) for demonstration")
    
co_results = estimate_cardiac_output(lv_sv_ml)

# ========== CREATE COMPREHENSIVE METRICS TABLE ==========
print("\n" + "="*60)
print("COMPREHENSIVE CARDIAC METRICS")
print("="*60)

metrics_table = []

# Helper functions
def get_normal_range(metric_name):
    """Return typical normal ranges for cardiac metrics"""
    ranges = {
        'Left Ventricle Volume': '50-150 mL',
        'Right Ventricle Volume': '50-150 mL',
        'Myocardium Volume': '80-200 mL',
        'Left Atrium Volume': '30-100 mL',
        'LV EF': '55-70%',
        'RV EF': '45-65%',
        'LV SV': '60-100 mL',
        'RV SV': '60-100 mL',
        'Wall Thickness': '6-10 mm',
        'Cardiac Output': '4.0-8.0 L/min'
    }
    return ranges.get(metric_name, 'N/A')

def assess_metric(value, metric_name, metric_type):
    """Assess if metric is normal"""
    normal_ranges = {
        'volume': {
            'Left Ventricle': (50, 150),
            'Right Ventricle': (50, 150),
            'Myocardium': (80, 200),
            'Left Atrium': (30, 100)
        },
        'thickness': {
            'Wall Thickness': (6, 10)
        }
    }
    
    for key, (low, high) in normal_ranges.get(metric_type, {}).items():
        if key in metric_name:
            if low <= value <= high:
                return "Normal"
            elif value < low:
                return f"Low (<{low})"
            else:
                return f"High (>{high})"
    
    return "Check manually"

# 1. Volume metrics
for struct in structures_metrics:
    metrics_table.append([
        f"{struct['name']} Volume",
        f"{struct['volume_ml']:.2f} mL",
        get_normal_range(f"{struct['name']} Volume"),
        assess_metric(struct['volume_ml'], struct['name'], 'volume')
    ])

# 2. Ejection fraction (simulated if needed)
if ef_results:
    for abbr, ef_data in ef_results.items():
        metrics_table.append([
            f"{abbr} Ejection Fraction",
            f"{ef_data['ef_percent']:.1f}%",
            get_normal_range(f"{abbr} EF"),
            ef_data['status']
        ])
        
        metrics_table.append([
            f"{abbr} Stroke Volume",
            f"{ef_data['sv_ml']:.1f} mL",
            get_normal_range(f"{abbr} SV"),
            assess_metric(ef_data['sv_ml'], f"{abbr} SV", 'volume')
        ])
else:
    # If no ventricles found, add simulated data for demonstration
    print("\nNote: No ventricle structures found. Using simulated values for demonstration.")
    metrics_table.append([
        "LV Ejection Fraction (simulated)",
        "62.5%",
        "55-70%",
        "Normal (simulated)"
    ])
    metrics_table.append([
        "LV Stroke Volume (simulated)",
        "75.0 mL",
        "60-100 mL",
        "Normal (simulated)"
    ])

# 3. Wall thickness
if myo_metrics:
    metrics_table.append([
        "Avg Myocardial Thickness",
        f"{myo_metrics['avg_thickness_mm']:.1f} mm",
        "6-10 mm",
        assess_metric(myo_metrics['avg_thickness_mm'], "Wall Thickness", 'thickness')
    ])
else:
    metrics_table.append([
        "Avg Myocardial Thickness",
        "N/A",
        "6-10 mm",
        "Not available"
    ])

# 4. Cardiac output
metrics_table.append([
    "Cardiac Output",
    f"{co_results['co_l_min']:.2f} L/min",
    "4.0-8.0 L/min",
    co_results['status']
])

# Print table
headers = ["Parameter", "Value", "Normal Range", "Assessment"]
print(f"\n{headers[0]:<35} {headers[1]:<15} {headers[2]:<15} {headers[3]:<20}")
print("-" * 90)

for row in metrics_table:
    print(f"{row[0]:<35} {row[1]:<15} {row[2]:<15} {row[3]:<20}")

# ========== VISUALIZATION DASHBOARD ==========
print("\nGenerating visualization dashboard...")

fig = plt.figure(figsize=(18, 12))
fig.suptitle('CardioTwin - Quantitative Cardiac Analysis Dashboard', fontsize=18, y=0.98)

# Plot 1: Volume distribution
ax1 = plt.subplot(2, 3, 1)
struct_names = [s['name'] for s in structures_metrics]
volumes = [s['volume_ml'] for s in structures_metrics]

if len(struct_names) > 0:
    colors = plt.cm.Set3(np.arange(len(struct_names)) / len(struct_names))
    bars = ax1.bar(range(len(struct_names)), volumes, color=colors, alpha=0.8)
    ax1.set_ylabel('Volume (mL)')
    ax1.set_title('Cardiac Structure Volumes')
    ax1.set_xticks(range(len(struct_names)))
    ax1.set_xticklabels([n.split()[0] for n in struct_names], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, vol in zip(bars, volumes):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                 f'{vol:.1f}', ha='center', va='bottom', fontsize=9)
else:
    ax1.text(0.5, 0.5, 'No structures found\nfor volume analysis', 
             transform=ax1.transAxes, ha='center', va='center', fontsize=12)
    ax1.set_title('Cardiac Structure Volumes')
    ax1.axis('off')

# Plot 2: Ejection fraction gauge
ax2 = plt.subplot(2, 3, 2)
if ef_results:
    ef_values = [ef_results.get('LV', {}).get('ef_percent', 0), 
                 ef_results.get('RV', {}).get('ef_percent', 0)]
    ef_labels = ['LV EF', 'RV EF']
    
    x_pos = range(len(ef_values))
    colors_ef = ['green' if 55 <= v <= 70 else 'orange' if 45 <= v < 55 else 'red' 
                 for v in ef_values]
    
    bars_ef = ax2.bar(x_pos, ef_values, color=colors_ef, alpha=0.7)
    ax2.set_ylabel('Ejection Fraction (%)')
    ax2.set_title('Ventricular Function')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(ef_labels)
    ax2.axhline(y=55, color='red', linestyle='--', alpha=0.5, label='Normal Min')
    ax2.axhline(y=70, color='green', linestyle='--', alpha=0.5, label='Normal Max')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Add EF values on bars
    for bar, ef in zip(bars_ef, ef_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                 f'{ef:.1f}%', ha='center', va='bottom', fontsize=10)
else:
    # Show simulated EF
    ef_values = [62.5, 55.0]
    ef_labels = ['LV EF (sim)', 'RV EF (sim)']
    x_pos = range(len(ef_values))
    bars_ef = ax2.bar(x_pos, ef_values, color=['green', 'blue'], alpha=0.7)
    ax2.set_ylabel('Ejection Fraction (%)')
    ax2.set_title('Ventricular Function (Simulated)')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(ef_labels)
    ax2.axhline(y=55, color='red', linestyle='--', alpha=0.5, label='Normal Min')
    ax2.axhline(y=70, color='green', linestyle='--', alpha=0.5, label='Normal Max')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Add values on bars
    for bar, ef in zip(bars_ef, ef_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                 f'{ef:.1f}%', ha='center', va='bottom', fontsize=10)

# Plot 3: Wall thickness visualization
ax3 = plt.subplot(2, 3, 3)
if myo_metrics:
    # Show myocardium slice with thickness
    slice_idx = int(structures_metrics[0]['center_of_mass'][2])
    myo_slice = (mask_data[:, :, slice_idx] == myo_metrics['label']).astype(np.uint8)
    
    im = ax3.imshow(myo_slice, cmap='hot', alpha=0.8)
    ax3.set_title(f'Myocardium (Slice {slice_idx})')
    ax3.axis('off')
    
    # Add thickness annotation
    thickness_text = f"Avg Thickness: {myo_metrics['avg_thickness_mm']:.1f} mm\n"
    thickness_text += f"Min: {myo_metrics['min_thickness_mm']:.1f} mm\n"
    thickness_text += f"Max: {myo_metrics['max_thickness_mm']:.1f} mm"
    ax3.text(0.02, 0.98, thickness_text, transform=ax3.transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
else:
    # Show what myocardium would look like
    ax3.text(0.5, 0.5, 'Myocardium data\nnot available\n\n(Shows heart muscle\nwall thickness)', 
             transform=ax3.transAxes, ha='center', va='center', fontsize=11)
    ax3.set_title('Myocardial Thickness')
    ax3.axis('off')

# Plot 4: Cardiac cycle simulation
ax4 = plt.subplot(2, 3, 4)
if len(structures_metrics) > 0:
    # Simulate simple cardiac cycle
    time = np.linspace(0, 1, 100)
    
    # Use first structure volume or simulated value
    if structures_metrics:
        base_volume = structures_metrics[0]['volume_ml']
    else:
        base_volume = 100  # Default
    
    # Get ESV for LV if available
    if 'LV' in ef_results:
        esv_ml = ef_results['LV']['esv_ml']
    else:
        esv_ml = base_volume * 0.4  # Estimate ESV as 40% of EDV
    
    # Create cardiac cycle waveform
    lv_volume = base_volume * (0.6 + 0.4 * np.sin(2 * np.pi * time - np.pi/2))
    
    ax4.plot(time, lv_volume, 'b-', linewidth=2)
    ax4.fill_between(time, esv_ml, lv_volume, alpha=0.3, color='blue')
    
    ax4.set_xlabel('Cardiac Cycle Phase')
    ax4.set_ylabel('LV Volume (mL)')
    ax4.set_title('Simulated Cardiac Cycle')
    ax4.grid(True, alpha=0.3)
    
    # Annotate key points
    ax4.axhline(y=base_volume, color='green', linestyle=':', alpha=0.7, label='EDV')
    ax4.axhline(y=esv_ml, color='red', linestyle=':', alpha=0.7, label='ESV')
    ax4.legend()
else:
    ax4.text(0.5, 0.5, 'Cardiac cycle simulation\nrequires ventricle data', 
             transform=ax4.transAxes, ha='center', va='center', fontsize=12)
    ax4.set_title('Cardiac Cycle Simulation')
    ax4.axis('off')

# Plot 5: Metrics summary table
ax5 = plt.subplot(2, 3, 5)
ax5.axis('tight')
ax5.axis('off')

# Create simplified table
table_data = [["Parameter", "Value", "Status"]]

# Add key metrics (take first 4 from metrics_table)
for i, row in enumerate(metrics_table[:4]):
    table_data.append([row[0], row[1], row[3]])

# Add cardiac output
for row in metrics_table:
    if "Cardiac Output" in row[0]:
        table_data.append([row[0], row[1], row[3]])
        break

table = ax5.table(cellText=table_data, cellLoc='left', 
                  colWidths=[0.35, 0.3, 0.35], loc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)

# Style header row
for i in range(3):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style normal values with green, abnormal with yellow
for i in range(1, len(table_data)):
    if "Normal" in str(table_data[i][2]):
        table[(i, 2)].set_facecolor('#90EE90')  # Light green
    elif "Low" in str(table_data[i][2]) or "High" in str(table_data[i][2]):
        table[(i, 2)].set_facecolor('#FFD700')  # Yellow
    elif "Not available" in str(table_data[i][2]):
        table[(i, 2)].set_facecolor('#D3D3D3')  # Light gray

ax5.set_title('Clinical Summary', fontsize=12, pad=20)

# Plot 6: 3D volume rendering
ax6 = plt.subplot(2, 3, 6, projection='3d')
try:
    # Find a structure to render in 3D
    render_structure = None
    for struct in structures_metrics:
        if struct['voxel_count'] > 1000:  # Only render if enough voxels
            render_structure = struct
            break
    
    if render_structure is not None:
        # Get the mask
        struct_mask = (mask_data == render_structure['label']).astype(np.uint8)
        
        # Smooth the mask
        mask_smoothed = gaussian_filter(struct_mask.astype(float), sigma=1.0)
        
        # Generate mesh
        from skimage import measure
        verts, faces, _, _ = measure.marching_cubes(mask_smoothed, level=0.5, step_size=2)
        
        # Plot
        ax6.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                        color='red', alpha=0.6, shade=True)
        ax6.set_title(f'3D {render_structure["name"]} Model')
        ax6.set_xlabel('X')
        ax6.set_ylabel('Y')
        ax6.set_zlabel('Z')
    else:
        # Create a simple 3D shape for demonstration
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = 10 * np.outer(np.cos(u), np.sin(v))
        y = 10 * np.outer(np.sin(u), np.sin(v))
        z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax6.plot_surface(x, y, z, color='blue', alpha=0.6)
        ax6.set_title('3D Heart Model (Demo)')
        ax6.set_xlabel('X')
        ax6.set_ylabel('Y')
        ax6.set_zlabel('Z')
except Exception as e:
    ax6.text(0.5, 0.5, 0.5, f'3D Visualization\n\nAvailable structures:\n{len(structures_metrics)}', 
             transform=ax6.transAxes, ha='center', va='center', fontsize=11)
    ax6.set_title('3D Cardiac Model')
    ax6.set_xlabel('X')
    ax6.set_ylabel('Y')
    ax6.set_zlabel('Z')

plt.tight_layout()
plt.show()

# ========== SAVE RESULTS ==========
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

# Create DataFrame for saving
df_metrics = pd.DataFrame(metrics_table, columns=["Parameter", "Value", "Normal Range", "Assessment"])

# Save to CSV
df_metrics.to_csv('cardiac_metrics_detailed.csv', index=False)
print(f"Detailed metrics saved to 'cardiac_metrics_detailed.csv'")

# Save summary statistics
summary_stats = {
    'Total Structures': len(structures_metrics),
    'Total Volume (mL)': sum([s['volume_ml'] for s in structures_metrics]),
    'Average Volume (mL)': np.mean([s['volume_ml'] for s in structures_metrics]) if structures_metrics else 0,
    'Largest Structure': max([s['name'] for s in structures_metrics], 
                            key=lambda x: next(s['volume_ml'] for s in structures_metrics if s['name'] == x)) if structures_metrics else "N/A",
    'Voxel Dimensions (mm)': str(voxel_dims)
}

print(f"\nSummary Statistics:")
for key, value in summary_stats.items():
    print(f"  {key}: {value}")

# ========== GENERATE REPORT ==========
print("\n" + "="*60)
print("CLINICAL INTERPRETATION")
print("="*60)

# Find Left Atrium metrics if available
la_metrics = None
for struct in structures_metrics:
    if 'atrium' in struct['name'].lower():
        la_metrics = struct
        break

if la_metrics:
    print(f"\n1. Left Atrium Analysis:")
    print(f"   Volume: {la_metrics['volume_ml']:.1f} mL")
    
    if 30 <= la_metrics['volume_ml'] <= 100:
        print("   -> Normal left atrial size")
    elif la_metrics['volume_ml'] > 100:
        print("   -> Left atrial enlargement detected")
        print("   -> May indicate diastolic dysfunction or atrial fibrillation risk")
    else:
        print("   -> Small left atrium")
    
    print(f"\n2. Dataset Information:")
    print(f"   This dataset appears to be segmented for Left Atrium only.")
    print(f"   For complete cardiac analysis, additional structures needed:")
    print(f"   - Left Ventricle (LV)")
    print(f"   - Right Ventricle (RV)")
    print(f"   - Myocardium")
else:
    print("\nNote: This dataset contains limited cardiac structures.")
    print("For comprehensive CardioTwin analysis, consider:")
    print("1. Using a full cardiac segmentation dataset")
    print("2. Training a multi-class segmentation model")
    print("3. Combining multiple single-structure segmentations")

print(f"\n3. Technical Details:")
print(f"   Image resolution: {voxel_dims[0]:.2f} x {voxel_dims[1]:.2f} x {voxel_dims[2]:.2f} mm")
print(f"   Total segmented volume: {sum([s['volume_ml'] for s in structures_metrics]):.1f} mL")
print(f"   Structures identified: {', '.join([s['name'] for s in structures_metrics])}")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print("\nNext steps:")
print("1. Review saved metrics in 'cardiac_metrics_detailed.csv'")
print("2. Compare with clinical reference values")
print("3. Use additional datasets for multi-structure analysis")
print("4. Implement temporal analysis for functional assessment")