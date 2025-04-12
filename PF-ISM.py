import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import math

# --- Dosya Seçimi ve Veri Okuma ---
Tk().withdraw()
file_path = askopenfilename(title="Select the expert opinions file", filetypes=[("Excel files", "*.xlsx *.xls")])
data = pd.read_excel(file_path, header=None)

non_empty_columns = data.dropna(axis=1, how='all')
non_empty_rows = data.dropna(axis=0, how='all')

factor_count = non_empty_columns.shape[1]
expert_count = non_empty_rows.shape[0] // factor_count
expert_opinions = np.array(non_empty_columns).reshape((expert_count, factor_count, factor_count))

# --- Picture Fuzzy Ölçek Dönüşümü ---
def convert_to_picture_fuzzy(value):
    if value == 0:
        return (0.1, 0.0, 0.85)
    elif value == 1:
        return (0.25, 0.05, 0.6)
    elif value == 2:
        return (0.5, 0.1, 0.4)
    elif value == 3:
        return (0.75, 0.05, 0.1)
    elif value == 4:
        return (0.9, 0.0, 0.05)
    elif value == 'R':
        return (0.0, 0.2, 0.0)
    else:
        return (0.0, 0.0, 1.0)

# --- PFN Dönüşümü ---
pf_expert_opinions = np.empty((expert_count, factor_count, factor_count, 3))
for exp in range(expert_count):
    for i in range(factor_count):
        for j in range(factor_count):
            pf_expert_opinions[exp, i, j] = convert_to_picture_fuzzy(expert_opinions[exp, i, j])

# --- PF Decision Matrix ---
PFDec = np.empty((factor_count, factor_count, 3))
for i in range(factor_count):
    for j in range(factor_count):
        mf_product = 1.0
        ind_product = 1.0
        nmf_product = 1.0
        for exp in range(expert_count):
            mf_product *= pf_expert_opinions[exp, i, j, 0] ** (1 / expert_count)
            ind_product *= pf_expert_opinions[exp, i, j, 1] ** (1 / expert_count)
            nmf_product *= (1 - pf_expert_opinions[exp, i, j, 2]) ** (1 / expert_count)
        PFDec[i, j, 0] = mf_product
        PFDec[i, j, 1] = ind_product
        PFDec[i, j, 2] = 1 - nmf_product

# --- Crisp Decision Matrix ---
CrispDec = np.empty((factor_count, factor_count))
for i in range(factor_count):
    for j in range(factor_count):
        mf, ind, nmf = PFDec[i, j]
        CrispDec[i, j] = 0.5 * (1 + 2 * mf - nmf - 0.5 * ind)

# --- IRM ---
threshold_value = np.mean(CrispDec)
IRM = np.zeros((factor_count, factor_count))
for i in range(factor_count):
    for j in range(factor_count):
        if CrispDec[i, j] >= threshold_value or i == j:
            IRM[i, j] = 1

# --- FRM ---
FRM = IRM.copy()
for k in range(factor_count):
    for i in range(factor_count):
        for j in range(factor_count):
            FRM[i, j] = max(FRM[i, j], min(FRM[i, k], FRM[k, j]))
FRM_backup = FRM.copy()

# --- MICMAC Analizi ---
DRIVING_POWER = FRM.sum(axis=1)
DEPENDENCE_POWER = FRM.sum(axis=0)
micmac_df = pd.DataFrame({
    'Factor': [f'Factor {i+1}' for i in range(factor_count)],
    'Driving Power': DRIVING_POWER,
    'Dependence Power': DEPENDENCE_POWER
})
factor_colors = {
    'Driving': 'FF0000',
    'Linkage': '0000FF',
    'Dependent': '008000',
    'Autonomous': 'FFD700'
}
micmac_df['Factor Type'] = micmac_df.apply(lambda row: (
    'Driving' if row['Driving Power'] > factor_count/2 and row['Dependence Power'] <= factor_count/2 else
    'Linkage' if row['Driving Power'] > factor_count/2 and row['Dependence Power'] > factor_count/2 else
    'Dependent' if row['Driving Power'] <= factor_count/2 and row['Dependence Power'] > factor_count/2 else
    'Autonomous'
), axis=1)
micmac_df['Color'] = micmac_df['Factor Type'].map(factor_colors)

# --- MICMAC Scatter Plot ---
plt.figure(figsize=(10, 8))
coordinate_groups = {}
for i in range(factor_count):
    coord = (DRIVING_POWER[i], DEPENDENCE_POWER[i])
    if coord not in coordinate_groups:
        coordinate_groups[coord] = []
    coordinate_groups[coord].append(f'{i+1}')
for coord, factors in coordinate_groups.items():
    color = f'#{micmac_df[micmac_df["Factor"] == f"Factor {factors[0]}"]["Color"].values[0]}'
    label = r"$Chl_{" + ",".join(factors) + r"}$"
    plt.scatter(coord[0], coord[1], color=color)
    plt.text(coord[0] + 0.1, coord[1] + 0.1, label, fontsize=12, color=color, fontweight='bold')
mid = factor_count / 2
plt.axvline(x=mid, color='black', linestyle='--')
plt.axhline(y=mid, color='black', linestyle='--')
axis_max = math.ceil(factor_count / 2) * 2 + 2
plt.xlim(0, axis_max)
plt.ylim(0, axis_max)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(2))
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(2))
plt.gca().set_aspect('equal', adjustable='box')
plt.text(factor_count * 0.9, factor_count * 0.95, "Linkage", fontsize=24, color='blue')
plt.text(factor_count * 0.1, factor_count * 0.95, "Dependent", fontsize=24, color='green')
plt.text(factor_count * 0.9, factor_count * 0.4, "Driving", fontsize=24, color='red')
plt.text(factor_count * 0.1, factor_count * 0.4, "Autonomous", fontsize=24, color='orange')
plt.xlabel('Driving Power', fontsize=16)
plt.ylabel('Dependence Power', fontsize=16)
plt.title('MICMAC Analysis Results', fontsize=24)
plt.grid(True)
plt.tight_layout()
plt.savefig('MICMAC_Results.pdf', format='pdf', bbox_inches='tight')
plt.show()

# --- Faktör Seviyelendirme ---
levels = []
remaining_factors = set(range(factor_count))
level = 1
while remaining_factors:
    current_level_factors = []
    for factor in remaining_factors:
        reachability_set = set(np.where(FRM[factor] == 1)[0])
        antecedent_set = set(np.where(FRM[:, factor] == 1)[0])
        intersection_set = reachability_set & antecedent_set
        if reachability_set == intersection_set:
            current_level_factors.append(factor)
    if not current_level_factors:
        break
    for factor in current_level_factors:
        levels.append({
            'Factor': factor + 1,
            'Level': level,
            'Reachability Set': list(map(int, np.where(FRM[factor] == 1)[0] + 1)),
            'Antecedent Set': list(map(int, np.where(FRM[:, factor] == 1)[0] + 1)),
            'Intersection Set': list(map(int, [x + 1 for x in (set(np.where(FRM[factor] == 1)[0]) & set(np.where(FRM[:, factor] == 1)[0]))]))
        })
        remaining_factors.remove(factor)
        FRM[factor, :] = 0
        FRM[:, factor] = 0
    level += 1
levels_df = pd.DataFrame(levels)

# --- Final Reachability Matrix with Trace ---
FRM_formatted = FRM_backup.astype(str)
for i in range(factor_count):
    for j in range(factor_count):
        if IRM[i, j] == 0 and FRM_backup[i, j] == 1:
            FRM_formatted[i, j] = "1*"

# --- PFDec formatlama (Excel için) ---
def format_pf(pf):
    mf, ind, nmf = pf
    return f"⟨{mf:.2f},{ind:.2f},{nmf:.2f}⟩"

PFDec_str = np.empty((factor_count, factor_count), dtype=object)
for i in range(factor_count):
    for j in range(factor_count):
        PFDec_str[i, j] = format_pf(PFDec[i, j])

# --- Excel'e Yaz ---
with pd.ExcelWriter('PF_ISM_MICMAC_Results.xlsx', engine='openpyxl') as writer:
    micmac_df.drop(columns=['Color']).to_excel(writer, sheet_name='MICMAC Results', index=False)
    pd.DataFrame(CrispDec).to_excel(writer, sheet_name='Crisp Decision Matrix', index=False)
    pd.DataFrame(IRM).to_excel(writer, sheet_name='Initial Reachability', index=False)
    pd.DataFrame(FRM_formatted).to_excel(writer, sheet_name='Final Reachability', index=False)
    levels_df.to_excel(writer, sheet_name='Factor Levels', index=False)
    pd.DataFrame(PFDec_str).to_excel(writer, sheet_name='Picture Fuzzy Matrix', index=False)

# --- Faktör Seviye Diyagramı ---
unique_levels = sorted(levels_df['Level'].unique())
num_levels = len(unique_levels)
fig, ax = plt.subplots(figsize=(8, num_levels * 2))
level_colors = plt.get_cmap('tab20')
for idx, level in enumerate(unique_levels):
    y_bottom = idx
    ax.add_patch(patches.Rectangle((0, y_bottom), 8, 1, color=level_colors(idx % 20), alpha=0.3))
    ax.text(8.5, y_bottom + 0.5, f'Level {level}', fontsize=14, verticalalignment='center')
factor_positions = {}
factor_levels_dict = {}
for level in unique_levels:
    level_factors = levels_df[levels_df['Level'] == level]['Factor'].tolist()
    num_factors = len(level_factors)
    start_x = 4 - (num_factors - 1) / 2
    for i, factor in enumerate(level_factors):
        x_pos = start_x + i
        y_pos = level - 0.5
        size = 0.5
        ax.add_patch(patches.Rectangle((x_pos - size / 2, y_pos - size / 2), size, size, edgecolor='black', facecolor='white'))
        ax.text(x_pos, y_pos, f'{factor}', ha='center', va='center', fontsize=10)
        factor_positions[factor] = (x_pos, y_pos)
        factor_levels_dict[factor] = level
ok_sayisi = 0
for i in range(factor_count):
    for j in range(factor_count):
        if FRM_backup[i, j] == 1 and i != j:
            factor_i = i + 1
            factor_j = j + 1
            level_i = factor_levels_dict[factor_i]
            level_j = factor_levels_dict[factor_j]
            if abs(level_j - level_i) == 1:
                start_pos = factor_positions[factor_i]
                end_pos = factor_positions[factor_j]
                ax.annotate("", xy=(end_pos[0], end_pos[1] + 0.3), xytext=(start_pos[0], start_pos[1] - 0.3),
                            arrowprops=dict(arrowstyle="->", color='black', lw=1))
                ok_sayisi += 1
print(f"\nToplam çizilen ok sayısı: {ok_sayisi}")
ax.set_xlim(0, 9)
ax.set_ylim(0, num_levels + 1)
ax.set_aspect('equal')
ax.axis('off')
plt.title('Factor Levels', fontsize=16)
plt.savefig('Factor_Levels.pdf', format='pdf', bbox_inches='tight')
plt.show()
