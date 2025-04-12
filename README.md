# PF-ISM-MICMAC Analysis Tool

This repository provides a Python implementation of the PF-ISM-MICMAC methodology, which combines Picture Fuzzy Interpretive Structural Modeling (PF-ISM) with MICMAC (Matrice d’Impacts Croisés Multiplication Appliquée à un Classement) analysis. It allows researchers and decision-makers to analyze complex systems involving uncertainty, ambiguity, and interdependencies among multiple factors.

---

## 📌 Key Features

- Supports Picture Fuzzy linguistic evaluations
- Aggregates multiple expert opinions using a weighted geometric operator
- Calculates crisp decision matrix from fuzzy input
- Constructs initial and final reachability matrices
- Generates a hierarchical ISM structure (factor levels)
- Performs MICMAC analysis (driving vs. dependence power)
- Outputs:
  - `PF_ISM_MICMAC_Results.xlsx` — All matrices and classification results
  - `MICMAC_Results.pdf` — MICMAC scatter plot
  - `Factor_Levels.pdf` — ISM-based factor hierarchy

---

## 📁 Input Format

The script reads expert evaluations from an Excel file selected via a file dialog. The Excel file should include:

- A square matrix for each expert (factor × factor)
- All expert matrices stacked vertically (no empty rows/columns)
- Evaluations using predefined linguistic terms: 0–4 or 'R'

---

## 🧮 Linguistic Scale to Picture Fuzzy Values

| Linguistic Term | Membership | Indeterminacy | Non-membership |
|-----------------|------------|---------------|----------------|
| 0               | 0.10       | 0.00          | 0.85           |
| 1               | 0.25       | 0.05          | 0.60           |
| 2               | 0.50       | 0.10          | 0.40           |
| 3               | 0.75       | 0.05          | 0.10           |
| 4               | 0.90       | 0.00          | 0.05           |
| R (Refusal)     | 0.00       | 0.20          | 0.00           |

---

## ⚙️ Dependencies

- Python 3.8+
- `pandas`
- `numpy`
- `matplotlib`
- `openpyxl`

Install dependencies with:

```bash
pip install pandas numpy matplotlib openpyxl
▶️ How to Run
Simply run the script:

bash
Kopyala
Düzenle
python PF_ISM_MICMAC.py
You will be prompted to select your Excel file containing expert judgments. Once selected, all calculations and visualizations will be generated automatically and saved in the working directory.

📤 Outputs
PF_ISM_MICMAC_Results.xlsx
Contains:

Picture Fuzzy Contextual Relationship Matrix

Crisp Decision Matrix

Initial and Final Reachability Matrices

Factor Levels

MICMAC Classification Table

MICMAC_Results.pdf
Scatter plot dividing challenges into four quadrants: Driving, Dependent, Linkage, Autonomous

Factor_Levels.pdf
Hierarchical ISM structure showing levels and inter-factor influences

📄 License
This project is open-source and licensed under the MIT License.

✍️ Author
Ahmet Öztel
Bartın University
aoztel@bartin.edu.tr
