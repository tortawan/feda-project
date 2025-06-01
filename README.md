# FEDA Project: A Forest-guided Estimation of Distribution Algorithm for Optimization

This project presents the implementation and evaluation of the **Forest-guided Estimation of Distribution Algorithm (FEDA)**, specifically the RF-MIMIC variant. This algorithm leverages a Random Forest classifier to model the distribution of elite solutions and guide the generation of new candidate solutions for complex optimization problems.

To benchmark performance, we also implement two variants of the MIMIC (Mutual Information Maximizing Input Clustering) algorithm:
- `MIMIC_O2`: uses a dependency tree based on pairwise mutual information.
- `MIMIC_MY`: assumes independence among variables and uses marginal probabilities only.

All algorithms are evaluated on the **NK-Landscape** problem—a tunable benchmark for rugged fitness landscapes.

---

## 🔍 Core Algorithm: RF-MIMIC (Forest-guided EDA)

**Key Concept**: RF-MIMIC evolves a population of candidate solutions by training a Random Forest classifier to differentiate elite (high-performing) solutions from non-elites. It then samples new candidates by probabilistically traversing the trained trees, effectively emulating the characteristics of the elite group.

**Highlights**:
- Uses scikit-learn's `RandomForestClassifier`
- Samples new solutions by biased tree traversal
- Handles sparse data gracefully

📂 File: `feda_algorithm/rf_mimic.py`  
🧱 Class: `RF_MIMIC`

---

## ⚖️ Comparative Algorithms

### MIMIC_O2 (Dependency Tree)
Constructs a tree of dependencies among variables based on mutual information from elite samples.

- Captures pairwise dependencies
- Uses a probabilistic model for sequential sampling
- Includes fallback diversity mechanisms

📂 File: `feda_algorithm/mimic_o2.py`  
🧱 Class: `MIMIC_O2`

### MIMIC_MY (Marginal Probabilities)
A lightweight variant using only independent marginal probabilities.

- Computationally efficient
- Does not model variable dependencies

📂 File: `feda_algorithm/mimic_my.py`  
🧱 Class: `MIMIC_MY`

---

## 📁 Project Structure
```
feda-project/
├── examples/                 # Scripts to run experiments
├── feda_algorithm/          # Optimizers: RF_MIMIC, MIMIC variants
├── problem_definitions/     # NK-Landscape generator
├── utils/                   # Debug logging, helper tools
├── images/                  # Result plots (fitness curves)
├── requirements.txt         # Python dependencies
└── README.md
```

---

## ⚙️ Setup Instructions

### Prerequisites
- Python 3.8+
- `pip`

### Installation
```bash
git clone https://github.com/tortawan/feda-project.git
cd feda-project
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate (Windows)
pip install -r requirements.txt
```

---

## 🚀 Running the Example

Use the included script to compare all algorithms on a defined NK-Landscape:
```bash
python examples/run_feda_nk.py
```

To enable debug logs:
```python
# utils/debugging.py
DEBUG_MODE = True
```

---

## 📊 Result Interpretation
Each algorithm will output:
- Fitness landscape configuration (N, K)
- Per-iteration fitness logs (optional)
- Final solution statistics (fitness, time)
- A plot of average population fitness vs. iteration

Key metrics:
- **Best fitness**: quality of final solution
- **Average fitness**: population-wide convergence trend
- **Execution time**: efficiency

---

## 📈 Experimental Results

### K=1 (Low Interaction)
![K=1](images/50_1.png)
Simple models excel; RF-MIMIC catches up later.

### K=5 (Moderate Interaction)
![K=5](images/50_5.png)
MIMIC_O2 is strong early, RF-MIMIC converges well.

### K=10 (High Interaction)
![K=10](images/50_10.png)
RF-MIMIC outperforms consistently with stronger modeling.

### K=20 (Very High Interaction)
![K=20](images/50_20.png)
Only RF-MIMIC shows significant progress under high complexity.

---

## 🧠 Key Insights

- For **simple landscapes**, marginal and pairwise models are sufficient.
- For **complex problems**, RF-MIMIC is superior due to its flexible and powerful modeling capabilities.
- RF-MIMIC is computationally more intensive, but yields better fitness and population quality in hard landscapes.

---

## 🏁 Conclusion
This project demonstrates how Random Forests can guide probabilistic model-based optimization algorithms effectively. The **RF-MIMIC** approach adapts well across complexity levels and shows strong potential in high-dimensional, epistatic optimization tasks.

> 💡 This project combines ML, probabilistic modeling, and combinatorial optimization in a clean, modular, and extensible way—ideal for recruiters evaluating algorithmic thinking and coding skills.
