Machine Learning Analysis of Electron Affinity in Actinide Carboxylate Complexes

This repository provides the complete computational workflow used for the machine learning analysis of electron affinity (EA) in actinide complexes of the form:

([M(propionate)₃]⁺ and [M(acrylate)₃]⁺; 

The study combines DFT-derived electronic descriptors with machine learning models to investigate the electronic structure and reactivity trends across the actinide series from Th to Lr.

Scientific Scope

The work focuses on understanding how metal electronic structure and ligand environment influence the electron affinity of actinide complexes.

Two ligand systems are considered:

• acrylate ligands
• propionate ligands

These ligands provide a useful comparison between saturated and unsaturated carboxylate coordination environments.

Dataset

The dataset includes complexes across the actinide series:

Th, Pa, U, Np, Pu, Am, Cm, Bk, Cf, Es, Fm, Md, No, Lr

For each complex the following descriptors were used:

Atomic number
Number of f-electrons
Covalent radius
Electronegativity
Spin–orbit coupling parameter
Metal charge
ADCH charges
Average M–C bond distances
Average M–O bond distances

Target property:

Electron Affinity (EA) obtained from DFT calculations.

Machine Learning Workflow

The workflow performs:

Dataset construction
Feature engineering
Hyperparameter optimization
Model comparison
Cross-validation
Model interpretation

The primary predictive model used is:

ExtraTreesRegressor

with 7-fold cross-validation.

Model Validation

To ensure robustness, several validation techniques are implemented:

• Y-Randomization test
• Bootstrap uncertainty analysis
• Applicability domain analysis (Williams plot)

Explainable AI

Feature contributions were analyzed using:

SHAP (SHapley Additive exPlanations)

Two interpretability plots are generated:

SHAP summary plot
SHAP dependence plot for f-electrons

These analyses help identify the descriptors controlling electron affinity across the actinide series.

Generated Figures

The workflow automatically generates high-quality figures:

• Parity plot (DFT vs ML predictions)
• Learning curve
• SHAP plots
• Y-randomization distribution
• Bootstrap uncertainty distribution
• Williams applicability domain plot

All figures are exported in 600 dpi resolution suitable for publication.

Output Files

Running the workflow produces:

ElectronAffinity_Predictions.csv

containing:

| Metal | Ligand | EA_DFT | EA_predicted | Error |

and the trained model:

Best_ExtraTrees_Model.pkl

Author

Ali Khairbek
Postdoctoral Researcher
Computational and Theoretical Chemistry