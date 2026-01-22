# ASE → RMG Thermochemistry Pipeline for Surface Adsorbates

This repository contains a set of notebooks and helper scripts for converting DFT/ASE
outputs for surface adsorbates into RMG-compatible thermochemistry libraries using
NASA polynomials.

The workflow takes:

- ASE trajectory files  
- Vibrational frequencies + ZPE from DFT  
- Generates per-species thermo  
- Fits NASA polynomials  
- Produces RMG thermo library entries  

---

## Dependencies

Core requirements:

- Python ≥ 3.8  
- NumPy  
- SciPy  
- Pandas  
- Matplotlib  
- ASE (Atomic Simulation Environment)

---

## Workflow Overview

1. **Preprocess DFT outputs**

Run: `standardize_inputs.ipynb`

This prepares:
- standardized trajectory names  
- vibrational + ZPE logs  

---

2. **Generate NASA polynomials and convert to RMG format**

Run: `compute_NASA_for_<surface>-adsorbates.ipynb`

This generates an RMG-ready thermo library.

