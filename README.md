# CytokineSignalingEngine

**NF-κB and JAK-STAT Signaling ODE Models**

A pure-Python pipeline for cytokine-driven signaling dynamics using ODE models of NF-κB and JAK-STAT pathways.

## Features
- NF-κB activation ODE (IκB/IKK/NF-κB dynamics, Lipniacki model)
- JAK-STAT signaling ODE (STAT phosphorylation/dephosphorylation)
- Cytokine dose-response modeling (TNF, IL-6, IFN-γ, IL-1β)
- Signaling crosstalk (SOCS-mediated NF-κB/STAT inhibition)
- Inflammatory gene expression signature scoring

## Results
- 100 cell lines × 4 cytokine stimulations
- Peak NF-κB (TNF=1.0 ng/mL): 0.778
- Peak STAT (IL-6=1.0 ng/mL): 0.359
- NF-κB oscillation period: 4.1 min (14 peaks)
- SOCS suppression range: 0.707-0.771

## Usage
```bash
pip install numpy scipy matplotlib
python cytokine_signaling_engine.py
```

## Tags
`cytokine` `innate-immunity` `nf-kb` `jak-stat` `inflammation-network` `ode-model`
