# Data

## Included in this repository

| File | Content |
|---|---|
| `tool_ids.csv` | Index linking each workpiece number of the campaign to its run (T1 to T18). It contains no measurements. |
| `phm2010/phm2010_wear_percut.csv` | Public PHM Society 2010 milling data: maximum flank wear over the three flutes after every cut for the three labelled cutters (c1, c4, c6), 315 cuts each. Used only for the external check in Appendix A. |

The PHM 2010 data come from the [2010 PHM Society Conference Data Challenge](https://phmsociety.org/phm_competition/2010-phm-society-conference-data-challenge/).

## Available from the authors on request

The wear records of the experimental campaign are available from the authors on reasonable request,
subject to laboratory restrictions. To run the full analysis, place them in this folder under these
names:

| File | Content | Used by |
|---|---|---|
| `DOE.xlsx` | DOE workbook, sheet `DoE2`: one row per workpiece (183 workpieces) with run, strategy, cutting condition, VBmax and notes | scripts 01 to 07 |
| `vibration_features.csv` | One row per machined part: `experiment_id`, `tool_id`, `within_tool_order`, `vc`, `fz`, `cooling` and 294 vibration features | script 02 |
| `micrographs/` | `micro_a_early_133um.jpg`, `micro_b_developed_220um.jpg`, `micro_c_chipping.jpg` | script 07 (Figure 2) |

### Columns of `DOE.xlsx` used by the analysis

Sheet `DoE2`: column 1 workpiece number, 2 tool number, 3 machining strategy (`OD` for the short tests,
`Long Time (OD)` for the long tests), 5 cooling, 6 cutting speed vc (m/min), 7 feed f (mm/rev),
11 VBmax (µm) and 20 notes (`Tool Breakage`).

### Vibration features

Axial (`A_`) and radial (`R_`) vibration sampled at 50 kHz. Each part is machined in six contacts; each
contact is segmented and described by time- and frequency-domain features, which are aggregated over
the contacts of the part, giving 294 features per part.
