# Ring-based phase identification in Inconel 718 powder streams: code and processed data

This repository reproduces the analysis of fourteen selected-area electron diffraction ring patterns from virgin, reuse, waste and filter Inconel 718 powder from laser powder bed fusion. Each pattern is searched against 575 plausible crystal structures with five scoring rules and tested against 20 decoy libraries, and recovery tests on 3894 synthetic patterns of known structure give a reliability map against ring count and calibration error. SEM particle shape, EDS metal fractions and XPS surface chemistry of the four streams are analysed from the same processed data.

The raw TEM, SEM, EDS and XPS files are not included. They are available from the corresponding author on reasonable request.

## Running

    pip install -r environment/requirements.txt
    cd code
    ./run

The whole run takes about fifteen minutes on an 8-core machine, most of it for the comparison of five scoring rules. Results go to `results/`: CSV tables, two short text summaries (`summary_diffraction.txt`, `summary_surfaces.txt`) and a few plots.

Python 3.10 or newer with numpy, scipy and matplotlib (`environment/requirements.txt`). Nothing else is needed. The crystal structures are read from the CSV files, so no crystallography package is required.

## Code

| File | What it does |
|---|---|
| `diffraction.py` | Ring spottiness and lattice-fringe summaries, the phase search of each ring pattern against 575 candidate structures and 20 decoy libraries, the recovery test on synthetic patterns, the reliability grid (ring count × calibration error), the geometric control for crossed fringes, and the comparison of five scoring rules, including the whole-profile score |
| `surfaces.py` | SEM particle shape (waste and filter) with Mann–Whitney tests, whole particles seen in TEM, EDS metal fractions, and XPS fits (Shirley background, Gaussian–Lorentzian components, Scofield factors) |
| `figures.py` | Plots of the main results |
| `run` | Runs the three scripts in order |

The random seeds are fixed, so a run repeats the numbers in the paper: the decoy p values of all 14 patterns, the recovery test (294 patterns), the reliability grid (3600 patterns), the comparison of scoring rules (`scoring_rules_compared.csv`, with the selection set of 608 patterns), the geometric control for crossed fringes, the SEM statistics and the XPS table. Results for the whole-profile score are in `phase_search_all_rules.csv` and `reliability_grid_all_rules.csv`. Ties at the top of a ranking are shared in the comparison of rules, so the score S there can differ from `reliability_grid.csv` by about one percentage point.

## Data

All files are plain CSV with one header row. Streams are `virgin`, `reuse`, `waste` and `filter`. Pattern and sample codes (V, R1–R7, W, F) are those in the paper. Image numbers are replaced by neutral identifiers (T001, S01, P01, ...).

| File | Contents |
|---|---|
| `diffraction_rings.csv` | Every ring of the 14 SAED patterns: spacing `d_A` (Å), prominence above noise, effective number of diffracting spots `n_eff`, full width at half maximum `fwhm_per_A` (1/Å) |
| `radial_profiles.csv` | Baseline-subtracted radial profiles of the 14 patterns, `q_per_A` = 1/d (1/Å) |
| `lattice_fringes.csv` | The 1238 lattice fringes from image power spectra: image, pixel size (Å), spacing (Å), orientation (°), strength over background, window position (pixels) |
| `fringe_pairs.csv` | The 671 crossed-fringe crystals: two spacings and the angle between them |
| `dark_feature_size.csv` | Median size of dark-contrast features per image (granulometry, nm) |
| `hand_tracing_check.csv` | Hand-traced crystallite size against granulometry for three images |
| `candidate_structures.csv` | The 978 structures from the Crystallography Open Database: COD id, formula, space group, mineral name, lattice vectors `a1x … a3z` (Å) |
| `candidate_atoms.csv` | Atoms of each structure in P1: element, occupancy, fractional coordinates |
| `sem_particles.csv` | Particles outlined in SEM images: equivalent diameter (µm), circularity, solidity, aspect ratio |
| `tem_whole_particles.csv`, `tem_edge_roughness.csv` | Diameters of particles too thick for the beam, and root-mean-square roughness of their edges |
| `eds_sum_spectra.csv` | EDS map sum spectra as printed by the acquisition software (wt%, ±1σ). Oxygen was not quantified |
| `xps_spectra.csv` | All XPS spectra: survey and narrow scans, binding energy as recorded (before charge correction) and counts. Virgin and reuse were measured with Mg Kα, waste and filter with Al Kα |

The candidate structures come from the Crystallography Open Database (Gražulis et al., Nucleic Acids Res. 40 (2012) D420), which is in the public domain.

## Notes

- The plausible subset of the candidate library (575 structures) is selected in the code from the chemical formula, by the rule given in the paper.
- XPS quantities from the two X-ray sources are compared through chemical states, O 1s components and the Ti/Cr ratio only.
- The lattice-fringe survey, granulometry and particle segmentation were run on the raw images, so this capsule starts from their outputs.

## Licence

Code: MIT (`LICENSE`). Data in `data/`: Creative Commons Attribution 4.0 International (`LICENSE-DATA`).
