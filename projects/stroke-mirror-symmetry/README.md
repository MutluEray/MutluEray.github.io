# Does the brain's mirror symmetry survive stroke?

Many motor-imagery BCI decoders assume that the left and right hemispheres mirror each other. They build this in through mirrored architectures, channel-reflection augmentation or mirror losses. This project tests whether that prior still helps, or starts to hurt, in stroke patients, whose lesion makes the hemispheres unequal. It compares the prior with lesion-side canonicalization.

The idea comes from geometric deep learning (MIT 6.S980, self-study): the mid-sagittal reflection is a group Z2, and a left/right MI decoder should be equivariant under it with the logit changing sign. The lecture's open question is whether we *want* perfect equivariance for groups other than translation. Here that question becomes a clinical one.

**Status:** in progress. Hypotheses and protocol: [`docs/SCOPE.md`](docs/SCOPE.md) (fixed before any stroke analysis).

## Treatments compared

| | |
|---|---|
| `none` | unconstrained decoder |
| `augment` | mirror each training trial with p = 0.5 and flip its label |
| `equivariant` | f(x) = g(x) − g(Px), exactly sign-flipping under the mirror |
| `soft` | mirror-invariant part of the features down-weighted by a, swept from 1 (`none`) to 0 (`equivariant`) |
| `canonical` | mirror right-hemiplegic patients so every affected hand is "left" (stroke only) |

## Data

- Stroke: Liu et al., *Sci Data* 2024, 50 acute stroke patients ([figshare](https://doi.org/10.6084/m9.figshare.21679035), CC BY 4.0).
- Healthy reference: PhysioNet EEG Motor Movement/Imagery, 50 subjects matched in number.

## Run

```bash
pip install -r requirements.txt
python3 -m pytest -q tests
python3 prep_physionet.py                      # healthy cohort
python3 prep_stroke.py --root data/raw/stroke  # stroke cohort + shared channel list + label check
python3 run_ts.py --dataset physionet --channels shared --n-subjects 50   # primary analysis
python3 run_ts.py --dataset stroke --channels shared
```

`run_cv.py` (EEGNet) is a slower secondary check of the final comparisons.

## License

Code: MIT. Data: see the original sources.
