# Paper configs

Every config the paper needs, and nothing else.

| file | what it is |
|---|---|
| `fiducial.yaml` | the fiducial model on the fiducial SuperBIT simulation |
| `unit_tests/{first,second,third,fourth}.yaml` | the simulation ladder, UT1-UT4: the model fixed, the simulation varied |
| `ablations/tier1/` | the PSF-response penalty, the paper's central claim (1 arm) |
| `ablations/tier2/` | the architecture ladder, galaxy-only up to the fiducial (8 arms) |
| `ablations/tier3/` | each response objective removed from the fiducial (7 arms) |
| `ablations/tier4/` | backbone schedule, training transforms, loss function (8 arms) |

Every arm is `fiducial.yaml` with a small named set of keys changed. They are
**generated**, not hand-written:

```
python configs/paper/generate.py           # rewrite them all
python configs/paper/generate.py --check   # verify, write nothing
```

Editing an arm by hand is undone by the next regeneration and caught by
`tests/test_ablation_configs.py`. Change the fiducial, or the arm's entry in
`generate.py`, and re-run. Each generated file opens with the keys that
changed, why the arm exists, and anything to know before trusting its number.
The generator validates every arm against the config schema, so a delta naming
a key the model would ignore stops there instead of training the control.

Each arm's run directory is `/home/adfield/ShearNet/runs/<arm path>` (its
`run_options.outdir`).

## Submitting

`generate.py` also writes `runs.txt`, one arm per line, for the one Slurm
script. From the repository root:

```
sbatch --array=0-27%4 scripts/shearnet.sbatch --list configs/paper/runs.txt
```

Each array task trains its arm and then evaluates it with the arm's own
`evaluation` block, so the run directory ends up holding the model and its
catalog, `evaluations/default/<run_name>_default.fits`. One arm by hand:

```
sbatch scripts/shearnet.sbatch configs/paper/ablations/tier3/orbit_k2.yaml
```

UT4 *is* the fiducial (same keys, different name), so `fiducial.yaml` is not in
the list; it exists to be the base the arms are written from.

### Tier 2 is cumulative

Rungs 1-8 each add one component to the rung above, so a row differs from the
*fiducial* in several keys while differing from its *predecessor* in one. Read
the table by adjacent differences. Rungs 1-6 have no `training.response` block
and use `generation: upfront`, because the response penalties only exist once
the renderer is inside the autodiff graph -- that is what "before in-loop
rendering" means, not an extra variable.

### One arm is documented but not generated

`tier4/spatial_dropout`. `model.dropout` reaches only a `research_backed`
branch; on the fiducial `shearnet-d4` backbone it does nothing (the config
schema refuses it), so the arm would train the fiducial again. It needs dropout
plumbed into `_ShearNetD4Backbone` first, and is listed in `generate.BLOCKED`
so it cannot be quietly forgotten.

### Loss and augmentation settings

`tier4/loss_mae` and `tier4/loss_huber` select their objectives through
`training.loss`; `tier2/05_d4_augmentation` enables `training.d4_augment`.
Confirm these settings in the resolved run config when interpreting results.

### Two things to check against the paper before writing the table

- **EMA and input standardization are ON in the fiducial.** The appendix table
  lists them as additions to a baseline, so those rows are *removals* here and
  the signs need to match.
- **`orbit_k` is 4 in the fiducial**, so `tier3/orbit_k2` is the ablation.
