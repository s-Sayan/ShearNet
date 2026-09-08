#!/bin/bash
#
# Submit every run the paper needs, and nothing else.
#
#   ./research/submit_paper_runs.sh              # DRY RUN: print, submit nothing
#   ./research/submit_paper_runs.sh --go         # actually submit
#   ./research/submit_paper_runs.sh --go --unit-tests
#   ./research/submit_paper_runs.sh --go --tier 1 --tier 3
#   ./research/submit_paper_runs.sh --go --skip-done
#
# Dry run is the default on purpose: this launches 28 full trainings on
# 3x10^5 galaxies, and a typo should cost a second rather than a queue.
#
# WHAT IS HERE, AND WHY EACH ROW EARNS ITS GPU
#
#   4  unit tests   Table: m1 and c2 across the simulation ladder. UT4 is also
#                   the fiducial model, so it supplies every headline number and
#                   the "Fiducial model (all terms)" row of the ablation table.
#   1  tier 1       lambda_PSF. The paper's central claim is that the training
#                   objective controls PSF leakage; this is the arm that tests
#                   it, and either outcome is publishable.
#   8  tier 2       The architecture ladder, galaxy-only up to the fiducial.
#   7  tier 3       Each response objective removed from the fiducial.
#   8  tier 4       Backbone schedule, training transforms, loss function.
#  --
#  28  total
#
# NOT HERE, deliberately:
#
#   The 36-point hyperparameter grid. It is 56% of the compute of the whole
#   campaign for a table no referee asks for, and the text now states that the
#   hyperparameters were fixed a priori with the sweep driver released for
#   reproduction.
#
#   tier4/spatial_dropout. `dropout` reaches only a research_backed branch, so
#   on the fiducial shearnet-d4 backbone the arm trains an identical network.
#   See generate_configs.BLOCKED.
#
# EVERY RUN WRITES AT catalog_level: paper.
#
#   Do NOT switch the ablation arms to `summary`. It looks tempting -- an
#   ablation row reports four scalars and `summary` is kilobytes -- but the
#   alpha column is NOT one of the derived tables. LEAKSUM carries the mean
#   shape and R^PSF only; alpha and beta are fitted by
#   research/shear_bias/leakage_vs_size.py from the per-object LEAKAGE columns
#   (gpsf, Tpsf, e_<est>_raw_ring), and `summary` deletes that table. An arm run
#   at `summary` cannot produce its own alpha.
#
#   At `paper` each run is ~800 MB, so the campaign is ~22 GB.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUB="$REPO/research/unit_test_variations/sub.sh"

[[ -x "$SUB" ]] || { echo "sub.sh not found or not executable: $SUB" >&2; exit 1; }

# --------------------------------------------------------------------------
# The runs. `sub.sh` resolves each of these against research/unit_test_variations,
# research/ablations and research/unit_tests in turn.
# --------------------------------------------------------------------------
UNIT_TESTS=(
    first
    second
    third
    fourth          # UT4 == the fiducial model on the fiducial simulation
)

TIER1=(
    tier1/no_psf_response
)

TIER2=(
    tier2/01_galaxy_only
    tier2/02_psf_branch_concat
    tier2/03_transformer_fusion
    tier2/04_auxiliary_targets
    tier2/05_d4_augmentation
    tier2/06_d4_equivariant
    tier2/07_inloop_rendering
    tier2/08_learned_pooling_head
)

TIER3=(
    tier3/no_gamma_response
    tier3/no_shift_response
    tier3/no_complement
    tier3/no_psf_orbit
    tier3/no_isotropy
    tier3/orbit_k2
    tier3/target_identity
)

TIER4=(
    tier4/no_multiscale_block
    tier4/untrimmed_stem
    tier4/full_resolution_psf_block
    tier4/no_label_normalization
    tier4/no_ema
    tier4/no_image_standardization
    tier4/loss_mae
    tier4/loss_huber
)

# --------------------------------------------------------------------------
GO=0
SKIP_DONE=0
PREFLIGHT=1
SELECTED=()
EXTRA=()

usage() {
    sed -n '2,48p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'USAGE'

Options:
  --go              actually submit (default is a dry run)
  --unit-tests      only the four simulation-ladder rungs
  --tier N          only tier N; repeatable (--tier 1 --tier 3)
  --skip-done       skip a run that already has benchmarking/evaluation.fits
  --no-preflight    skip the static checks (they run by default and gate
                    submission; see research/ablations/preflight.py)
  --no-train        pass --no-train through to sub.sh (evaluate existing
                    checkpoints; useful to re-measure without retraining)
  -h, --help        this
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --go) GO=1; shift ;;
        --unit-tests) SELECTED+=(unit); shift ;;
        --tier) SELECTED+=("tier$2"); shift 2 ;;
        --tier=*) SELECTED+=("tier${1#*=}"); shift ;;
        --skip-done) SKIP_DONE=1; shift ;;
        --no-preflight) PREFLIGHT=0; shift ;;
        --no-train) EXTRA+=(--no-train); shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

# No group named => everything. (Named SELECTED, not GROUPS: `GROUPS` is a bash
# builtin array of the caller's group IDs, and assigning to it silently does
# nothing -- which quietly selects the wrong set of runs rather than erroring.)
if [[ ${#SELECTED[@]} -eq 0 ]]; then
    SELECTED=(unit tier1 tier2 tier3 tier4)
fi

RUNS=()
for group in "${SELECTED[@]}"; do
    case "$group" in
        unit)  RUNS+=("${UNIT_TESTS[@]}") ;;
        tier1) RUNS+=("${TIER1[@]}") ;;
        tier2) RUNS+=("${TIER2[@]}") ;;
        tier3) RUNS+=("${TIER3[@]}") ;;
        tier4) RUNS+=("${TIER4[@]}") ;;
        *) echo "Unknown group: $group" >&2; exit 1 ;;
    esac
done

# Resolve each name to its directory the same way sub.sh does, so a missing
# config is caught here -- before anything is submitted -- rather than 20 jobs in.
resolve() {
    local name="$1" candidate
    for candidate in "$REPO/research/unit_test_variations/$name" \
                     "$REPO/research/ablations/$name" \
                     "$REPO/research/unit_tests/$name" \
                     "$REPO/$name"; do
        if [[ -f "$candidate/config.yaml" ]]; then echo "$candidate"; return 0; fi
    done
    return 1
}

MISSING=()
for name in "${RUNS[@]}"; do
    resolve "$name" >/dev/null || MISSING+=("$name")
done
if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "No config.yaml for:" >&2
    printf '  %s\n' "${MISSING[@]}" >&2
    echo "Run: python research/ablations/generate_configs.py" >&2
    exit 1
fi

# Preflight before anything is submitted. It checks the failures that cost a
# night: a NameError in the training entry point that no unit test reaches, and
# a config the evaluation refuses AFTER training has completed.
if [[ "$PREFLIGHT" -eq 1 ]]; then
    echo "--- preflight ---"
    if ! python "$REPO/research/ablations/preflight.py" --skip-models "${RUNS[@]}"; then
        echo >&2
        echo "Preflight failed. Nothing submitted." >&2
        exit 1
    fi
    echo
fi

echo "=================================================================="
if [[ "$GO" -eq 1 ]]; then
    echo "SUBMITTING ${#RUNS[@]} run(s)"
else
    echo "DRY RUN -- nothing will be submitted. Add --go to submit."
    echo "${#RUNS[@]} run(s) would go:"
fi
echo "=================================================================="

SUBMITTED=0
SKIPPED=0
for name in "${RUNS[@]}"; do
    dir="$(resolve "$name")"
    if [[ "$SKIP_DONE" -eq 1 && -f "$dir/benchmarking/evaluation.fits" ]]; then
        printf '  %-34s SKIP (evaluation.fits exists)\n' "$name"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    if [[ "$GO" -eq 1 ]]; then
        printf '  %-34s ' "$name"
        "$SUB" "$name" ${EXTRA[@]+"${EXTRA[@]}"} | sed -n 's/^Submitted: *//p'
        SUBMITTED=$((SUBMITTED + 1))
    else
        printf '  %-34s %s\n' "$name" "$dir"
    fi
done

echo "=================================================================="
if [[ "$GO" -eq 1 ]]; then
    echo "Submitted $SUBMITTED, skipped $SKIPPED."
    echo "Watch:  squeue -u \$USER"
else
    echo "Nothing submitted. Re-run with --go."
fi
echo "=================================================================="
