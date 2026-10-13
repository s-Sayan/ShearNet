"""Submit, inspect, and resume the training/evaluation job graph.

    shearnet-pipeline submit CONFIG [--time 4-00:00:00]   # from the login node
    shearnet-pipeline status CONFIG
    shearnet-pipeline resume CONFIG [--memory ngmix=48G] [--cpus STAGE=N] [--time STAGE=T]

``status`` and ``resume`` take the config you submitted (or its run directory,
or one generation directory) and use its newest generation. ``status`` says what
each stage is doing, why a stage stopped, and the command that continues it.
``resume`` keeps everything already done and resubmits the rest, with any
resources changed on the command line.

``submit`` makes a new ``generation-*`` directory under the config's
``run_options.outdir`` and submits four jobs: training (GPU), ShearNet
evaluation (GPU, after training), ngmix evaluation (CPUs, right away) and the
merge (after both evaluations). ``--time`` is the limit of each compute job.
"""
import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from ..artifacts.recovery import Interrupted, exclusive
from ..artifacts.runs import RunDir, write_json
from ..config import Config
from ..evaluation.identity import file_digest

STAGES = ("train", "shearnet", "ngmix", "merge")
DEPENDENCIES = {"train": (), "ngmix": (), "shearnet": ("train",),
                "merge": ("shearnet", "ngmix")}
#: Per-stage requests; override any of them with ``--resources FILE`` (JSON).
#: The ShearNet stage renders and predicts on the GPU, but ngmix's metacal image
#: generation (~90 ms per observation) still runs on its CPUs, so the CPU count
#: is what sets that job's length. Every worker process holds ~0.5 GB (it imports
#: ShearNet, so JAX and Flax come with it), and the ngmix job's main process
#: renders on the CPU, so 18 workers do not fit in 16G: the first pilot was
#: OOM-killed there. The merge reads two full catalogs and writes a third (~9 GB
#: at 4M observations); 16G was OOM-killed on all four unit tests.
DEFAULT_RESOURCES = {
    "train": {"cpus": 8, "memory": "128G", "gres": "gpu:1"},
    "shearnet": {"cpus": 16, "memory": "48G", "gres": "gpu:1"},
    "ngmix": {"cpus": 18, "memory": "48G"},
    "merge": {"cpus": 1, "memory": "96G", "time": "02:00:00"},
}

#: Compute-stage time limit when none is given and this is not a Slurm job.
DEFAULT_TIME = "4-00:00:00"


def command(args, **kwargs):
    return subprocess.run(args, check=True, text=True, capture_output=True, **kwargs).stdout.strip()


def seconds(value):
    days, clock = value.split("-", 1) if "-" in value else (0, value)
    parts = list(map(int, clock.split(":")))
    if len(parts) == 1:
        return int(days) * 86400 + parts[0] * 60
    if len(parts) == 2:
        return int(days) * 86400 + parts[0] * 60 + parts[1]
    return int(days) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def job_limit(explicit=None):
    """``--time`` if given, else this Slurm job's own limit (``sbatch --time``)."""
    if explicit:
        seconds(explicit)  # reject a malformed limit before anything is submitted
        return explicit
    if os.environ.get("SLURM_JOB_ID"):
        info = command(["scontrol", "show", "job", os.environ["SLURM_JOB_ID"], "-o"])
        fields = dict(field.split("=", 1) for field in info.split() if "=" in field)
        return fields["TimeLimit"]
    return DEFAULT_TIME


def max_time(partition):
    """A partition's MaxTime in seconds, as Slurm reports it (None if unlimited)."""
    info = command(["scontrol", "show", "partition", partition, "-o"])
    fields = dict(f.split("=", 1) for f in info.split() if "=" in f)
    maximum = fields.get("MaxTime", "UNLIMITED")
    return None if maximum == "UNLIMITED" else seconds(maximum)


def validate_resources(resources, ceiling):
    for stage, resource in resources.items():
        limit = resource.setdefault("time", ceiling)
        if "partition" not in resource:
            # short when the limit fits in it (its queue is separate and quicker)
            short = max_time("short")
            fits = short is None or seconds(limit) <= short
            resource["partition"] = "short" if fits else "long"
        if seconds(limit) > 7 * 86400 or resource["cpus"] < 1:
            raise ValueError(f"invalid resource request for {stage}")
        maximum = max_time(resource["partition"])
        if maximum is not None and seconds(limit) > maximum:
            raise ValueError(f"{stage}: {limit} exceeds {resource['partition']}'s "
                             f"MaxTime of {maximum // 3600} h")
        if stage in ("ngmix", "merge") and resource.get("gres"):
            raise ValueError(f"{stage} must not request a GPU")


def completed(root, stage):
    if stage == "merge":
        return (root / "merged.fits").is_file() and (root / "merge.completed").is_file()
    directory = root / ("training" if stage == "train" else stage)
    status = directory / "status.json"
    return status.exists() and json.loads(status.read_text()).get("state") == "completed"


def submit(config_path, base=None, resources_file=None, overwrite=False, time_limit=None):
    from .train import check_inputs

    ceiling = job_limit(time_limit)  # a bad limit fails before anything is created
    config = Config.from_file(config_path)
    if (config.get("simulation.psf.mode") == "superbit"
            and not config.get("simulation.psf.psfex_file")):
        from ..core.dataset import PSF_DATA_DIR
        psfex = str(Path(PSF_DATA_DIR).resolve())
        config = config.with_overrides({"simulation": {"psf": {"psfex_file": psfex}}})
    check_inputs(config)
    if set(config.get("evaluation.estimators")) != {"shearnet", "ngmix"}:
        raise ValueError("the pipeline requires both estimators; use shearnet-eval for one")
    base = Path(base or config.get("run_options.outdir")).resolve()
    with exclusive(base.parent / ("." + base.name + ".submission.lock")):
        if base.exists() and any(base.iterdir()) and not overwrite:
            raise FileExistsError(f"{base} exists; resume its generation or use --overwrite "
                                  "for a new one")
        root = base / ("generation-" + uuid.uuid4().hex[:12])
        root.mkdir(parents=True)
        frozen = root / "code"
        frozen.mkdir()
        source = Path(__file__).resolve().parents[2]
        shutil.copytree(source / "shearnet", frozen / "shearnet",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        config = config.with_overrides({"run_options": {"outdir": str(root / "training"),
                                                       "ncores": None}})
        config.save(root / "config.yaml")
        resources = {name: dict(values) for name, values in DEFAULT_RESOURCES.items()}
        if resources_file:
            custom = json.loads(Path(resources_file).read_text())
            if set(custom) - set(STAGES):
                raise ValueError("unknown stage in resource file")
            for name, values in custom.items():
                if set(values) - {"cpus", "memory", "gres", "time", "partition"}:
                    raise ValueError(f"unknown resource setting for {name}")
                resources[name].update(values)
        validate_resources(resources, ceiling)
        manifest = {"id": root.name, "resources": resources, "jobs": {}, "attempts": {},
                    "config_sha256": file_digest(root / "config.yaml"),
                    "account": os.environ.get("SLURM_JOB_ACCOUNT"),
                    "qos": os.environ.get("SLURM_JOB_QOS")}
        write_json(root / "pipeline.json", manifest)
        (root / "logs").mkdir()
        resume(root)
        print(f"Pipeline: {root}\nResume: shearnet-pipeline resume {root}")
        return root


def write_stage_script(root):
    """The script every stage job runs. Rewritten on each resume, so an older
    generation picks up fixes to it; the code it runs stays the frozen copy."""
    frozen = root / "code"
    (root / "stage.sh").write_text(
        "#!/bin/bash\nset -euo pipefail\n"
        "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1\n"
        "export JAX_ENABLE_X64=1 MPLBACKEND=Agg\nunset JAX_PLATFORMS\n"
        'if [[ "$1" == ngmix || "$1" == merge ]]; then\n'
        '    export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""\nfi\n'
        # Slurm starts a job where it was submitted -- usually the repository,
        # whose shearnet/ would shadow the frozen copy on `python -m`'s path
        f"cd {shlex.quote(str(root))}\n"
        "export PYTHONSAFEPATH=1\n"
        f"export PYTHONPATH={shlex.quote(str(frozen))}\n"
        f"exec {shlex.quote(sys.executable)} -m shearnet.cli.pipeline stage "
        f'{shlex.quote(str(root))} "$1"\n')


def find_generation(target):
    """The generation directory ``target`` names: a generation itself, or the
    newest generation of a run directory or of the config that was submitted."""
    target = Path(target).expanduser().resolve()
    if (target / "pipeline.json").is_file():
        return target
    if target.is_file():
        base = Path(Config.from_file(target).get("run_options.outdir")).resolve()
    else:
        base = target
    generations = [g for g in base.glob("generation-*") if (g / "pipeline.json").is_file()]
    if not generations:
        raise FileNotFoundError(f"no submitted generation under {base}")
    return max(generations, key=lambda g: (g / "config.yaml").stat().st_mtime)


def change_resources(manifest, root, changes):
    """Apply ``{stage: {setting: value}}`` to the stages still to be submitted."""
    for stage, settings in changes.items():
        if completed(root, stage):
            raise ValueError(f"{stage} has completed; its resources cannot change")
        resource = manifest["resources"][stage]
        resource.update(settings)
        if "time" in settings and "partition" not in settings:
            resource.pop("partition", None)
        validate_resources({stage: resource}, resource["time"])


def resume(root, changes=None):
    root = find_generation(root)
    with exclusive(root / ".submission.lock"):
        manifest = json.loads((root / "pipeline.json").read_text())
        if file_digest(root / "config.yaml") != manifest["config_sha256"]:
            raise ValueError("pipeline configuration changed")
        if changes:
            for stage in changes:
                if _live_state(manifest, stage):
                    raise RuntimeError(f"{stage} is already queued or running "
                                       f"({_live_state(manifest, stage)}); scancel it first "
                                       "to change its resources")
            change_resources(manifest, root, changes)
            write_json(root / "pipeline.json", manifest)
        write_stage_script(root)
        env = {k: v for k, v in os.environ.items() if not k.startswith("SBATCH_")}
        changed = set()
        for stage in STAGES:
            if completed(root, stage):
                continue
            previous = manifest["jobs"].get(stage)
            active = command(["squeue", "--noheader", "--name", f"{manifest['id']}-{stage}",
                              "--format", "%i %T"])
            live = [line.split() for line in active.splitlines() if line.strip()]
            if len(live) > 1:
                raise RuntimeError(f"multiple active jobs for {stage}: {live}")
            if live:
                job, state = live[0]
                if not any(dep in changed for dep in DEPENDENCIES[stage]):
                    if previous != job:
                        changed.add(stage)
                    manifest["jobs"][stage] = job
                    write_json(root / "pipeline.json", manifest)
                    # Submission is held until the whole graph is recorded below.
                    continue
                if state != "PENDING":
                    raise RuntimeError(f"cannot replace dependency of active {stage} job {job}")
                command(["scancel", job])
            r = manifest["resources"][stage]
            args = ["sbatch", "--parsable", "--hold", "--nodes=1", "--ntasks=1",
                    f"--job-name={manifest['id']}-{stage}", f"--partition={r['partition']}",
                    f"--cpus-per-task={r['cpus']}", f"--mem={r['memory']}", f"--time={r['time']}",
                    "--signal=B:USR1@300", "--open-mode=append",
                    f"--output={root}/logs/{stage}-%j.out"]
            if r.get("gres"):
                args.append(f"--gres={r['gres']}")
            else:
                args.append("--gres=none")
            for key in ("account", "qos"):
                if manifest.get(key):
                    args.append(f"--{key}={manifest[key]}")
            deps = [manifest["jobs"][d] for d in DEPENDENCIES[stage] if not completed(root, d)]
            if deps:
                args.append("--dependency=afterok:" + ":".join(deps))
            job = command(args + [str(root / "stage.sh"), stage], env=env).split(";")[0]
            if not job.isdigit():
                raise RuntimeError(f"unexpected sbatch job ID: {job}")
            manifest["jobs"][stage] = job
            manifest["attempts"][stage] = manifest["attempts"].get(stage, 0) + 1
            write_json(root / "pipeline.json", manifest)
            if previous != job:
                changed.add(stage)
        # Release only jobs still held by the user, including a recovered submission.
        for stage, job in manifest["jobs"].items():
            info = command(["squeue", "--noheader", "--name", f"{manifest['id']}-{stage}",
                            "--format", "%r"])
            if info.strip() == "JobHeldUser":
                command(["scontrol", "release", job])
        print(f"Pipeline: {root}")
        for stage in STAGES:
            print(f"  {stage:9s}" + ("completed" if completed(root, stage)
                                     else f"job {manifest['jobs'][stage]}"))
        return root


#: What Slurm's final job states mean for a stage, and what to do about it.
ADVICE = {
    "OUT_OF_MEMORY": "ran out of memory at {memory}",
    "TIMEOUT": "hit its {time} time limit",
    "NODE_FAIL": "lost its node",
    "PREEMPTED": "was preempted",
    "CANCELLED": "was cancelled",
    "FAILED": "failed",
}


def _doubled(memory):
    number, unit = int(memory[:-1]), memory[-1]
    return f"{2 * number}{unit}"


def status(target):
    """Each stage's state, why it stopped, and the command that continues it."""
    root = find_generation(target)
    manifest = json.loads((root / "pipeline.json").read_text())
    print(f"Pipeline: {root}")
    stopped, flags = False, []
    for stage in STAGES:
        job = manifest["jobs"].get(stage)
        resource = manifest["resources"][stage]
        line = "completed" if completed(root, stage) else _live_state(manifest, stage)
        if line is None:
            # sacct cuts columns at 10 characters ("OUT_OF_ME+") unless told not to
            final = command(["sacct", "--noheader", "-X", "--jobs", job,
                             "--format", "State%40"]).split()
            state = final[0] if final else "UNKNOWN"
            why = ADVICE.get(state, "stopped ({state})").format(state=state, **resource)
            line = f"stopped: job {job} {why}"
            record = root / ("training" if stage == "train" else stage) / "status.json"
            error = json.loads(record.read_text()).get("error") if record.is_file() else None
            if error:
                line += f" ({error})"
            stopped = True
            if state == "OUT_OF_MEMORY":
                flags.append(f"--memory {stage}={_doubled(resource['memory'])}")
        progress = root / stage / "progress.json"
        if progress.is_file() and not completed(root, stage):
            saved = json.loads(progress.read_text())
            line += f"; {saved['completed']}/{saved['total']} observations saved"
        print(f"  {stage:9s}{line}")
        print(f"  {'':9s}{resource['cpus']} CPUs, {resource['memory']}, {resource['time']}"
              f"{', ' + resource['gres'] if resource.get('gres') else ''}; "
              f"logs {root / 'logs'}/{stage}-*.out")
    if stopped:
        print("\nTo continue, keeping everything already saved:\n"
              f"  shearnet-pipeline resume {root}{''.join(' ' + f for f in flags)}")
    return root


def _live_state(manifest, stage):
    """``'running (job N)'`` / ``'pending (job N, reason)'``, or None if not queued."""
    queued = command(["squeue", "--noheader", "--name", f"{manifest['id']}-{stage}",
                      "--format", "%i %T %r"]).split()
    if not queued:
        return None
    job, state, reason = (queued + ["", "", ""])[:3]
    detail = f", {reason}" if reason and reason != "None" else ""
    return f"{state.lower()} (job {job}{detail})"


def stage(root, name):
    root = Path(root).resolve()
    manifest = json.loads((root / "pipeline.json").read_text())
    if file_digest(root / "config.yaml") != manifest["config_sha256"]:
        raise ValueError("pipeline configuration changed")
    config = Config.from_file(root / "config.yaml")
    if name in ("train", "shearnet") and "gpu" in manifest["resources"][name].get("gres", ""):
        import jax
        if not any(device.platform == "gpu" for device in jax.devices()):
            raise RuntimeError("this stage requested a GPU, but JAX cannot use one; check the "
                               "CUDA installation")
    if name == "train":
        from ..training.service import train
        train(config, RunDir(root / "training"), resume=True)
    elif name in ("shearnet", "ngmix"):
        from ..evaluation.service import evaluate
        evaluate(config, root / name, estimator=name, resume=True,
                 run=RunDir(root / "training") if name == "shearnet" else None)
    else:
        from ..io.merge import merge
        merge(root / "shearnet/shearnet.fits", root / "ngmix/ngmix.fits", root / "merged.fits")
        from ..artifacts.runs import atomic_write
        atomic_write(root / "merge.completed", b"complete\n")


def _parse_changes(parser, args):
    changes = {}
    for setting in ("memory", "cpus", "time"):
        for item in getattr(args, setting):
            stage, _, value = item.partition("=")
            if stage not in STAGES or not value:
                parser.error(f"--{setting} takes STAGE=VALUE with STAGE one of "
                             f"{', '.join(STAGES)}; got {item!r}")
            changes.setdefault(stage, {})[setting] = int(value) if setting == "cpus" else value
    return changes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("submit")
    p.add_argument("config")
    p.add_argument("directory", nargs="?")
    p.add_argument("--resources")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--time", help="time limit of each compute job (default: this Slurm "
                   f"job's own limit, or {DEFAULT_TIME} outside Slurm)")
    where = "the submitted config, its run directory, or a generation directory"
    p = sub.add_parser("status", help="what each stage is doing")
    p.add_argument("target", help=where)
    p = sub.add_parser("resume", help="resubmit what is unfinished, keeping what is saved")
    p.add_argument("target", help=where)
    for setting, example in (("memory", "ngmix=48G"), ("cpus", "shearnet=8"),
                             ("time", "ngmix=2-00:00:00")):
        p.add_argument(f"--{setting}", action="append", default=[], metavar="STAGE=VALUE",
                       help=f"change a stage's {setting} before resubmitting, e.g. {example}")
    p = sub.add_parser("stage", help=argparse.SUPPRESS)
    p.add_argument("directory")
    p.add_argument("stage", choices=STAGES)
    args = parser.parse_args(argv)
    try:
        if args.action == "submit":
            flags = shlex.split(os.environ.get("SHEARNET_TRAIN_ARGS", ""))
            if any(flag != "--overwrite" for flag in flags):
                parser.error("SHEARNET_TRAIN_ARGS accepts only --overwrite; configure training "
                             "in YAML")
            if os.environ.get("SHEARNET_EVAL_ARGS"):
                parser.error("set evaluation options in the configuration")
            submit(args.config, args.directory, args.resources,
                   args.overwrite or "--overwrite" in flags, args.time)
        elif args.action == "resume":
            resume(args.target, _parse_changes(parser, args))
        elif args.action == "stage":
            stage(args.directory, args.stage)
        else:
            status(args.target)
    except Interrupted as exc:
        print(exc, file=sys.stderr)
        return 75
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
