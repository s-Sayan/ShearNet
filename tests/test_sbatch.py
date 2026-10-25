"""The dispatcher allocates no GPU and submits recoverable dependencies."""
import json
from pathlib import Path
import pytest
from shearnet.cli import pipeline


class Slurm:
    def __init__(self):
        self.jobs = {}
        self.submissions = []
        self.next_id = 100
        self.fail_once = False
        self.final = {}  # job -> sacct State once it has left the queue

    def __call__(self, args, **kwargs):
        if args[:3] == ["scontrol", "show", "partition"]:
            # Turing's limits
            return "MaxTime=1-00:00:00" if args[3] == "short" else "MaxTime=7-00:00:00"
        if args[:3] == ["scontrol", "show", "job"]:
            return "TimeLimit=4-00:00:00"
        if args[0] == "squeue":
            name = args[args.index("--name") + 1]
            jobs = [(job, value) for job, value in self.jobs.items() if value["name"] == name]
            if args[-1] == "%r":
                return "\n".join(value["reason"] for _, value in jobs)
            return "\n".join(f"{job} {value['state']}" for job, value in jobs)
        if args[0] == "sbatch":
            job = str(self.next_id)
            self.next_id += 1
            name = next(a.split("=", 1)[1] for a in args if a.startswith("--job-name="))
            self.jobs[job] = {"name": name, "state": "PENDING", "reason": "JobHeldUser"}
            self.submissions.append(args)
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError("connection lost after sbatch accepted job")
            return job
        if args[:2] == ["scontrol", "release"]:
            self.jobs[args[2]]["reason"] = "Resources"
            return ""
        if args[0] == "sacct":
            assert args[-1] == "State%40", "sacct truncates State to 10 characters"
            return self.final.get(args[args.index("--jobs") + 1], "COMPLETED")
        if args[0] == "scancel":
            del self.jobs[args[1]]
            return ""
        raise AssertionError(args)


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    slurm = Slurm()
    monkeypatch.setattr(pipeline, "command", slurm)
    monkeypatch.setenv("SLURM_JOB_ID", "99")
    config = tmp_path / "config.yaml"
    config.write_text("run_options: {run_name: tiny}\nsimulation: {backend: jax-galsim}\n")
    return slurm, config, tmp_path / "runs"


def test_split_graph_and_cpu_resource_isolation(experiment):
    slurm, config, base = experiment
    root = pipeline.submit(config, base)
    jobs = json.loads((root / "pipeline.json").read_text())["jobs"]
    calls = {args[-1]: args for args in slurm.submissions}
    assert f"--dependency=afterok:{jobs['train']}" in calls["shearnet"]
    assert f"--dependency=afterok:{jobs['shearnet']}:{jobs['ngmix']}" in calls["merge"]
    assert not any(a.startswith("--dependency") for a in calls["ngmix"])
    for name in ("ngmix", "merge"):
        assert "--gres=none" in calls[name]
    assert "--gres=gpu:1" in calls["train"]
    # the timed jobs are sized like NERSC Perlmutter nodes
    assert "--gres=gpu:A100:1" in calls["shearnet"]
    assert "--cpus-per-task=16" in calls["shearnet"] and "--mem=64G" in calls["shearnet"]
    assert "--cpus-per-task=128" in calls["ngmix"] and "--mem=480G" in calls["ngmix"]
    assert all(j["reason"] != "JobHeldUser" for j in slurm.jobs.values())
    assert (root / "code/shearnet/evaluation/service.py").is_file()
    script = Path(__file__).parents[1] / "scripts/shearnet.sbatch"
    assert "#SBATCH --gres=gpu" not in script.read_text()


def test_partial_submission_is_adopted_without_duplicate_jobs(experiment):
    slurm, config, base = experiment
    slurm.fail_once = True
    with pytest.raises(RuntimeError, match="connection lost"):
        pipeline.submit(config, base)
    root = next(base.glob("generation-*"))
    pipeline.resume(root)
    assert len(slurm.submissions) == 4
    assert all(j["reason"] != "JobHeldUser" for j in slurm.jobs.values())


def test_resume_repairs_failed_dependencies(experiment):
    slurm, config, base = experiment
    root = pipeline.submit(config, base)
    old = json.loads((root / "pipeline.json").read_text())["jobs"]
    del slurm.jobs[old["train"]]
    pipeline.resume(root)
    new = json.loads((root / "pipeline.json").read_text())["jobs"]
    assert new["train"] != old["train"]
    assert new["shearnet"] != old["shearnet"]
    assert new["merge"] != old["merge"]
    assert new["ngmix"] == old["ngmix"]


def test_overwrite_creates_a_generation_without_removing_results(experiment):
    slurm, config, base = experiment
    old = pipeline.submit(config, base)
    marker = old / "keep-me"
    marker.write_text("results")
    new = pipeline.submit(config, base, overwrite=True)
    assert new != old and marker.read_text() == "results"


def test_default_psf_path_is_resolved_before_freezing_the_package(experiment, monkeypatch,
                                                                  tmp_path):
    import shearnet.core.dataset as dataset
    slurm, config, base = experiment
    psfs = tmp_path / "psfs"
    psfs.mkdir()
    monkeypatch.setattr(dataset, "PSF_DATA_DIR", str(psfs))
    config.write_text("run_options: {run_name: tiny}\n"
                      "simulation: {backend: jax-galsim, psf: {mode: superbit}}\n")
    root = pipeline.submit(config, base)
    from shearnet.config import Config
    frozen = Config.from_file(root / "config.yaml")
    assert frozen.get("simulation.psf.psfex_file") == str(psfs)


def test_gpu_stage_refuses_a_cpu_only_jax_installation(experiment, monkeypatch):
    import jax
    from types import SimpleNamespace
    _, config, base = experiment
    root = pipeline.submit(config, base)
    monkeypatch.setattr(jax, "devices", lambda: [SimpleNamespace(platform="cpu")])
    with pytest.raises(RuntimeError, match="requested a GPU"):
        pipeline.stage(root, "train")
    assert not (root / "training").exists()


def test_submit_from_the_login_node_takes_its_time_limit(experiment, monkeypatch):
    """Run with bash instead of sbatch, the dispatcher never waits in a queue."""
    slurm, config, base = experiment
    monkeypatch.delenv("SLURM_JOB_ID")
    pipeline.submit(config, base, time_limit="12:00:00")
    calls = {args[-1]: args for args in slurm.submissions}
    for stage in ("train", "shearnet", "ngmix"):
        assert "--time=12:00:00" in calls[stage] and "--partition=short" in calls[stage]
    assert "--time=02:00:00" in calls["merge"]
    with pytest.raises(ValueError):
        pipeline.submit(config, base / "other", time_limit="soon")
    assert not (base / "other").exists()


def test_stages_run_the_frozen_code_not_the_repository(experiment, monkeypatch):
    """A stage must import the generation's copy of ShearNet even when the job
    starts in the repository, or resuming after a git pull mixes two versions."""
    import os
    import subprocess

    _, config, base = experiment
    root = pipeline.submit(config, base)
    frozen = root / "code/shearnet/cli/pipeline.py"
    frozen.write_text("import os, sys\nif os.environ.get('SHEARNET_PROBE'):\n"
                      "    print(__file__)\n    sys.exit(0)\n" + frozen.read_text())
    repository = Path(__file__).parents[1]
    env = dict(os.environ, SHEARNET_PROBE="1")
    env.pop("SLURM_JOB_ID", None)
    result = subprocess.run(["bash", str(root / "stage.sh"), "merge"], cwd=repository,
                            env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(frozen)


def _finish(slurm, root, stage, state):
    """Take a stage's job out of the queue with a final Slurm state."""
    job = json.loads((root / "pipeline.json").read_text())["jobs"][stage]
    del slurm.jobs[job]
    slurm.final[job] = state
    return job


def test_status_explains_an_oom_and_resume_raises_the_memory(experiment, capsys):
    slurm, config, base = experiment
    root = pipeline.submit(config, base)
    for job in slurm.jobs.values():
        job["state"] = "RUNNING"
    killed = _finish(slurm, root, "ngmix", "OUT_OF_MEMORY")
    capsys.readouterr()
    pipeline.status(base)
    shown = capsys.readouterr().out
    assert f"stopped: job {killed} ran out of memory at 480G" in shown
    assert f"shearnet-pipeline resume {root} --memory ngmix=960G" in shown

    # the merge waits on the dead job, so it is replaced along with it
    merge = json.loads((root / "pipeline.json").read_text())["jobs"]["merge"]
    slurm.jobs[merge]["state"] = "PENDING"
    before = len(slurm.submissions)
    assert pipeline.main(["resume", str(base), "--memory", "ngmix=960G"]) == 0
    calls = {args[-1]: args for args in slurm.submissions[before:]}
    assert set(calls) == {"ngmix", "merge"}
    assert "--mem=960G" in calls["ngmix"]
    jobs = json.loads((root / "pipeline.json").read_text())["jobs"]
    assert f"--dependency=afterok:{jobs['shearnet']}:{jobs['ngmix']}" in calls["merge"]


def test_resume_and_status_find_the_newest_generation_from_the_config(experiment, capsys):
    slurm, config, base = experiment
    config.write_text(f"run_options: {{run_name: tiny, outdir: {base}}}\n"
                      "simulation: {backend: jax-galsim}\n")
    first = pipeline.submit(config)
    newest = pipeline.submit(config, overwrite=True)
    assert pipeline.find_generation(config) == newest
    assert pipeline.find_generation(base) == newest
    assert pipeline.find_generation(first) == first
    pipeline.status(config)
    assert f"Pipeline: {newest}" in capsys.readouterr().out
    with pytest.raises(FileNotFoundError, match="no submitted generation"):
        pipeline.find_generation(base / "nothing-here")


def test_resume_refreshes_the_stage_script(experiment):
    slurm, config, base = experiment
    root = pipeline.submit(config, base)
    (root / "stage.sh").write_text("#!/bin/bash\nexec python -m shearnet.cli.pipeline stage\n")
    pipeline.resume(root)
    assert f"cd {root}" in (root / "stage.sh").read_text()


def test_resources_of_a_queued_stage_are_not_changed_behind_its_back(experiment):
    slurm, config, base = experiment
    root = pipeline.submit(config, base)
    with pytest.raises(RuntimeError, match="scancel it first"):
        pipeline.resume(root, {"ngmix": {"memory": "96G"}})
    resources = json.loads((root / "pipeline.json").read_text())["resources"]
    assert resources["ngmix"]["memory"] == "480G"


def test_the_partition_follows_slurms_own_limits(experiment):
    """short is a day on Turing: a longer limit goes to long, and asking short
    for more than it allows is refused before anything is submitted."""
    slurm, config, base = experiment
    root = pipeline.submit(config, base, time_limit="1-12:00:00")
    calls = {args[-1]: args for args in slurm.submissions}
    assert "--partition=long" in calls["ngmix"] and "--partition=short" in calls["merge"]

    def unqueue(stage):
        for job in [j for j, v in slurm.jobs.items() if v["name"].endswith(f"-{stage}")]:
            del slurm.jobs[job]

    unqueue("train")
    unqueue("shearnet")
    pipeline.resume(root, {"train": {"time": "20:00:00"}})
    train = [args for args in slurm.submissions if args[-1] == "train"][-1]
    assert "--time=20:00:00" in train and "--partition=short" in train
    unqueue("shearnet")
    before = json.loads((root / "pipeline.json").read_text())["resources"]
    with pytest.raises(ValueError, match="exceeds short's MaxTime of 24 h"):
        pipeline.resume(root, {"shearnet": {"time": "2-00:00:00", "partition": "short"}})
    assert json.loads((root / "pipeline.json").read_text())["resources"] == before


def _complete(slurm, root, stage):
    """Mark a stage finished on disk and take its job out of the queue."""
    if stage == "merge":
        (root / "merged.fits").write_text("old merge")
        (root / "merge.completed").write_text("complete\n")
    else:
        directory = root / ("training" if stage == "train" else stage)
        directory.mkdir(exist_ok=True)
        (directory / "status.json").write_text(json.dumps({"state": "completed"}))
    _finish(slurm, root, stage, "COMPLETED")


def test_rerun_redoes_the_evaluations_on_new_hardware_and_keeps_the_model(experiment):
    """Timing on other hardware needs only the two evaluations (and the merge
    that joins them) run again; the trained model and the old results stay."""
    slurm, config, base = experiment
    root = pipeline.submit(config, base)
    for stage in pipeline.STAGES:
        _complete(slurm, root, stage)
    before = len(slurm.submissions)
    assert pipeline.main(["resume", str(base), "--rerun", "shearnet", "--rerun", "ngmix",
                          "--gres", "shearnet=gpu:A100:1", "--cpus", "ngmix=128",
                          "--memory", "ngmix=480G"]) == 0
    calls = {args[-1]: args for args in slurm.submissions[before:]}
    assert set(calls) == {"shearnet", "ngmix", "merge"}      # no retraining
    assert "--gres=gpu:A100:1" in calls["shearnet"]
    assert "--cpus-per-task=128" in calls["ngmix"] and "--mem=480G" in calls["ngmix"]
    assert not any(a.startswith("--dependency") for a in calls["shearnet"])  # model kept
    jobs = json.loads((root / "pipeline.json").read_text())["jobs"]
    assert f"--dependency=afterok:{jobs['shearnet']}:{jobs['ngmix']}" in calls["merge"]
    for stage in ("shearnet", "ngmix"):
        kept = list(root.glob(f"{stage}.replaced-*"))
        assert len(kept) == 1 and (kept[0] / "status.json").is_file()
    assert (next(root.glob("merge.replaced-*")) / "merged.fits").read_text() == "old merge"
    assert not (root / "merged.fits").exists()
    assert (root / "training/status.json").is_file()


def test_rerun_refuses_a_queued_stage_and_training(experiment):
    slurm, config, base = experiment
    root = pipeline.submit(config, base)
    with pytest.raises(RuntimeError, match="scancel it first"):
        pipeline.resume(root, rerun=["ngmix"])
    with pytest.raises(SystemExit):
        pipeline.main(["resume", str(base), "--rerun", "train"])
