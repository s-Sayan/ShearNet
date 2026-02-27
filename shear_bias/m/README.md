In order to switch between including shearnet and not:

- Go to main.py#L100 and uncomment the STATE init
- Go to run_mcal.sh, change gres=gpu:0 to gres=gpu:1
- Don't forget to go into config and change the fits output name so you don't override previous outputs

To quickly get started:

- make sure to change the conda initialization and conda env in run_mcal.sh
- change filepaths in config.yaml
- run sbatch run_mcal!
