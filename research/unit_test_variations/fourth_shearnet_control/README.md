Control for the ShearNet arms (`fourth_inloop_shearnet` and
`fourth_shearnet_k4`): the same ShearNet, with every response weight set to 0.0
and nothing else changed. It is a valid control for either, because with all
weights at zero `every_n_steps` and `orbit_k` are inert.

Without it a flagship m is a number with nothing to compare against -- you
cannot separate the response terms from the wider backbone, from the in-loop
fresh noise, or from anything else that moved at the same time. The parsed
configs differ in exactly the five response weights, `orbit_k` (inert at zero
weight), the model name and the output root.

`report: true` is kept ON. Measuring R^gamma and R^PSF on a model that was
never trained to control them is exactly the quantity the response arm has to
beat.

    ./sub.sh fourth_shearnet_control
