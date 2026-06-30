"""ShearNet plotting subpackage.

Promoted from the former single ``shearnet/plotting.py`` module; split into
cohesive submodules (curves, scatter, animation, psf). All public names are
re-exported here, so ``shearnet.plotting.<name>`` keeps working.
"""

from .curves import (
    plot_learning_curve,
)
from .scatter import (
    plot_residuals,
    visualize_galaxy_samples,
    visualize_psf_samples,
    plot_true_vs_predicted,
)
from .animation import (
    plot_true_vs_predicted_anim,
    animate_model_epochs,
)
from .psf import (
    extract_psf_properties_from_obs,
    calculate_image_moments,
    plot_psf_systematics,
    plot_psf_systematics_from_eval,
    calculate_psf_leakage_coefficients,
)

__all__ = [
    "plot_learning_curve",
    "plot_residuals",
    "visualize_galaxy_samples",
    "visualize_psf_samples",
    "plot_true_vs_predicted",
    "plot_true_vs_predicted_anim",
    "animate_model_epochs",
    "extract_psf_properties_from_obs",
    "calculate_image_moments",
    "plot_psf_systematics",
    "plot_psf_systematics_from_eval",
    "calculate_psf_leakage_coefficients",
]
