Control for `fourth_inloop_response`: identical in every respect except that
every response weight is 0.0.

Without it that run's m is a number with nothing to compare against -- you
cannot separate the response terms from the wider backbone, the in-loop fresh
noise, or anything else that changed at the same time.

`report: true` is kept ON. Measuring R^gamma and R^PSF on a model that was never
trained to control them is exactly the quantity the response arm has to beat.
