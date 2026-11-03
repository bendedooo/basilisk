.. _ImpactFlashDetection:

Impact Flash Detection
======================

Overview
--------

This module detects localized positive brightness changes in sequential
camera images using an adaptive background model and adaptive threshold.

The input image is converted to grayscale and optionally blurred.
A floating-point background image is updated recursively:

.. math::

    B_{t+1} = \alpha I_t + (1-\alpha)B_t

The positive difference image is calculated as:

.. math::

    D_t = \max(I_t-B_t,0)

The adaptive threshold is calculated using the mean of the column means
and the maximum column standard deviation:

.. math::

    T_t = \mu_{\mathrm{img}} + k\sigma_{\mathrm{img}}

Pixels above the threshold are grouped using connected-component analysis.
Accepted candidates are reported through the existing
:ref:`OpNavCirclesMsgPayload` message.

The background is initialized from zero and updated during a configurable
warm-up period. Following detection, background updates are temporarily
frozen to avoid absorbing a short-lived bright event.

Parameters
----------

- `backgroundAlpha`: background update coefficient.
- `thresholdSigma`: multiplier applied to the image standard deviation.
- `warmupFrames`: number of frames used to initialize the background.
- `freezeDurationNanos`: background freeze duration after candidate detection.
- `minimumArea`, `maximumArea`: accepted connected-component area bounds.
- `blurSize`: Gaussian blur kernel size; values of 0 or 1 disable blur.
- `noiseSF`: scalar used for the reported diagonal uncertainty entries.

Limitations
-----------

The detector does not register images before differencing. Spacecraft or
lunar image motion may therefore create false detections.

The adaptive threshold is calculated over the full image. A lunar-region
mask or region-specific threshold may be considered as a future extension.

The existing OpNavCircles message is used as a temporary carrier for
candidate centers and equivalent radii. Its uncertainty matrix represents
only the main candidate.