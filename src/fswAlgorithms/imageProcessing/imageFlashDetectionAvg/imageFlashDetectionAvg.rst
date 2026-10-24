Image Flash Detection with Rolling Average
==========================================

Overview
--------

``ImageFlashDetectionAvg`` detects localized positive brightness changes in
camera images by comparing each valid grayscale image with the pixelwise
average of the previous ``numFramesToAverage`` frames.

The current image is excluded from its own reference average. The first N
valid frames fill the history buffer and do not produce detections. When image
dimensions change, the history is cleared and the warm-up period begins again.

Input Message
-------------

- ``imageInMsg``: ``CameraImageMsgPayload`` camera image.

Output Message
--------------

- ``opnavCirclesOutMsg``: ``OpNavCirclesMsgPayload`` containing the center and
  equivalent radius of the largest connected region satisfying the configured
  area limits. ``valid`` is set to 1 when such a region is found.

Configuration
-------------

- ``numFramesToAverage``: number of previous frames in the rolling baseline
  (default 5).
- ``differenceThreshold``: minimum positive grayscale difference (default 25).
- ``minimumArea`` / ``maximumArea``: connected-component area limits in pixels
  (defaults 2 and 5000).
- ``blurSize``: optional Gaussian blur kernel size; 1 disables blur (default 1).
- ``noiseSF``: uncertainty scale factor copied to the output (default 2.0).

Notes
-----

This is an initial temporal-change detector, not a complete flash classifier.
Large illumination changes, camera motion, or a flash persisting across many
frames can still affect detection performance. Those cases should be evaluated
with representative camera sequences before tuning thresholds.
