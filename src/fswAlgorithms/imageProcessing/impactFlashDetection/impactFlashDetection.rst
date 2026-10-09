.. _ImpactFlashDetection:

Impact Flash Detection
======================

Overview
--------

This prototype detects a localized positive brightness change in sequential
camera images. It converts each image to grayscale, optionally applies a small
Gaussian blur, subtracts a reference frame, thresholds the positive difference,
and selects the largest connected component inside configured area limits.
The component centroid and equivalent-area radius are published through the
existing :ref:`OpNavCirclesMsgPayload` message for an initial integration test.

Parameters
----------

- `differenceThreshold`: minimum grayscale increase for a pixel to be selected.
- `minimumArea`, `maximumArea`: accepted connected-component area bounds.
- `blurSize`: Gaussian blur kernel size; values of 0 or 1 disable blur.
- `useFirstFrameAsReference`: when 1, keep the first frame as the baseline to
  detect a sustained bright patch; when 0, update the baseline each frame to
  emphasize frame-to-frame changes.
- `noiseSF`: scalar used for the reported diagonal uncertainty entries.

Limitations
-----------

The detector is a first prototype, not a flight-qualified optical-navigation
algorithm. It does not register images before differencing, so spacecraft or
Moon image motion can create false detections. It currently reports only the
largest accepted component and uses the existing circles output message as a
temporary carrier for centroid and equivalent radius. A dedicated detection
message and unit tests should follow after the first simulation integration.
