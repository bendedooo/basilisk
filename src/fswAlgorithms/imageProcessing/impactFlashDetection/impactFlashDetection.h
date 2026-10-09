/*
 ISC License

 Copyright (c) 2016, Autonomous Vehicle Systems Lab, University of Colorado at Boulder
*/

#ifndef IMPACT_FLASH_DETECTION_H
#define IMPACT_FLASH_DETECTION_H

#include <stdint.h>
#include <string>
#include <opencv2/core.hpp>
#include "architecture/messaging/messaging.h"
#include "architecture/msgPayloadDefC/CameraImageMsgPayload.h"
#include "architecture/msgPayloadDefC/OpNavCirclesMsgPayload.h"
#include "architecture/_GeneralModuleFiles/sys_model.h"
#include "architecture/utilities/bskLogging.h"

/*! Prototype detector for localized bright changes in sequential camera images.
 *  The largest connected bright-change region is reported as a center and
 *  equivalent radius using the existing OpNavCircles message format.
 */
class ImpactFlashDetection: public SysModel {
public:
    ImpactFlashDetection();
    ~ImpactFlashDetection();

    void Reset(uint64_t CurrentSimNanos);
    void UpdateState(uint64_t CurrentSimNanos);

public:
    ReadFunctor<CameraImageMsgPayload> imageInMsg; //!< Input camera image
    Message<OpNavCirclesMsgPayload> opnavCirclesOutMsg; //!< Detection output (center/radius)

    int32_t differenceThreshold; //!< [pixel intensity] Minimum positive change to count as bright
    int32_t minimumArea; //!< [pixels] Minimum connected-component area
    int32_t maximumArea; //!< [pixels] Maximum connected-component area
    int32_t blurSize; //!< [pixels] Odd Gaussian blur kernel size; 1 disables blur
    int32_t useFirstFrameAsReference; //!< [-] 1: fixed first-frame baseline; 0: previous-frame baseline
    double noiseSF; //!< [-] Scale factor for output uncertainty
    BSKLogger bskLogger;

private:
    cv::Mat referenceGray;
    int referenceInitialized;
};

#endif
