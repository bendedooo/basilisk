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

/*! Detect localized positive brightness changes in sequential camera images
 *  using an adaptive background model and adaptive threshold.
 */
class ImpactFlashDetection: public SysModel {
public:
    ImpactFlashDetection();
    ~ImpactFlashDetection();

    void Reset(uint64_t CurrentSimNanos);
    void UpdateState(uint64_t CurrentSimNanos);

public:
    ReadFunctor<CameraImageMsgPayload> imageInMsg;
    Message<OpNavCirclesMsgPayload> opnavCirclesOutMsg;

    double backgroundAlpha;       //!< [-] Background update coefficient
    double thresholdSigma;        //!< [-] Standard-deviation multiplier
    int32_t warmupFrames;          //!< [frames] Frames used to initialize background
    int32_t minimumArea;            //!< [pixels] Minimum connected-component area
    int32_t maximumArea;            //!< [pixels] Maximum connected-component area
    int32_t blurSize;               //!< [pixels] Odd Gaussian blur kernel size
    double noiseSF;                 //!< [-] Output uncertainty scale factor
    double highThresholdRatio; //!< [-] High threshold multiplier
    double lowThresholdRatio; //!< [-] Low threshold multiplier




    BSKLogger bskLogger;

private:
    cv::Mat backgroundGray;         //!< Floating-point background image
    int32_t warmupCounter;          //!< Number of background initialization frames
};

#endif