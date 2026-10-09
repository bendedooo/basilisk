%module impactFlashDetection
%{
#include "impactFlashDetection.h"
%}

%pythoncode %{
from Basilisk.architecture.swig_common_model import *
%}

%include "stdint.i"
%include "std_string.i"
%include "sys_model.i"
%include "swig_conly_data.i"

%include "impactFlashDetection.h"
%include "architecture/msgPayloadDefC/CameraImageMsgPayload.h"
struct CameraImageMsg_C;
%include "architecture/msgPayloadDefC/OpNavCirclesMsgPayload.h"
struct OpNavCirclesMsg_C;

%pythoncode %{
import sys
protectAllClasses(sys.modules[__name__])
%}
