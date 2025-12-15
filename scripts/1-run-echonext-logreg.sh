#!/bin/bash

set -e

ECHONEXT_DATA=/opt/gpudata/ecg/echonext
RUN_DIR=/opt/gpudata/steven/ecg-prototype-transfer/runs

python _echonext_logreg.py --echonext-data $ECHONEXT_DATA --target-type multitask --output-path $RUN_DIR/logreg-multitask-unweighted
python _echonext_logreg.py --echonext-data $ECHONEXT_DATA --target-type composite --output-path $RUN_DIR/logreg-composite-unweighted
python _echonext_logreg.py --echonext-data $ECHONEXT_DATA --target-type multitask --balance-class-weight --output-path $RUN_DIR/logreg-multitask-weighted
python _echonext_logreg.py --echonext-data $ECHONEXT_DATA --target-type composite --balance-class-weight --output-path $RUN_DIR/logreg-composite-weighted