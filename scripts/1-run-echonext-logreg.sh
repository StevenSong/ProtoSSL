#!/bin/bash

set -e

ECHONEXT_DATA=/opt/gpudata/ecg/echonext
RUN_DIR=/opt/gpudata/steven/ecg-prototype-transfer/runs

python _echonext_logreg.py --echonext-data $ECHONEXT_DATA --output-path $RUN_DIR/logreg-unweighted
python _eval_probs.py --echonext-data $ECHONEXT_DATA --probs-npy $RUN_DIR/logreg-unweighted/probs.npy --output-path $RUN_DIR/logreg-unweighted

python _echonext_logreg.py --echonext-data $ECHONEXT_DATA --balance-class-weight --output-path $RUN_DIR/logreg-weighted
python _eval_probs.py --echonext-data $ECHONEXT_DATA --probs-npy $RUN_DIR/logreg-weighted/probs.npy --output-path $RUN_DIR/logreg-weighted
