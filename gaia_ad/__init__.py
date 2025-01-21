#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.
# flake8: noqa

from .supervised.predict_on_dataset import predict_on_dataset
from .plots.plot_anomaly_scores import plot_anomaly_scores
from .plots.plot_feature_importance import plot_feature_importance
from .plots.plot_tsne import plot_tsne

from .utils.set_log_level import set_log_level


GAIAAD_LOGLEVEL = "WARNING"
set_log_level(GAIAAD_LOGLEVEL)
