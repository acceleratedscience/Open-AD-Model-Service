#
# MIT License
#
# Copyright (c) 2022 GT4SD team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
"""Module initialization for gt4sd traning pipelines."""

import json
import logging
from typing import Any, Dict

import sentencepiece as _sentencepiece
import torch as _torch

# import tensorflow as _tensorflow
from gt4sd_trainer.hf_pl.core import (  # noqa: F401
    LanguageModelingDataArguments,
    LanguageModelingModelArguments,
    LanguageModelingSavingArguments,
    LanguageModelingTrainingPipeline,
)
from gt4sd_trainer.hf_pl.pytorch_lightning_trainer import (  # noqa: F401
    PytorchLightningTrainingArguments,
)

from gt4sd_common.cli.load_arguments_from_dataclass import extract_fields_from_class
from gt4sd_common.tests.utils import exitclose_file_creator

"""from .cgcnn.core import (
    CGCNNDataArguments,
    CGCNNModelArguments,
    CGCNNSavingArguments,
    CGCNNTrainingArguments,
    CGCNNTrainingPipeline,
)"""
"""from .crystals_crf.core import (
    CrystalsRFCDataArguments,
    CrystalsRFCModelArguments,
    CrystalsRFCSavingArguments,
    CrystalsRFCTrainingArguments,
    CrystalsRFCTrainingPipeline,
)"""
"""from .diffusion.core import (
    DiffusionDataArguments,
    DiffusionForVisionTrainingPipeline,
    DiffusionModelArguments,
    DiffusionSavingArguments,
    DiffusionTrainingArguments,
)"""

"""
from .guacamol_baselines.core import GuacaMolDataArguments, GuacaMolSavingArguments
from .guacamol_baselines.smiles_lstm.core import (
    GuacaMolLSTMModelArguments,
    GuacaMolLSTMTrainingArguments,
    GuacaMolLSTMTrainingPipeline,
)
from .moses.core import MosesDataArguments, MosesSavingArguments
from .moses.organ.core import (
    MosesOrganModelArguments,
    MosesOrganTrainingArguments,
    MosesOrganTrainingPipeline,
)
from .moses.vae.core import (
    MosesVAEModelArguments,
    MosesVAETrainingArguments,
    MosesVAETrainingPipeline,
)
from .paccmann.core import (
    PaccMannDataArguments,
    PaccMannSavingArguments,
    PaccMannTrainingArguments,
)
from .paccmann.vae.core import PaccMannVAEModelArguments, PaccMannVAETrainingPipeline
from .pytorch_lightning.gflownet.core import (
    GFlowNetDataArguments,
    GFlowNetModelArguments,
    GFlowNetPytorchLightningTrainingArguments,
    GFlowNetSavingArguments,
    GFlowNetTrainingPipeline,
)
from .pytorch_lightning.granular.core import (
    GranularDataArguments,
    GranularModelArguments,
    GranularPytorchLightningTrainingArguments,
    GranularSavingArguments,
    GranularTrainingPipeline,
)
from .pytorch_lightning.molformer.core import (
    MolformerDataArguments,
    MolformerModelArguments,
    MolformerSavingArguments,
    MolformerTrainingArguments,
    MolformerTrainingPipeline,
)
"""
from .regression_transformer.core import (  # noqa: E402
    RegressionTransformerDataArguments,
    RegressionTransformerSavingArguments,
    RegressionTransformerTrainingArguments,
)
from .regression_transformer.implementation import (  # noqa: E402
    RegressionTransformerModelArguments,
    RegressionTransformerTrainingPipeline,
)

"""
from .torchdrug.core import (
    TorchDrugDataArguments,
    TorchDrugSavingArguments,
    TorchDrugTrainingArguments,
)
from .torchdrug.gcpn.core import (
    TorchDrugGCPNModelArguments,
    TorchDrugGCPNTrainingPipeline,
)
from .torchdrug.graphaf.core import (
    TorchDrugGraphAFModelArguments,
    TorchDrugGraphAFTrainingPipeline,
)"""

# imports that have to be loaded before lightning to avoid segfaults
_sentencepiece
# _tensorflow
_torch

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

TRAINING_PIPELINE_NAME_METADATA_MAPPING = {
    "mock_training_pipeline": "mock_training_pipeline.json",
    "Terminator training": "terminator_training.json",
}

TRAINING_PIPELINE_ARGUMENTS_MAPPING = {
    "regression-transformer-trainer": (
        RegressionTransformerTrainingArguments,
        RegressionTransformerDataArguments,
        RegressionTransformerModelArguments,
    ),
}

TRAINING_PIPELINE_MAPPING = {
    "regression-transformer-trainer": RegressionTransformerTrainingPipeline,
}

TRAINING_PIPELINE_ARGUMENTS_FOR_MODEL_SAVING = {
    "regression-transformer-trainer": RegressionTransformerSavingArguments,
}


def training_pipeline_name_to_metadata(name: str) -> Dict[str, Any]:
    """Retrieve training pipeline metadata from the name.

    Args:
        name: name of the pipeline.

    Returns:
        dictionary describing the parameters of the pipeline. If the pipeline is not found, no metadata (a.k.a., an empty dictionary is returned).
    """

    metadata: Dict[str, Any] = {"training_pipeline": name, "parameters": {}}
    if name in TRAINING_PIPELINE_NAME_METADATA_MAPPING:
        try:
            path = exitclose_file_creator(f"training_pipelines/{TRAINING_PIPELINE_NAME_METADATA_MAPPING[name]}")
            with open(path, "rt") as fp:
                metadata["parameters"] = json.load(fp)
        except Exception:
            logger.exception(
                f'training pipeline "{name}" metadata fetching failed, returning an empty metadata dictionary'
            )

    elif name in TRAINING_PIPELINE_ARGUMENTS_MAPPING:
        for training_argument_class in TRAINING_PIPELINE_ARGUMENTS_MAPPING[name]:
            field_types = extract_fields_from_class(training_argument_class)
            metadata["parameters"].update(field_types)

    else:
        logger.warning(f'training pipeline "{name}" metadata not found, returning an empty metadata dictionary')
    metadata["description"] = metadata["parameters"].pop("description", "A training pipeline.")
    return metadata
