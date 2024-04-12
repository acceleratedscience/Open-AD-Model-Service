from gt4sd_inference_regression.algorithms.conditional_generation.regression_transformer import (
    RegressionTransformer,
    RegressionTransformerMolecules,
)

from selfies import encoder


import logging
import sys
from gt4sd_common.algorithms.registry import ApplicationsRegistry

import mols2grid
from rdkit import Chem

import pandas as pd

logging.disable(sys.maxsize)

smi = "CC(C#C)N(C)C(=O)NC1=CC=C(Cl)C=C1"
true_esol = -3.9
Chem.MolFromSmiles(smi)


config = RegressionTransformerMolecules(algorithm_version="solubility", search="greedy")
target = f"<esol>[MASK][MASK][MASK][MASK][MASK]|{encoder(smi)}"
esol_predictor = RegressionTransformer(configuration=config, target=target)
score_str = list(esol_predictor.sample(1))[0]
print(f"For Buturuon, the predicted ESOL is {score_str.split('>')[-1]}")
target_esol = -3.53
config = RegressionTransformerMolecules(
    algorithm_version="solubility",
    search="sample",  # the alternative is 'greedy' but 'sample' is recommended for generative tasks
    temperature=2,
    tolerance=5,  # percentage of tolerated deviation from the desired property value (here -3.53)
    sampling_wrapper={
        "property_goal": {"<esol>": target_esol},
        "fraction_to_mask": 0.2,
    },
)
esol_generator = RegressionTransformer(configuration=config, target=smi)
generations = list(esol_generator.sample(8))
print("-------------------------------------------------------------------------------")
print(generations)

smiles = [g[0] for g in generations]
esols = [float(g[1].split(">")[-1]) for g in generations]
print("\033[1m" "\t\t\tButuruon-inspired molecules with a higher solubility score ")
result = pd.DataFrame(
    {
        "SMILES": [smi] + smiles,
        "Name": ["Buturon"] + len(smiles) * ["Novel (from RT)"],
        "ESOL": [true_esol] + [round(e, 3) for e in esols],
    }
)

print("-------------------------------------------------------------------------------")
mols2grid.display(
    result,
    tooltip=["Name", "SMILES", "ESOL"],
    size=(300, 150),
    fixedBondLength=25,
    n_cols=3,
    width="100%",
    height=None,
    name="Results",
)
arget_esol = -3.53
config = RegressionTransformerMolecules(
    algorithm_version="solubility",
    search="sample",  # the alternative is 'greedy' but 'sample' is recommended for generative tasks
    temperature=2,
    tolerance=5,  # percentage of tolerated deviation from the desired property value (here -3.53)
    sampling_wrapper={
        "property_goal": {"<esol>": target_esol},
        "fraction_to_mask": 1.0,
        "tokens_to_mask": ["Cl", "N"],
    },
)
esol_generator = RegressionTransformer(configuration=config, target=smi)
generations = list(esol_generator.sample(8))
generations

smiles = [g[0] for g in generations]
esols = [float(g[1].split(">")[-1]) for g in generations]
print("\033[1m" "\t\t\tButuruon-inspired molecules with a higher solubility score ")
result = pd.DataFrame(
    {
        "SMILES": [smi] + smiles,
        "Name": ["Buturon"] + len(smiles) * ["Novel (from RT)"],
        "ESOL": [true_esol] + [round(e, 3) for e in esols],
    }
)
mols2grid.display(
    result,
    tooltip=["Name", "SMILES", "ESOL"],
    size=(300, 150),
    fixedBondLength=25,
    n_cols=3,
    width="100%",
    height=None,
    name="Results",
)

config = RegressionTransformerMolecules(
    algorithm_version="solubility", search="sample", temperature=2, tolerance=5
)
target = "<esol>-3.53|[C][C][Branch1_3][Ring1][C][#C][N][Branch1_3][epsilon][C][C][Branch1_3][epsilon][MASK][MASK][MASK][MASK][C][=C][Branch1_3][epsilon][Cl][C][=C][Ring1][Branch1_2]"
esol_generator = RegressionTransformer(configuration=config, target=target)
generations = list(esol_generator.sample(5))
generations
smiles = [g[0] for g in generations]
esols = [float(g[1].split(">")[-1]) for g in generations]
print("\033[1m" "\t\t\tButuruon-inspired molecules with a higher solubility score ")
result = pd.DataFrame(
    {
        "SMILES": [smi] + smiles,
        "Name": ["Buturon"] + 5 * ["Novel (from RT)"],
        "ESOL": [true_esol] + [round(e, 3) for e in esols],
    }
)
mols2grid.display(
    result,
    tooltip=["Name", "SMILES", "ESOL"],
    size=(300, 150),
    fixedBondLength=25,
    n_cols=3,
    width="100%",
    height=None,
    name="Results",
)


algorithms = ApplicationsRegistry.list_available()
for a in algorithms:
    if a["algorithm_name"] == "RegressionTransformer":
        print(a["algorithm_application"], a["algorithm_version"])
