from gt4sd_common.properties import PropertyPredictorRegistry

import tempfile
import glob

from rdkit import Chem

import importlib_resources
import os

seed = "CCO"
target = "drd2"
molecule = "C12C=CC=NN1C(C#CC1=C(C)C=CC3C(NC4=CC(C(F)(F)F)=CC=C4)=NOC1=3)=CN=2"
protein = "KFLIYQMECSTMIFGL"
molecule = Chem.MolFromSmiles(molecule)
molecule_properties_old = [
    "esol",
    "lipinski",
    "sas",
    "plogp",
    "logp",
    "molecular_weight",
    "number_of_aromatic_rings",
    "number_of_atoms",
    "number_of_h_acceptors",
    "number_of_h_donors",
    "number_of_heterocycles",
    "number_of_large_rings",
    "number_of_rings",
    "number_of_rotatable_bonds",
    "qed",
    "tpsa",
    "bertz",
    "OrganTox",
]

test_files = []
with importlib_resources.as_file(
    importlib_resources.files("gt4sd_common") / "properties/tests/",
) as file_path:
    for i in glob.glob(str(file_path) + "/*.cif"):
        cif_file = open(i, "r", encoding="utf-8")
        contents = cif_file.read()
        test_files.append({"name": os.path.basename(i), "contents": contents})

    for i in glob.glob(str(file_path) + "/*.csv"):
        cif_file = open(i, "r", encoding="utf-8")
        contents = cif_file.read()
        test_files.append({"name": os.path.basename(i), "contents": contents})


crystal_props = [
    "formation_energy",
    "absolute_energy",
    "band_gap",
    "fermi_energy",
    "bulk_moduli",
    "shear_moduli",
    "poisson_ratio",
    "metal_semiconductor_classifier",
    "metal_nonmetal_classifier",
]

proteins = [
    "length",
    "protein_weight",
    # rule-based properties
    "boman_index",
    "charge_density",
    "charge",
    "aliphaticity",
    "hydrophobicity",
    "isoelectric_point",
    "aromaticity",
    "instability",
]


molecule_properties = [
    "absolute_energy",
    "aliphaticity",
    "aromaticity",
    "band_gap",
    "bertz",
    "boman_index",
    "bulk_moduli",
    "charge",
    "charge_density",
    # "clintox",
    "activity_against_target",
    "esol",
    "fermi_energy",
    "formation_energy",
    "hydrophobicity",
    "instability",
    "is_scaffold",
    "isoelectric_point",
    "length",
    "lipinski",
    "logp",
    "metal_nonmetal_classifier",
    "metal_semiconductor_classifier",
    "molecular_weight",
    # "molformer_classification",
    # "molformer_multitask_classification",
    # "molformer_regression",
    "number_of_aromatic_rings",
    "number_of_atoms",
    "number_of_h_acceptors",
    "number_of_h_donors",
    "number_of_heterocycles",
    "number_of_large_rings",
    "number_of_rings",
    "number_of_rotatable_bonds",
    "number_of_stereocenters",
    # "organtox",
    "penalized_logp",
    "plogp",
    "poisson_ratio",
    "protein_weight",
    "qed",
    "sas",
    "scscore",
    "shear_moduli",
    # "sider",
    "similarity_seed",
    "tpsa",
    #    "tox21",
    # # properties from models requiring authentification Not strategic for MVP
    # "molecule_one",
    # askcos
    # "docking",
    # "docking_tdc",
]
error_props = []
results = []
print("go")
for prop in molecule_properties:
    if prop != "docking":
        continue
    try:
        if prop == "docking":
            # name: str = Field(default="pyscreener")
            # receptor_pdb_file: str = Field(example="/tmp/2hbs.pdb", description="Path to receptor PDB file")
            # box_center: List[int] = Field(example=[15.190, 53.903, 16.917], description="Docking box center")
            # box_size: List[float] = Field(example=[20, 20, 20], description="Docking box size")

            parms = {
                "receptor_pdb_file": "./45ew.pdb",
                "box_center": [15.190, 53.903, 16.917],
                "box_size": [20, 20, 20],
            }
            prop_object = PropertyPredictorRegistry.get_property_predictor(
                name=prop, parameters=parms
            )
            results.append(prop + " = " + str(prop_object(protein)))
        elif prop not in crystal_props:
            parms = {"algorithm_version": "v0"}
            if prop == "activity_against_target":
                parms = {"target": target}
                molecule = "C1=CC(=CC(=C1)Br)CN"
            if prop == "similarity_seed":
                parms = {
                    "smiles": seed,
                }
                molecule = "CCC"
            prop_object = PropertyPredictorRegistry.get_property_predictor(
                name=prop, parameters=parms
            )
            if prop in proteins:
                results.append(
                    "protein property " + prop + " = " + str(prop_object(protein))
                )
            elif prop in crystal_props:
                results.append(
                    "crystal  property " + prop + " = " + str(prop_object(molecule))
                )
            else:
                results.append(
                    "molecules property " + prop + " = " + str(prop_object(molecule))
                )
        else:
            tmpdir = tempfile.TemporaryDirectory(prefix="./")

            for i in test_files:
                temp_file = open(tmpdir.name + "/" + i["name"], "w")
                temp_file.write(i["contents"])
                temp_file.close()

            parms = {"algorithm_version": "v0"}

            # data_module = importlib_resources.as_file(importlib_resources.files(tmpdir.name))
            from pathlib import Path

            if prop != "metal_nonmetal_classifier":
                data_module = Path(tmpdir.name)
            else:
                data_module = Path(tmpdir.name + "/crf_data.csv")

            # data_path = Path(data_dir, tmpdir.name + "/*")

            model = PropertyPredictorRegistry.get_property_predictor(
                name=prop, parameters=parms
            )

            out = model(input=data_module)  # type: ignore
            if prop == "metal_nonmetal_classifier":
                formulas = out["formulas"]
                predictions = out["predictions"]
                i = 0
                for formula in formulas:
                    results.append(
                        "crystal  property "
                        + prop
                        + "  :  "
                        + formula
                        + " = "
                        + str(predictions[i])
                    )
                    i += 1

            else:
                pred_dict = dict(zip(out["cif_ids"], out["predictions"]))  # type: ignore
                # for results in pred_dict:
                for key in pred_dict:
                    results.append(
                        "crystal  property "
                        + prop
                        + "  :  "
                        + key
                        + " = "
                        + str(pred_dict[key])
                    )
                tmpdir.cleanup()

    except Exception as e:
        error_props.append(prop)
        print(prop)
        print(e)

print("error properties:")
print(error_props)

print()
print("Here are the results")
results.sort()
for result in results:
    print(result)
"""
import importlib_resources
from gt4sd_common.properties.crystals import CRYSTALS_PROPERTY_PREDICTOR_FACTORY

# property_class, parameters_class = CRYSTALS_PROPERTY_PREDICTOR_FACTORY["shear_moduli"]
parms = {"algorithm_version": "v0"}
model = PropertyPredictorRegistry.get_property_predictor(name="shear_moduli", parameters=parms)
# model = property_class(parameters_class(algorithm_version="v0"))  # type: ignore
with importlib_resources.as_file(
    importlib_resources.files("gt4sd_common") / "properties/tests1/",
) as file_path:
    print(file_path)
    out = model(input=file_path)  # type: ignore
    print(out)
    pred_dict = dict(zip(out["cif_ids"], out["predictions"]))  # type: ignore
    print(pred_dict)
    prediction = pred_dict["1000041"]
    print(prediction)
"""
