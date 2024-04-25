from gt4sd_common.properties import PropertyPredictorRegistry
from rdkit import Chem

gentrl_ddr1_smi = "C12C=CC=NN1C(C#CC1=C(C)C=CC3C(NC4=CC(C(F)(F)F)=CC=C4)=NOC1=3)=CN=2"
gentrl_ddr1_mol = Chem.MolFromSmiles(gentrl_ddr1_smi)
molecule_properties = [
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
]
for i in molecule_properties:
    prop_object = PropertyPredictorRegistry.get_property_predictor(i)
    print(i + " = " + str(prop_object(gentrl_ddr1_mol)))
