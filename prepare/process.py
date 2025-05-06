import os
import torch
from torch_geometric.data import Data, InMemoryDataset
from lxml import etree
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors as rdmd
from random import randint

def extract_atom_symbols(gxl_dir):
    atom_symbols = set()
    
    for filename in os.listdir(gxl_dir):
        if not filename.endswith(".gxl"):
            continue

        path = os.path.join(gxl_dir, filename)
        try:
            tree = etree.parse(path)
            graph = tree.find('.//{*}graph')

            for node in graph.findall('{*}node'):
                symbol_elem = node.find(".//{*}attr[@name='symbol']/{*}string")
                if symbol_elem is not None:
                    symbol = symbol_elem.text.strip()
                    atom_symbols.add(symbol)
        except Exception as e:
            print(f"Error parsing {filename}: {e}")

    return sorted(atom_symbols)


# === Unique atom types found ===
ATOM_TYPES = [
    'As', 'B', 'Bi', 'Br', 'C', 'Cl', 'Co', 'Cu', 'F', 'Ga', 'Ge', 'Hg', 'I', 'K', 'Li', 'Mg',
    'N', 'Na', 'O', 'P', 'Pt', 'Rh', 'Ru', 'S', 'Se', 'Si', 'Tb', 'Te', 'Tl', 'W', 'Zn'
]
ATOM_TO_IDX = {atom: i for i, atom in enumerate(ATOM_TYPES)}
IDX_TO_ATOM = {i: atom for atom, i in ATOM_TO_IDX.items()}

# === Bond types mapped to one-hot and RDKit bond types ===
BOND_TYPES = {
    1: ([1, 0, 0], Chem.BondType.SINGLE),
    2: ([0, 1, 0], Chem.BondType.DOUBLE),
    3: ([0, 0, 1], Chem.BondType.TRIPLE)
}

# === Parse a single GXL file into a PyG Data object ===
def parse_gxl_file(filepath):
    with open(filepath, 'rb') as f:
        tree = etree.parse(f)
    root = tree.getroot()
    graph = root.find('.//{*}graph')

    node_ids = []
    node_symbols = []
    for node in graph.findall('{*}node'):
        node_id = node.attrib['id']
        symbol_elem = node.find(".//{*}attr[@name='symbol']/{*}string")
        symbol = symbol_elem.text.strip() if symbol_elem is not None else 'C'
        node_ids.append(node_id)
        node_symbols.append(symbol)

    id_to_index = {node_id: i for i, node_id in enumerate(node_ids)}

    node_features = []
    for symbol in node_symbols:
        vec = torch.zeros(len(ATOM_TYPES))
        if symbol in ATOM_TO_IDX:
            vec[ATOM_TO_IDX[symbol]] = 1.0
        node_features.append(vec)
    x = torch.stack(node_features)

    edge_list = []
    edge_attrs = []
    rdkit_bonds = []
    for edge in graph.findall('{*}edge'):
        src_id = edge.attrib['from']
        dst_id = edge.attrib['to']
        src = id_to_index[src_id]
        dst = id_to_index[dst_id]

        valence_elem = edge.find(".//{*}attr[@name='valence']/{*}int")
        bond_type_id = int(valence_elem.text.strip()) if valence_elem is not None else 1
        bond_vec, bond_type = BOND_TYPES.get(bond_type_id, ([1, 0, 0], Chem.BondType.SINGLE))

        edge_list.append([src, dst])
        edge_list.append([dst, src])

        edge_attrs.append(torch.tensor(bond_vec, dtype=torch.float))
        edge_attrs.append(torch.tensor(bond_vec, dtype=torch.float))

        rdkit_bonds.append((src, dst, bond_type))

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(edge_attrs)

    # === Build RDKit molecule ===
    mol = Chem.RWMol()
    atom_idx_map = {}
    for i, symbol in enumerate(node_symbols):
        atom = Chem.Atom(symbol)
        atom_idx_map[i] = mol.AddAtom(atom)

    for src, dst, bond_type in rdkit_bonds:
        try:
            mol.AddBond(atom_idx_map[src], atom_idx_map[dst], bond_type)
        except Exception:
            continue

    y = None
    try:
        Chem.SanitizeMol(mol)
        mol_final = mol.GetMol()
        smile = Chem.MolToSmiles(mol_final)
        mol_rd = Chem.MolFromSmiles(smile)
        if mol_rd is not None:
            y = torch.tensor([
                Descriptors.MolWt(mol_rd),
                Descriptors.MolLogP(mol_rd),
                Descriptors.NumHDonors(mol_rd),
                Descriptors.NumHAcceptors(mol_rd),
                Descriptors.HeavyAtomCount(mol_rd),
                Descriptors.TPSA(mol_rd),
                Descriptors.FractionCSP3(mol_rd),
                Descriptors.RingCount(mol_rd),
                Descriptors.BalabanJ(mol_rd),
                Descriptors.MolMR(mol_rd),
                Descriptors.BertzCT(mol_rd),
                Descriptors.NHOHCount(mol_rd),
                Descriptors.NOCount(mol_rd),
                Descriptors.NumAliphaticRings(mol_rd),
                Descriptors.NumAromaticRings(mol_rd),
                Descriptors.NumSaturatedRings(mol_rd),
                Descriptors.NumHeteroatoms(mol_rd),
                Descriptors.NumRotatableBonds(mol_rd),
                Descriptors.NumValenceElectrons(mol_rd),
                Descriptors.LabuteASA(mol_rd),
                Crippen.MolMR(mol_rd),
                Crippen.MolLogP(mol_rd),
                Lipinski.FractionCSP3(mol_rd),
                rdmd.CalcNumBridgeheadAtoms(mol_rd),
                rdmd.CalcNumSpiroAtoms(mol_rd),
                rdmd.CalcNumAmideBonds(mol_rd),
                rdmd.CalcNumAtomStereoCenters(mol_rd),
                rdmd.CalcNumUnspecifiedAtomStereoCenters(mol_rd),
                rdmd.CalcChi0n(mol_rd),
                rdmd.CalcChi1n(mol_rd)
            ], dtype=torch.float)
    except Exception as e:
        print(f"Sanitization/descriptor error in {filepath}: {e}")

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

# === Custom dataset class ===
class AIDS_GraphDataset(InMemoryDataset):
    def __init__(self, root_dir, transform=None, pre_transform=None):
        self.root_dir = root_dir
        super().__init__(root_dir, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list = []
        for fname in os.listdir(self.root_dir):
            if fname.endswith('.gxl'):
                path = os.path.join(self.root_dir, fname)
                try:
                    data = parse_gxl_file(path)
                    if data.y is not None:
                        data_list.append(data)
                except Exception as e:
                    print(f"Failed to parse {fname}: {e}")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

