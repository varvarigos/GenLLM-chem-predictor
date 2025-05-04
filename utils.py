from torch_geometric.data import Data
import torch

descriptor_names = [
    "Molecular weight", "Lipophilicity (LogP)", "Number of hydrogen donors", "Number of hydrogen acceptors",
    "Heavy atom count", "Topological polar surface area (TPSA)", "Fraction of sp3-hybridized carbons",
    "Number of rings", "Balaban's J index", "Molar refractivity", "Bertz complexity index",
    "NH or OH count", "NO count", "Number of aliphatic rings", "Number of aromatic rings",
    "Number of saturated rings", "Number of heteroatoms", "Number of rotatable bonds",
    "Valence electron count", "Labute ASA"
]

# Map from one-hot atom indices to atom types
ATOM_TYPES = [
    'As', 'B', 'Bi', 'Br', 'C', 'Cl', 'Co', 'Cu', 'F', 'Ga', 'Ge', 'Hg', 'I', 'K', 'Li', 'Mg',
    'N', 'Na', 'O', 'P', 'Pt', 'Rh', 'Ru', 'S', 'Se', 'Si', 'Tb', 'Te', 'Tl', 'W', 'Zn'
]
atom_to_symbol = {i: atom for i, atom in enumerate(ATOM_TYPES)}

def context_metrics(batch):
    x = batch.x  # (num_nodes, num_atom_types)
    edge_index = batch.edge_index  # (2, num_edges)
    edge_attr = batch.edge_attr  # (num_edges, 3)
    batch_vec = batch.batch  # (num_nodes)

    num_graphs = int(batch_vec.max()) + 1

    # === Nodes per graph
    num_nodes = torch.bincount(batch_vec, minlength=num_graphs)

    # === Edges per graph
    src, dst = edge_index
    same_graph_edges = batch_vec[src] == batch_vec[dst]
    valid_src = src[same_graph_edges]
    edge_batch = batch_vec[valid_src]
    num_edges = torch.bincount(edge_batch, minlength=num_graphs)

    # === Average degree (2 * edges / nodes)
    avg_degree = (2 * num_edges.float()) / num_nodes.clamp(min=1)

    # === Atom type counts per graph
    atom_counts = torch.zeros((num_graphs, x.size(1)), device=x.device)
    atom_counts.index_add_(0, batch_vec, x)

    most_common_atom_idx = atom_counts.argmax(dim=1)
    most_common_atom = [atom_to_symbol.get(idx.item(), "Unknown") for idx in most_common_atom_idx]

    # === Bond type fractions per graph
    bond_counts = torch.zeros((num_graphs, 3), device=x.device)  # [num_graphs, 3]
    bond_counts.index_add_(0, edge_batch, edge_attr[same_graph_edges])

    bond_sums = bond_counts.sum(dim=1).clamp(min=1)  # avoid division by zero
    frac_single = (bond_counts[:, 0] / bond_sums).cpu().tolist()
    frac_double = (bond_counts[:, 1] / bond_sums).cpu().tolist()
    frac_triple = (bond_counts[:, 2] / bond_sums).cpu().tolist()

    # === Assemble
    results = []
    for i in range(num_graphs):
        results.append({
            "num_nodes": num_nodes[i].item(),
            "num_edges": num_edges[i].item(),
            "avg_degree": avg_degree[i].item(),
            "most_common_atom": most_common_atom[i],
            "frac_single": frac_single[i],
            "frac_double": frac_double[i],
            "frac_triple": frac_triple[i],
        })

    return results

def expand_dataset(dataset):
    new_data = []
    for data in dataset:
        for i, descriptor in enumerate(descriptor_names):
            d = Data(
                x=data.x,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr,
                y=torch.tensor([data.y[i]]),
                descriptor=descriptor
            )
            new_data.append(d)
    return new_data
