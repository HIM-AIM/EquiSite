import os
import argparse
import torch
from tqdm import tqdm
from torch_geometric.data import Data
import esm
import h5py
import numpy as np

# Dependencies
from dataset.utils.PyProtein import PyProtein
from dataset.utils.PyPeriodicTable import PyPeriodicTable
from model.equisite_t3_pro import EquiSite

# Global ESM Load
print("Loading ESM-2...")
esm_model, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
esm_batch_converter = esm_alphabet.get_batch_converter()
esm_model = esm_model.eval().cuda()


# ================= Helpers =================

def remove_hetatm(src, dst):
    with open(src, "r") as f:
        lines = [L for L in f.readlines() if not L.startswith("HETATM")]
    with open(dst, "w") as f:
        f.writelines(lines)


def pdb_to_hdf5(pdb_path, hdf5_path):
    prot = PyProtein(PyPeriodicTable())
    prot.load_molecular_file(pdb_path)
    prot.compute_covalent_bonds()
    prot.compute_hydrogen_bonds()
    prot.save_hdf5(hdf5_path)


def compute_esm(seq):
    _, _, toks = esm_batch_converter([("_", seq)])
    with torch.no_grad():
        res = esm_model(toks.cuda(), repr_layers=[33], return_contacts=False)
        rep = res["representations"][33].squeeze()
    return rep[1:-1, :].cpu()


def get_seq_from_hdf5(hdf5_path):
    with h5py.File(hdf5_path, "r") as h5:
        names = [x.decode("utf-8") for x in h5["atom_residue_names"][()]]
        ids = h5["atom_amino_id"][()]

    mapping = {
        'GLY': 'G', 'ALA': 'A', 'VAL': 'V', 'ILE': 'I', 'LEU': 'L', 'PHE': 'F', 'PRO': 'P', 'MET': 'M',
        'TRP': 'W', 'CYS': 'C', 'SER': 'S', 'THR': 'T', 'ASN': 'N', 'GLN': 'Q', 'TYR': 'Y', 'HIS': 'H',
        'ASP': 'D', 'GLU': 'E', 'LYS': 'K', 'ARG': 'R', 'UNK': 'X'
    }

    seq = []
    prev = None
    for n, i in zip(names, ids):
        if i != prev:
            seq.append(mapping.get(n, 'X'))
        prev = i
    return "".join(seq)


class DummyDB:
    from dataset.DNA_Check.PBdataset import DBdataset as _DB
    side_chain_embs = _DB.side_chain_embs
    bb_embs = _DB.bb_embs
    get_atom_pos = _DB.get_atom_pos
    compute_dihedrals = _DB.compute_dihedrals
    _normalize = _DB._normalize


def build_graph(hdf5_path, seq, db):
    h5 = h5py.File(hdf5_path, "r")
    data = Data()

    at = h5["amino_types"][()]
    at[at == -1] = 25

    ids = h5["atom_amino_id"][()]
    names = h5["atom_names"][()]
    pos = h5["atom_pos"][()][0]

    # ESM
    data.esm_emb = compute_esm(seq)

    # Geometry
    pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h = db.get_atom_pos(
        at, names, ids, pos
    )

    data.side_chain_embs = db.side_chain_embs(pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h)
    data.side_chain_embs[torch.isnan(data.side_chain_embs)] = 0

    bb = db.bb_embs(torch.cat((pos_n.unsqueeze(1), pos_ca.unsqueeze(1), pos_c.unsqueeze(1)), 1))
    bb[torch.isnan(bb)] = 0
    data.bb_embs = bb

    data.x = torch.tensor(at).unsqueeze(1)
    data.coords_ca = pos_ca
    data.coords_n = pos_n
    data.coords_c = pos_c

    h5.close()
    return data


# ================= Inference =================

def run_inference(model_path, pdb_dir, out_dir, device):
    os.makedirs(out_dir, exist_ok=True)
    temp_dir = os.path.join(out_dir, "temp_processed")
    os.makedirs(temp_dir, exist_ok=True)

    print(f"Loading Model: {model_path}")
    ckpt = torch.load(model_path, map_location=device)

    model = EquiSite(
        num_blocks=4,
        hidden_channels=128,
        out_channels=1,
        cutoff=11.5,
        dropout=0.25,
        level="allatom+esm",
        args=None
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    db = DummyDB()

    pdb_files = [x for x in os.listdir(pdb_dir) if x.endswith(".pdb")]
    print(f"Found {len(pdb_files)} PDBs.")

    for fn in tqdm(pdb_files):
        try:
            name = fn.replace(".pdb", "")
            raw_pdb = os.path.join(pdb_dir, fn)
            clean_pdb = os.path.join(temp_dir, fn)
            h5_path = os.path.join(temp_dir, name + ".hdf5")
            out_txt = os.path.join(out_dir, f"{name}.out")

            # Pipeline
            remove_hetatm(raw_pdb, clean_pdb)
            pdb_to_hdf5(clean_pdb, h5_path)
            seq = get_seq_from_hdf5(h5_path)
            data = build_graph(h5_path, seq, db).to(device)

            with torch.no_grad():
                pred, _, _ = model(data)

            # Save
            scores = pred.squeeze().cpu().tolist()
            if isinstance(scores, float): scores = [scores]

            with open(out_txt, "w") as f:
                for s in scores:
                    f.write(f"{s}\n")

        except Exception as e:
            print(f"Error processing {fn}: {e}")


# ================= Main =================

if __name__ == "__main__":
    # SET YOUR DEFAULT PATHS HERE
    DEFAULT_DNA_PATH = "model/checkpoints/DNA/best_val.pt"
    DEFAULT_RNA_PATH = "model/checkpoints/RNA/best_val.pt"

    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, choices=["DNA", "RNA"], default="DNA", help="Predict binding type")
    parser.add_argument("--model_path", type=str, default=None, help="Override default model path")
    parser.add_argument("--pdb_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="inference_results")
    parser.add_argument("--device", type=int, default=0)

    args = parser.parse_args()

    # Path Logic
    target_model = args.model_path
    if target_model is None:
        if args.type == "DNA":
            target_model = DEFAULT_DNA_PATH
        else:
            target_model = DEFAULT_RNA_PATH

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    run_inference(target_model, args.pdb_dir, args.out_dir, device)