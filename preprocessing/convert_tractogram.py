import argparse
import os
import numpy as np
from dipy.io.streamline import load_tractogram, save_tractogram

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Convertir des tractogrammes entre formats, en lot, avec option de flip d'orientation (LPS<->RAS)."
    )
    # Plusieurs fichiers d'entrée
    parser.add_argument(
        "input_files",
        nargs="+",
        help="Chemins des fichiers tractogramme d'entrée"
    )
    # Soit plusieurs -o (même nombre que les inputs), soit --output-format global
    parser.add_argument(
        "--output-file", "-o",
        action="append", nargs="+", default=[],
        help="Chemins de sortie. Répétez l'option ou fournissez plusieurs valeurs pour faire correspondre les fichiers d'entrée."
    )
    parser.add_argument(
        "--output-format", "-of",
        choices=["vtk", "trk", "tck"],
        help="Format de sortie global (remplace l'extension de chaque fichier d'entrée)."
    )
    # Références: une seule pour tous ou autant que d'inputs
    parser.add_argument(
        "--reference", "-r",
        action="append", default=[],
        help="Fichier(s) de référence. Un seul pour tous les inputs ou autant que d'inputs."
    )
    parser.add_argument(
        "--flip", "-f",
        action="store_true",
        help="Activer le flip d'orientation LPS<->RAS"
    )
    return parser.parse_args()

def _flatten(list_of_lists):
    flat = []
    for item in list_of_lists:
        if isinstance(item, (list, tuple)):
            flat.extend(item)
        else:
            flat.append(item)
    return flat

def _strip_all_ext(path):
    # retire .gz + extension précédente si présent (p.ex. .tck.gz -> base)
    base, ext = os.path.splitext(path)
    if ext == ".gz":
        base, _ = os.path.splitext(base)
    return base

def _is_ext(path, ext_no_dot):
    # vrai si path se termine par .ext ou .ext.gz
    return path.endswith(f".{ext_no_dot}") or path.endswith(f".{ext_no_dot}.gz")

def _derive_outputs(inputs, output_files_chunks, output_format):
    outputs = _flatten(output_files_chunks)
    if output_format and outputs:
        raise ValueError("Ne pas utiliser simultanément -o/--output-file et --output-format.")
    if output_format:
        ext = f".{output_format}"
        return [_strip_all_ext(p) + ext for p in inputs]
    if len(outputs) != len(inputs):
        raise ValueError(f"Le nombre de sorties (-o) doit égaler le nombre d'entrées ({len(inputs)}).")
    return outputs

def _derive_references(inputs, refs):
    if not refs:
        return [None] * len(inputs)
    if len(refs) == 1:
        return [refs[0]] * len(inputs)
    if len(refs) == len(inputs):
        return refs
    raise ValueError("Le nombre de références (-r) doit être 1 ou égal au nombre d'entrées.")

def flip_tractogram(tractogram):
    """
    Flip LPS<->RAS: inversion des x et y sur chaque streamline.
    """
    # Flip x and y coordinates (first and second columns)
    flipped = []
    for sl in tractogram.streamlines:
        sl2 = sl.copy()
        if sl2.shape[1] >= 2:
            sl2[:, 0] *= -1  # x
            sl2[:, 1] *= -1  # y
        flipped.append(sl2)
    tractogram.streamlines = flipped
    return tractogram

def main():
    args = parse_arguments()

    inputs = args.input_files
    try:
        outputs = _derive_outputs(inputs, args.output_file, args.output_format)
        refs = _derive_references(inputs, args.reference)
    except ValueError as e:
        print(f"Erreur: {e}")
        raise SystemExit(2)

    # Validation spécifique: .tck nécessite une référence
    for i, in_path in enumerate(inputs):
        if _is_ext(in_path, "tck") and refs[i] is None:
            print(f"Erreur: {in_path} est un .tck et nécessite une référence (-r).")
            raise SystemExit(2)

    for in_path, out_path, ref in zip(inputs, outputs, refs):
        print(f"Chargement: {in_path}")
        # Utiliser 'same' pour .trk si pas de référence fournie
        loader_ref = "same" if ref is None and _is_ext(in_path, "trk") else ref
        tractogram = load_tractogram(
            in_path,
            loader_ref,
            bbox_valid_check=False,
            trk_header_check=False
        )
        # Only flip if specified (not the default behavior anymore)
        if args.flip:
            print(f"Flip tractogram (LPS<->RAS)...")
            tractogram = flip_tractogram(tractogram)
        else:
            print("Conversion de format seule (pas de flip)...")

        print(f"Sauvegarde: {out_path}")
        save_tractogram(
            tractogram,
            out_path,
            bbox_valid_check=False
        )
        print("OK.")

    print("Terminé.")
if __name__ == "__main__":
    main()
