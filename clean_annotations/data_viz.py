import numpy as np
import os

def list_npy_files(directory="."):
    """Liste tous les fichiers .npy dans un dossier."""
    return [f for f in os.listdir(directory) if f.endswith(".npy")]

def load_npy_file(filepath):
    """Charge et affiche le contenu d'un fichier .npy."""
    try:
        data = np.load(filepath, allow_pickle=True)
        print(f"\n📂 Fichier: {filepath}")
        print(f"🔹 Type de données : {type(data)}")
        print(f"📏 Shape: {data.shape if hasattr(data, 'shape') else 'N/A'}\n")
        print("📝 Contenu:")
        print(data)
    except Exception as e:
        print(f"❌ Erreur lors du chargement de {filepath}: {e}")

if __name__ == "__main__":
    folder = input("📁 Entrez le chemin du dossier contenant les .npy (laisser vide pour le dossier actuel) : ") or "."
    npy_files = list_npy_files(folder)

    if not npy_files:
        print("⚠️ Aucun fichier .npy trouvé.")
    else:
        print("\n📋 Fichiers trouvés :")
        for i, file in enumerate(npy_files):
            print(f"{i+1}. {file}")

        choice = input("\nEntrez le numéro du fichier à afficher (laisser vide pour tous) : ")
        if choice.isdigit() and 1 <= int(choice) <= len(npy_files):
            load_npy_file(os.path.join(folder, npy_files[int(choice) - 1]))
        else:
            print("\n🔍 Chargement de tous les fichiers .npy...\n")
            for file in npy_files:
                load_npy_file(os.path.join(folder, file))
                print("-" * 50)  # Séparateur entre les fichiers
