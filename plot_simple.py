import sys
import csv
import os
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) < 2:
        print("Uso: python plot_simple.py <ruta/al/archivo.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    if not os.path.isfile(csv_path):
        print(f"Error: no existe el archivo: {csv_path}")
        sys.exit(1)

    xs, ys = [], []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter=",")
        if not {"episode", "rewards_epi"}.issubset(set(reader.fieldnames or [])):
            print("Error: el CSV debe tener columnas 'episode' y 'rewards_epi'.")
            print(f"Encabezados detectados: {reader.fieldnames}")
            sys.exit(1)
        for row in reader:
            try:
                xs.append(float(row["episode"]))
                ys.append(float(row["rewards_epi"]))
            except ValueError:
                # ignora filas no numéricas
                continue

    if not xs:
        print("Error: no se pudieron leer datos válidos.")
        sys.exit(1)

    plt.figure(figsize=(10, 5))
    plt.plot(xs, ys, linewidth=1.2)
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa por episodio")
    plt.title(os.path.basename(csv_path))
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

