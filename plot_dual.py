import sys
import csv
import os
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) < 2:
        print("Uso: python plot_dual.py <ruta/al/archivo.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    if not os.path.isfile(csv_path):
        print(f"Error: no existe el archivo: {csv_path}")
        sys.exit(1)

    xs, rewards, eps = [], [], []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter=",")
        headers = set(reader.fieldnames or [])
        need = {"episode", "rewards_epi", "epsilon"}
        if not need.issubset(headers):
            print("Error: el CSV debe tener columnas 'episode', 'rewards_epi' y 'epsilon'.")
            print(f"Encabezados detectados: {reader.fieldnames}")
            sys.exit(1)

        for row in reader:
            try:
                xs.append(float(row["episode"]))
                rewards.append(float(row["rewards_epi"]))
                eps.append(float(row["epsilon"]))
            except ValueError:
                # ignora filas no numéricas
                continue

    if not xs:
        print("Error: no se pudieron leer datos válidos.")
        sys.exit(1)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Curva de recompensas (eje izquierdo)
    line1, = ax1.plot(xs, rewards, linewidth=1.2, label="rewards_epi")
    ax1.set_xlabel("Episodio")
    ax1.set_ylabel("Recompensa por episodio")
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Eje derecho para epsilon
    ax2 = ax1.twinx()
    line2, = ax2.plot(xs, eps, linestyle="--", linewidth=1.2, label="epsilon")
    ax2.set_ylabel("Epsilon")

    # Leyenda combinada (ambas series)
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best")

    plt.title(os.path.basename(csv_path))
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

