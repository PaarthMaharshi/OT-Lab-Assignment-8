"""Draw a map with a placement (saved to PNG)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"gun": "tab:blue", "cannon": "tab:red", "frost": "tab:cyan"}


def draw(gmap, placement, path_png, title=""):
    fig, ax = plt.subplots(figsize=(gmap.width * 0.5 + 1, gmap.height * 0.5 + 1))
    ax.set_xlim(-0.5, gmap.width - 0.5)
    ax.set_ylim(gmap.height - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(gmap.width))
    ax.set_yticks(range(gmap.height))
    ax.grid(True, alpha=0.3)
    px, py = zip(*gmap.path)
    ax.plot(px, py, "-", color="saddlebrown", lw=6, alpha=0.5, label="path")
    for (x, y), n in placement.items():
        ax.scatter([x], [y], s=180, color=COLORS.get(n, "k"), zorder=3)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path_png, dpi=120)
    plt.close(fig)
