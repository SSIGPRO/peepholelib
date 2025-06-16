#import cuml
# cuml.accel.install()
from pathlib import Path
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import cupy as cp
#from cuml import TSNE as cuTSNE

def cifar100_fine_to_coarse_map():
    """
    Returns an array m of shape (100,), where m[fine_label] = coarse_label in {0..19}.
    Coarse label ordering is the canonical CIFAR-100 'coarse_labels' order:
      0 aquatic mammals, 1 fish, 2 flowers, 3 food containers, 4 fruit/vegetables,
      5 household electrical devices, 6 household furniture, 7 insects, 8 large carnivores,
      9 large man-made outdoor things, 10 large natural outdoor scenes, 11 large omnivores/herbivores,
      12 medium mammals, 13 non-insect invertebrates, 14 people, 15 reptiles,
      16 small mammals, 17 trees, 18 vehicles 1, 19 vehicles 2.
    """
    coarse_to_fine = {
        0:  [4, 30, 55, 72, 95],                 # aquatic mammals
        1:  [1, 32, 67, 73, 91],                 # fish
        2:  [54, 62, 70, 82, 92],                # flowers
        3:  [9, 10, 16, 28, 61],                 # food containers
        4:  [0, 51, 53, 57, 83],                 # fruit/vegetables
        5:  [22, 39, 40, 86, 87],                # household electrical devices
        6:  [5, 20, 25, 84, 94],                 # household furniture
        7:  [6, 7, 14, 18, 24],                  # insects
        8:  [3, 42, 43, 88, 97],                 # large carnivores
        9:  [12, 17, 37, 68, 76],                # large man-made outdoor things
        10: [23, 33, 49, 60, 71],                # large natural outdoor scenes
        11: [15, 19, 21, 31, 38],                # large omnivores/herbivores
        12: [34, 63, 64, 66, 75],                # medium mammals
        13: [26, 45, 77, 79, 99],                # non-insect invertebrates
        14: [2, 11, 35, 46, 98],                 # people
        15: [27, 29, 44, 78, 93],                # reptiles
        16: [36, 50, 65, 74, 80],                # small mammals
        17: [47, 52, 56, 59, 96],                # trees
        18: [8, 13, 48, 58, 90],                 # vehicles 1
        19: [41, 69, 81, 85, 89],                # vehicles 2
    }

    m = np.empty(100, dtype=np.int64)
    for coarse, fine_list in coarse_to_fine.items():
        for f in fine_list:
            m[f] = coarse
    return m

def cifar100_fine_to_super10_map():
    """
    Returns s of shape (100,), where s[fine_label] = super_label in {0..9}.

    Superclasses (10):
      0 Aquatic animals                : coarse {0 aquatic mammals, 1 fish}
      1 Plants & produce               : coarse {2 flowers, 4 fruit/vegetables, 17 trees}
      2 Household & containers         : coarse {3 food containers, 5 household electrical, 6 household furniture}
      3 Small non-mammal animals       : coarse {7 insects, 13 non-insect invertebrates, 15 reptiles}
      4 Outdoor (scenes/structures)    : coarse {9 large man-made outdoor things, 10 large natural outdoor scenes}
      5 Vehicles                       : coarse {18 vehicles 1, 19 vehicles 2}
      6 Large mammals                  : coarse {8 large carnivores, 11 large omnivores/herbivores}
      7 Medium mammals                 : coarse {12 medium mammals}
      8 Small mammals                  : coarse {16 small mammals}
      9 People                         : coarse {14 people}
    """
    fine_to_coarse = cifar100_fine_to_coarse_map()

    coarse_to_super = np.array([
        0,  # 0 aquatic mammals
        0,  # 1 fish
        1,  # 2 flowers
        1,  # 17 trees
        2,  # 3 food containers
        2,  # 5 household electrical devices
        2,  # 6 household furniture
        3,  # 7 insects
        3,  # 13 non-insect invertebrates
        3,  # 15 reptiles
        6,  # 8 large carnivores
        6,  # 11 large omnivores/herbivores
        4,  # 9 large man-made outdoor things
        4,  # 10 large natural outdoor scenes
        7,  # 12 medium mammals
        7,  # 16 small mammals
        8,  # 4 fruit/vegetables
        9,  # 14 people
        5,  # 18 vehicles 1
        5,  # 19 vehicles 2
    ], dtype=np.int64)

    return coarse_to_super[fine_to_coarse]



def plot_tsne(**kwargs):
    """
    Arguments (all via kwargs):
        corevector : already loaded
        layer : str            (only one layer)
        save_path : str|Path   (output directory)
        file_name : str        (output filename)
        cv_dim : int           (use a low one, otherwise doesnt work)

        Optional (labels coloring):
        ds : ParsedDataset 
        loader : ex 'CIFAR100-train'
        n_classes : int        (100, 20, or 10)

        Optional kwargs (TSNE parameters):
        n_components, perplexity, learning_rate, init,
        random_state, n_iter, verbose, etc.
    """
    corevector = kwargs.pop("corevector")
    cv_dim = kwargs.pop("cv_dim", 10)
    layer = kwargs.pop("layer")
    save_path = Path(kwargs.pop("save_path"))
    file_name = kwargs.pop("file_name", f"tsne_plot_{layer}.png")
    ds = kwargs.pop("ds", None)
    loader = kwargs.pop("loader", "CIFAR100-train")
    n_classes = int(kwargs.pop("n_classes", 100))

    y_np = None

    # Load X
    X = corevector._corevds[loader][layer]
    X_np = X[:, :cv_dim].cpu().numpy()

    # Load y if provided
    if ds is not None and loader is not None:
        y = ds._dss[loader][:]["label"]
        y_np = y.cpu().numpy()

        if len(y_np) != len(X_np):
            print(
                f"Warning: labels length ({len(y_np)}) "
                f"!= X_np length ({len(X_np)}). Ignoring labels."
            )
            y_np = None

    # Optional mapping: CIFAR-100 fine → coarse(20) or super10(10)
    if y_np is not None:
        y_int = y_np.astype(np.int64)

        if n_classes in (20, 10):
            if y_int.min() < 0 or y_int.max() > 99:
                print(
                    "Warning: labels are not in [0, 99]; cannot apply CIFAR-100 label mapping. "
                    "Proceeding with original labels."
                )
            else:
                if n_classes == 20:
                    fine_to_coarse = cifar100_fine_to_coarse_map()
                    y_np = fine_to_coarse[y_int]
                else:  # n_classes == 10
                    fine_to_super10 = cifar100_fine_to_super10_map()
                    y_np = fine_to_super10[y_int]

        elif n_classes != 100:
            print(
                f"Warning: n_classes={n_classes} not supported. "
                "Use 100, 20, or 10. Proceeding with original labels."
            )

    save_path.mkdir(parents=True, exist_ok=True)

    # t-SNE
    n_components = kwargs.get("n_components", 2)
    tsne = TSNE(**kwargs)
    X_tsne = tsne.fit_transform(X_np)

    # Plot
    if n_components == 2:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

        if y_np is not None:
            if n_classes in (20, 10):
                # Stable colormap with exactly n_classes colors
                colors = plt.cm.hsv(np.linspace(0, 1, n_classes, endpoint=False))
                cmap = ListedColormap(colors)

                y_idx = y_np.astype(np.int64)
                scatter = ax.scatter(
                    X_tsne[:, 0],
                    X_tsne[:, 1],
                    c=y_idx,
                    cmap=cmap,
                    s=5,
                    alpha=0.8,
                    vmin=0,
                    vmax=n_classes - 1,
                )

                cbar = plt.colorbar(scatter, ticks=np.arange(n_classes))
                cbar.set_label(f"Class index (0–{n_classes-1})")

            else:
                labels = np.unique(y_np)
                n_labels = len(labels)

                colors = plt.cm.hsv(np.linspace(0, 1, n_labels))
                cmap = ListedColormap(colors)

                label_to_idx = {label: i for i, label in enumerate(labels)}
                y_idx = np.array([label_to_idx[y] for y in y_np])

                scatter = ax.scatter(
                    X_tsne[:, 0],
                    X_tsne[:, 1],
                    c=y_idx,
                    cmap=cmap,
                    s=5,
                    alpha=0.8,
                )

                cbar = plt.colorbar(
                    scatter,
                    ticks=np.linspace(0, n_labels - 1, min(n_labels, 10))
                )
                cbar.set_label("Label index")

        else:
            ax.scatter(X_tsne[:, 0], X_tsne[:, 1], s=5, alpha=0.7)

        title_suffix = ""
        if n_classes == 20:
            title_suffix = " (CIFAR-100 coarse: 20)"
        elif n_classes == 10:
            title_suffix = " (CIFAR-100 super: 10)"
        elif n_classes == 100:
            title_suffix = " (CIFAR-100 fine: 100)"
        else:
            title_suffix = f" (n_classes={n_classes})"

        ax.set_title("t-SNE embedding" + title_suffix)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")

    elif n_components == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        if y_np is not None:
            labels = np.unique(y_np)
            n_labels = len(labels)

            colors = plt.cm.hsv(np.linspace(0, 1, n_labels))
            cmap = ListedColormap(colors)

            label_to_idx = {label: i for i, label in enumerate(labels)}
            y_idx = np.array([label_to_idx[y] for y in y_np])

            scatter = ax.scatter(
                X_tsne[:, 0],
                X_tsne[:, 1],
                X_tsne[:, 2],
                c=y_idx,
                cmap=cmap,
                s=3,
                alpha=0.8,
            )

            cbar = plt.colorbar(
                scatter,
                ticks=np.linspace(0, n_labels - 1, min(n_labels, 10))
            )
            cbar.set_label("Label index")

        else:
            ax.scatter(
                X_tsne[:, 0],
                X_tsne[:, 1],
                X_tsne[:, 2],
                s=3,
                alpha=0.7,
            )

        ax.set_title("t-SNE embedding (3D)")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_zlabel("t-SNE 3")

    else:
        raise ValueError("n_components must be 2 or 3 for visualization.")

    plt.tight_layout()
    print("Saving t-SNE plot to:", save_path / file_name)
    plt.savefig(save_path / file_name, dpi=300)
    plt.close()

    return X_tsne, save_path


def plot_tsne_CUDA(**kwargs):
    """
    Arguments (all via kwargs):
        corevector : already loaded
        layer : str            (only one layer)
        save_path : str|Path   (output directory)
        file_name : str        (output filename)

        Optional (labels coloring):
        ds : ParsedDataset 
        loader : ex 'CIFAR100-train'

        Optional:
        superclasses : bool    (if True, map CIFAR-100 fine labels (100) -> coarse labels (20))

        Optional kwargs (TSNE parameters):
        n_components, perplexity, learning_rate, init,
        random_state, n_iter, etc.
        check https://docs.rapids.ai/api/cuml/stable/api/#cuml.TSNE for more detail
    """
    corevector = kwargs.pop("corevector")
    layer = kwargs.pop("layer")
    save_path = Path(kwargs.pop("save_path"))
    file_name = kwargs.pop("file_name", "tsne_plot.png")
    ds = kwargs.pop("ds", None)
    loader = kwargs.pop("loader", "CIFAR100-train")
    n_classes = int(kwargs.pop("n_classes", 100))

    # T-SNE params
    perplexity = kwargs.pop("perplexity", 400)        # based on paper https://arxiv.org/pdf/2308.15513
    n_neighbors = kwargs.pop("n_neighbors", 1200)     # should be at least 3*perplexity
    method = kwargs.pop("method", "exact")            # slower but more accurate
    learning_rate = kwargs.pop("learning_rate", 500)  # has to be high for 40000 samples
    n_iter = kwargs.pop("n_iter", 2000)               # the more the better
    late_exaggeration = kwargs.pop("late_exaggeration", 1)
    init = kwargs.pop("init", "pca")
    random_state = kwargs.pop("random_state", 42)

    # Load X 
    y_np = None
    X = corevector._corevds[loader][layer]
    X_np = X.cpu().numpy()
    X_cp = cp.asarray(X_np)

    # Load y if provided 
    if ds is not None and loader is not None:
        y = ds._dss[loader][:]["label"]
        y_np = y.cpu().numpy()

        if len(y_np) != len(X_np):
            print(
                f"Warning: labels length ({len(y_np)}) "
                f"!= X_np length ({len(X_np)}). Ignoring labels."
            )
            y_np = None

    # Optional mapping: fine -> coarse(20) or fine -> super10(10)
    if y_np is not None:
        y_int = y_np.astype(np.int64)

        if n_classes in (20, 10):
            if y_int.min() < 0 or y_int.max() > 99:
                print(
                    "Warning: labels are not in [0, 99]; cannot apply CIFAR-100 label mapping. "
                    "Proceeding with original labels."
                )
            else:
                if n_classes == 20:
                    fine_to_coarse = cifar100_fine_to_coarse_map()
                    y_np = fine_to_coarse[y_int]  # {0..19}
                else:  # n_classes == 10
                    fine_to_super10 = cifar100_fine_to_super10_map()
                    y_np = fine_to_super10[y_int]  # {0..9}

        elif n_classes != 100:
            print(f"Warning: n_classes={n_classes} not supported. Use 100, 20, or 10. Proceeding with original labels.")

    save_path.mkdir(parents=True, exist_ok=True)

    # t-SNE 
    tsne = cuTSNE(
        perplexity=perplexity,
        n_neighbors=n_neighbors,
        method=method,
        learning_rate=learning_rate,
        n_iter=n_iter,
        late_exaggeration=late_exaggeration,
        init=init,
        random_state=random_state,
        **kwargs
    )
    X_tsne_cp = tsne.fit_transform(X_cp)
    X_tsne = cp.asnumpy(X_tsne_cp)

    # Plot
    plt.figure(figsize=(8, 8))

    if y_np is not None:
        if n_classes in (20, 10):
            # Enforce exactly N colors, stable across runs/splits
            colors = plt.cm.hsv(np.linspace(0, 1, n_classes, endpoint=False))
            cmap = ListedColormap(colors)

            y_idx = y_np.astype(np.int64)
            scatter = plt.scatter(
                X_tsne[:, 0],
                X_tsne[:, 1],
                c=y_idx,
                cmap=cmap,
                s=5,
                alpha=0.8,
                vmin=0,
                vmax=n_classes - 1,
            )

            cbar = plt.colorbar(scatter, ticks=np.arange(n_classes))
            cbar.set_label(f"Class index (0–{n_classes-1})")

        else:
            # Colors depend on the set of labels present (fine classes or arbitrary labels)
            labels = np.unique(y_np)
            n_labels = len(labels)

            colors = plt.cm.hsv(np.linspace(0, 1, n_labels))
            cmap = ListedColormap(colors)

            label_to_idx = {label: i for i, label in enumerate(labels)}
            y_idx = np.array([label_to_idx[y] for y in y_np])

            scatter = plt.scatter(
                X_tsne[:, 0],
                X_tsne[:, 1],
                c=y_idx,
                cmap=cmap,
                s=5,
                alpha=0.8,
            )

            cbar = plt.colorbar(scatter, ticks=np.linspace(0, n_labels - 1, min(n_labels, 10)))
            cbar.set_label("Label index")

    else:
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=5, alpha=0.7)

    title_suffix = ""
    if n_classes == 20:
        title_suffix = " (CIFAR-100 coarse: 20)"
    elif n_classes == 10:
        title_suffix = " (CIFAR-100 super: 10)"
    elif n_classes == 100:
        title_suffix = " (CIFAR-100 fine: 100)"
    else:
        title_suffix = f" (n_classes={n_classes})"

    plt.title("t-SNE embedding" + title_suffix)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()

    print("Saving t-SNE plot to:", save_path / file_name)
    plt.savefig(save_path / file_name, dpi=300)
    plt.close()

    return X_tsne, save_path

def plot_corevec3D(**kwargs):
    """
    Plots 3 raw dimensions of the corevectors (a 3D window).

    Arguments:
        corevector : already loaded
        layer : str            (only one layer)
        save_path : str|Path   (output directory)
        file_name : str        (output filename)

        Optional (labels coloring):
        ds : ParsedDataset
        loader : ex 'CIFAR100-train'
        n_classes : int        (100, 20, or 10; applies CIFAR-100 fine->coarse/super10 mapping if possible)

        Window selection:
        start_dim : int        (start dimension of the 3D window; default 0)
        cv_dim : int           (must be 3 for this 3D plot; default 3)
    """
    corevector = kwargs.pop("corevector")
    layer = kwargs.pop("layer")
    save_path = Path(kwargs.pop("save_path"))
    file_name = kwargs.pop("file_name", "corevec_3d.png")
    ds = kwargs.pop("ds", None)
    loader = kwargs.pop("loader", None)
    n_classes = int(kwargs.pop("n_classes", 100))

    start_dim = int(kwargs.pop("start_dim", 0))
    cv_dim = int(kwargs.pop("cv_dim", 3))

    if cv_dim != 3:
        raise ValueError(f"plot_corevec3D requires cv_dim=3, got cv_dim={cv_dim}")

    y_np = None

    if loader is None:
        raise ValueError("plot_corevec3D requires 'loader' (e.g., 'CIFAR100-train').")

    X = corevector._corevds[loader][layer]
    D = int(X.shape[1])

    end_dim = start_dim + cv_dim
    if start_dim < 0 or end_dim > D:
        raise ValueError(
            f"Invalid slice start_dim={start_dim}, end_dim={end_dim} for corevector dim D={D}."
        )

    X_np = X[:, start_dim:end_dim].cpu().numpy()

    # Load labels (if provided)
    if ds is not None and loader is not None:
        y = ds._dss[loader][:]["label"]
        y_np = y.cpu().numpy()

        if len(y_np) != len(X_np):
            print(
                f"Warning: labels length ({len(y_np)}) "
                f"!= X_np length ({len(X_np)}). Ignoring labels."
            )
            y_np = None

    # Optional mapping: CIFAR-100 fine -> coarse(20) or fine -> super10(10)
    if y_np is not None:
        y_int = y_np.astype(np.int64)

        if n_classes in (20, 10):
            if y_int.min() < 0 or y_int.max() > 99:
                print(
                    "Warning: labels are not in [0, 99]; cannot apply CIFAR-100 label mapping. "
                    "Proceeding with original labels."
                )
            else:
                if n_classes == 20:
                    fine_to_coarse = cifar100_fine_to_coarse_map()
                    y_np = fine_to_coarse[y_int]  # {0..19}
                else:  # n_classes == 10
                    fine_to_super10 = cifar100_fine_to_super10_map()
                    y_np = fine_to_super10[y_int]  # {0..9}

        elif n_classes != 100:
            print(
                f"Warning: n_classes={n_classes} not supported. "
                "Use 100, 20, or 10. Proceeding with original labels."
            )

    save_path.mkdir(parents=True, exist_ok=True)

    # Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    if y_np is not None:
        if n_classes in (20, 10):
            # Stable colormap with exactly n_classes colors
            colors = plt.cm.hsv(np.linspace(0, 1, n_classes, endpoint=False))
            cmap = ListedColormap(colors)

            y_idx = y_np.astype(np.int64)
            scatter = ax.scatter(
                X_np[:, 0],
                X_np[:, 1],
                X_np[:, 2],
                c=y_idx,
                cmap=cmap,
                s=3,
                alpha=0.8,
                vmin=0,
                vmax=n_classes - 1,
            )

            cbar = plt.colorbar(scatter, ticks=np.arange(n_classes))
            cbar.set_label(f"Class index (0–{n_classes-1})")

        else:
            labels = np.unique(y_np)
            n_labels = len(labels)

            colors = plt.cm.hsv(np.linspace(0, 1, n_labels))
            cmap = ListedColormap(colors)

            label_to_idx = {label: i for i, label in enumerate(labels)}
            y_idx = np.array([label_to_idx[y] for y in y_np])

            scatter = ax.scatter(
                X_np[:, 0],
                X_np[:, 1],
                X_np[:, 2],
                c=y_idx,
                cmap=cmap,
                s=3,
                alpha=0.8,
            )

            cbar = plt.colorbar(scatter, ticks=np.linspace(0, n_labels - 1, min(n_labels, 10)))
            cbar.set_label("Label index")

    else:
        ax.scatter(X_np[:, 0], X_np[:, 1], X_np[:, 2], s=3, alpha=0.7)

    title_suffix = f" (dims {start_dim}-{end_dim-1})"
    if n_classes == 20:
        title_suffix += " | CIFAR-100 coarse: 20"
    elif n_classes == 10:
        title_suffix += " | CIFAR-100 super: 10"
    elif n_classes == 100:
        title_suffix += " | CIFAR-100 fine: 100"
    else:
        title_suffix += f" | n_classes={n_classes}"

    ax.set_title("Corevectors – 3D window" + title_suffix)
    ax.set_xlabel(f"dim {start_dim}")
    ax.set_ylabel(f"dim {start_dim+1}")
    ax.set_zlabel(f"dim {start_dim+2}")

    plt.tight_layout()
    print("Saving 3D corevector plot to:", save_path / file_name)
    plt.savefig(save_path / file_name, dpi=300)
    plt.close()

    return X_np, save_path
