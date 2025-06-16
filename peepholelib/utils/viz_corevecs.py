import cuml
cuml.accel.install()
from pathlib import Path
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import cupy as cp
from cuml import TSNE as cuTSNE


def plot_tsne(**kwargs):
    """
    Arguments (all via kwargs):
        corevector : already loaded
        layer : str            (only one layer)
        save_path : str|Path   (output directory)
        file_name : str        (output filename)
        cv_dim : int (use a low one, otherwise doesnt work)

        Optional (labels coloring):
        ds : ParsedDataset 
        loader : ex 'CIFAR100-train'

        Optional kwargs (TSNE parameters):
        n_components, perplexity, learning_rate, init,
        random_state, n_iter, verbose, etc.
    """
    corevector = kwargs.get("corevector")
    cv_dim = kwargs.pop("cv_dim", 10)
    layer = kwargs.pop("layer")
    save_path = Path(kwargs.pop("save_path"))
    file_name = kwargs.pop("file_name", f"tsne_plot_{layer}.png")
    ds = kwargs.pop("ds", None)
    loader = kwargs.pop("loader", 'train')


    y_np = None
    X = corevector._corevds[loader][layer]
    X_np = X[:, :cv_dim].cpu().numpy()

    if ds is not None and loader is not None:
        y = ds._dss[loader][:]["label"]
        y_np = y.cpu().numpy()
        if len(y_np) != len(X_np):
            print(
                f"Warning: labels length ({len(y_np)}) "
                f"!= X_np length ({len(X_np)}). Ignoring labels."
            )
            y_np = None

    save_path.mkdir(parents=True, exist_ok=True)

    #  t-SNE 
    n_components = kwargs.get("n_components", 2)
    tsne = TSNE(**kwargs)
    X_tsne = tsne.fit_transform(X_np)

    #  plot 
    if n_components == 2:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

        if y_np is not None:
            labels = np.unique(y_np)
            n_labels = len(labels)

            # distinct colors per label
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
                alpha=0.8
            )
            cbar = plt.colorbar(scatter, ticks=np.linspace(0, n_labels-1, min(n_labels, 10)))
            cbar.set_label("Label index")
        else:
            ax.scatter(X_tsne[:, 0], X_tsne[:, 1], s=5, alpha=0.7)

        ax.set_title("t-SNE embedding (2D)")
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
                alpha=0.8
            )
            cbar = plt.colorbar(scatter, ticks=np.linspace(0, n_labels-1, min(n_labels, 10)))
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
    loader = kwargs.pop("loader", 'train')

    # T-SNE params
    perplexity = kwargs.pop("perplexity", 400) # based on paper https://arxiv.org/pdf/2308.15513
    n_neighbors = kwargs.pop("n_neighbors", 1200) # should be at least 3*perplexity
    method = kwargs.pop("method", "exact") # slower but more accurate
    learning_rate = kwargs.pop("learning_rate", 500) # has to be high for 40000 samples
    n_iter = kwargs.pop("n_iter", 2000) # the more the better
    late_exaggeration = kwargs.pop("late_exaggeration", 1) # for clearer clustering put more than 1
    init = kwargs.pop("init", "pca") # pcsa is more stable 
    random_state = kwargs.pop("random_state", 42)

    y_np = None
    X = corevector._corevds[loader][layer]
    X_np = X.cpu().numpy()
    X_cp = cp.asarray(X_np) 

    if ds is not None and loader is not None:
        y = ds._dss[loader][:]["label"]
        y_np = y.cpu().numpy()
        if len(y_np) != len(X_np):
            print(
                f"Warning: labels length ({len(y_np)}) "
                f"!= X_np length ({len(X_np)}). Ignoring labels."
            )
            y_np = None

    save_path.mkdir(parents=True, exist_ok=True)

    tsne = cuTSNE(perplexity = perplexity,
        n_neighbors = n_neighbors,
        method = method,
        learning_rate = learning_rate,
        n_iter = n_iter, 
        late_exaggeration = late_exaggeration,
        init = init,
        random_state = random_state,
        **kwargs)            
    X_tsne_cp = tsne.fit_transform(X_cp)   
    X_tsne = cp.asnumpy(X_tsne_cp) 

    # plot 
    plt.figure(figsize=(8, 8))

    if y_np is not None:
        labels = np.unique(y_np)
        n_labels = len(labels)

        # a diff color per each label
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

        cbar = plt.colorbar(scatter, ticks=np.linspace(0, n_labels-1, min(n_labels,10)))
        cbar.set_label("Label index")

    else:
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=5, alpha=0.7)

    plt.title("t-SNE embedding")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()

    print("Saving t-SNE plot to:", save_path / file_name)
    plt.savefig(save_path / file_name, dpi=300)
    plt.close()

    return X_tsne, save_path


def plot_corevec2D(**kwargs):
    """
    Plots the first raw 2 dimensions of the corevectors.
    Arguments:
        corevector : already loaded
        layer : str            (only one layer)
        save_path : str|Path   (output directory)
        file_name : str        (output filename)

        Optional (labels coloring):
        ds : ParsedDataset 
        loader : ex 'CIFAR100-train'
    """
    corevector = kwargs.pop("corevector")
    layer = kwargs.pop("layer")
    save_path = Path(kwargs.pop("save_path"))
    file_name = kwargs.pop("file_name", "corevec_2d.png")
    ds = kwargs.pop("ds", None)
    loader = kwargs.pop("loader", None)

    y_np = None
    X = corevector._corevds[loader][layer]   

    X_np = X[:, :2].cpu().numpy()

    if ds is not None and loader is not None:
        y = ds._dss[loader][:]["label"]
        y_np = y.cpu().numpy()

    save_path.mkdir(parents=True, exist_ok=True)


    # plot 
    plt.figure(figsize=(8, 8))

    if y_np is not None:
        labels = np.unique(y_np)
        n_labels = len(labels)

        # a diff color per each label
        colors = plt.cm.hsv(np.linspace(0, 1, n_labels))
        cmap = ListedColormap(colors)

        label_to_idx = {label: i for i, label in enumerate(labels)}
        y_idx = np.array([label_to_idx[y] for y in y_np])

        scatter = plt.scatter(
            X_np[:, 0],
            X_np[:, 1],
            c=y_idx,
            cmap=cmap,
            s=5,
            alpha=0.8,
        )

        cbar = plt.colorbar(scatter, ticks=np.linspace(0, n_labels-1, min(n_labels, 10)))
        cbar.set_label("Label index")

    else:
        plt.scatter(X_np[:, 0], X_np[:, 1], s=5, alpha=0.7)

    plt.title("Corevectors – first 2 dimensions")
    plt.xlabel("dim 0")
    plt.ylabel("dim 1")
    plt.tight_layout()

    print("Saving 2D corevector plot to:", save_path / file_name)
    plt.savefig(save_path / file_name, dpi=300)
    plt.close()

    return X_np, save_path