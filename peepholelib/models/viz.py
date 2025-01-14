from pathlib import Path
from matplotlib import pyplot as plt

def viz_singular_values(wrap, dir_path):
    
    # create folder / check if folder already exist
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)

    for key, item in wrap._svds.items():

        img_name = Path(key + '.png')

        fig = plt.figure(figsize=(8, 5))
        plt.plot(item['s'])

        plt.title(key)
        plt.xlabel('Index')
        plt.ylabel('Singular Value')
        plt.grid(True, linestyle='--', alpha=0.5)

        #plt.show()
        plt.savefig(fname=dir_path/img_name)
        plt.close(fig)


def viz_compare(wrap, dir_path):        #TODO log scale

    # create folder / check if folder already exist
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)

    layers = [f"encoder_layer_{i}" for i in range(0, 12)]

    for _l in layers:
        #tensor_to_plot = [wrap._svds[] for _k in ]
        k1 = 'encoder.layers.' + _l + '.self_attention.out_proj'
        k2 = 'encoder.layers.' + _l + '.mlp.0'
        k3 = 'encoder.layers.' + _l + '.mlp.3'

        fig, ax = plt.subplots(1, 3, figsize=(20,5), sharey=True)

        plt.title(_l)

        fig.suptitle(_l)
        ax[0].plot(wrap._svds[k1]['s'])
        ax[0].set_yscale('log')
        ax[0].set_title('.out_proj')
        ax[0].grid(True, linestyle='--', alpha=0.5)
        ax[0].tick_params(axis='y', labelleft=True)

        ax[1].plot(wrap._svds[k2]['s'])
        ax[1].set_yscale('log')
        ax[1].set_title('.mlp.0')
        ax[1].grid(True, linestyle='--', alpha=0.5)
        ax[1].tick_params(axis='y', labelleft=True)

        ax[2].plot(wrap._svds[k3]['s'])
        ax[2].set_yscale('log')
        ax[2].set_title('.mlp.3')
        ax[2].grid(True, linestyle='--', alpha=0.5)
        ax[2].tick_params(axis='y', labelleft=True)

        plt.savefig(fname=dir_path/Path(_l))
        plt.close(fig)


def viz_compare_per_layer_type(wrap, dir_path):

    # create folder / check if folder already exist
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    
    dict_layers = {}
    dict_layers['out_proj'] = [f"encoder.layers.encoder_layer_{i}.self_attention.out_proj" for i in range(0, 12)]
    dict_layers['l_mlp0'] = [f"encoder.layers.encoder_layer_{i}.mlp.0" for i in range(0, 12)]
    dict_layers['l_mlp3'] = [f"encoder.layers.encoder_layer_{i}.mlp.3" for i in range(0, 12)]

    for key_layers in dict_layers:
        
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)


        for _k in dict_layers[key_layers]:

            ax.plot(wrap._svds[_k]['s'], label=_k, alpha=0.6)
            ax.set_yscale('log')
            ax.set_title(key_layers)
            ax.set_xlabel('Index')
            ax.set_ylabel('Singular Value')
            ax.grid(True, linestyle='--', alpha=0.5)
            
        plt.savefig(fname=dir_path/Path(key_layers))
        plt.close(fig)