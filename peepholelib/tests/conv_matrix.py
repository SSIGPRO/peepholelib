import torch
from torch.nn import Conv2d
from scipy.signal import convolve2d

def toeplitz_mult_ch(kernel, input_size):
    kernel_size = kernel.shape
    output_size = (kernel_size[0], input_size[1] - (kernel_size[1]-1), input_size[2] - (kernel_size[2]-1))
    T = np.zeros((output_size[0], int(np.prod(output_size[1:])), input_size[0], int(np.prod(input_size[1:]))))

    for i,ks in enumerate(kernel):  # loop over output channel
        for j,k in enumerate(ks):  # loop over input channel
            T_k = toeplitz_1_ch(k, input_size[1:])
            T[i, :, j, :] = T_k

    T.shape = (np.prod(output_size), np.prod(input_size))

    return T

if __name__ == "__main__":
    from torch.nn import Conv2d
    n_channels = 2
    n_out=3
    kernel_size =3
    batch_size=1
    x_size = (4, 5)

    layer = Conv2d(n_channels, n_out, kernel_size)
    w = layer.weight
    b = layer.bias
    print('\nw, b:', w, b)

    x=torch.rand((batch_size, n_channels)+x_size)
    print('\nx: ', x)
    y = layer(x)
    print('\ny: ', y)
    
    print('\n')
    for xx in x:
        for out_k in w:
            #print('out kernel: ', out_k)
            for ch_k, ch_x in zip(out_k, xx):
                #print('channel kernel: ', ch_k)
                #print('channel x: ', ch_x)
                a = ch_x.detach().numpy()
                b = ch_k.detach().numpy()
                #print(a, b)
                r = convolve2d(a, b)
                print('r: ', r)
