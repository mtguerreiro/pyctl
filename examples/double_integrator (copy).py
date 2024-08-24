import numpy as np

a = np.random.randn(6)
b = np.random.randn(4,4)

def _export_np_array_to_c(self, arr, arr_name, fill=True):

    if arr.ndim == 1:
        n = arr.shape[0]
        m = 1
    else:
        if (arr.shape[0] == 1) or (arr.shape[1] == 1):
            arr = arr.flatten()
            n = arr.shape[0]
            m = 1
        else:
            n, m = arr.shape

    arr_str = np.array2string(arr, separator=',')
    arr_str = arr_str.replace('[', '{')
    arr_str = arr_str.replace(']', '}')

    if m == 1:
        arr_txt = '{:}[{:}];'.format(arr_name, n)
    else:
        arr_txt = '{:}[{:}][{:}];'.format(arr_name, n, m)

    if fill is True:
        arr_txt = arr_txt[:-1] + ' = {:};'.format(arr_str)
        
    return arr_txt
