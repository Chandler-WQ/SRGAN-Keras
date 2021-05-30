# SRGAN-Keras
超分辨重建毕业设计，主要基于ESRGAN和SRGAN实现超分重建模型，同时实现了LPIPS\NIQE\SSIM\PSNR的计算脚本和画图脚本

运行前需要将一下一些数据下载挪进项目中。

链接：https://pan.baidu.com/s/128aKMREjR26D-doEBovYWA 
提取码：yq0c 


# 运行环境
_tflow_select             2.1.0                       gpu

absl-py                   0.12.0           py36haa95532_0

astor                     0.8.1            py36haa95532_0

blas                      1.0                         mkl

ca-certificates           2021.4.13            haa95532_1

certifi                   2020.12.5        py36haa95532_0

cloudpickle               1.6.0                      py_0

coverage                  5.5              py36h2bbff1b_2

cudatoolkit               9.0                           14

cudnn                     7.6.5                 cuda9.0_0

cycler                    0.10.0           py36haa95532_0

cython                    0.29.23          py36hd77b12b_0

cytoolz                   0.11.0           py36he774522_0

dask-core                 2021.3.0           pyhd3eb1b0_0

dataclasses               0.8                       <pip>
  
decorator                 5.0.6              pyhd3eb1b0_0
  
et_xmlfile                1.0.1                   py_1001
  
freetype                  2.10.4               hd328e21_0
  
future                    0.18.2                    <pip>
  
gast                      0.4.0                      py_0
  
grpcio                    1.36.1           py36hc60d5dd_1
  
h5py                      2.10.0           py36h5e291fa_0
  
hdf5                      1.10.4               h7ebc959_0
  
icc_rt                    2019.0.0             h0cc432a_1
  
icu                       58.2                 ha925a31_3
  
imageio                   2.9.0              pyhd3eb1b0_0
  
importlib-metadata        3.10.0           py36haa95532_0
  
intel-openmp              2020.2                      254
  
jdcal                     1.4.1                      py_0
  
jpeg                      9b                   hb83a4c4_2
  
keras                     2.2.4                         0
  
keras-applications        1.0.8                      py_1
  
keras-base                2.2.4                    py36_0
  
keras-preprocessing       1.1.2              pyhd3eb1b0_0
  
kiwisolver                1.3.1            py36hd77b12b_0
  
libpng                    1.6.37               h2a8f88b_0
  
libprotobuf               3.14.0               h23ce68f_0
  
libtiff                   4.1.0                h56a325e_1
  
lpips                     0.1.3                     <pip>
  
lz4-c                     1.9.3                h2bbff1b_0
  
markdown                  3.3.4            py36haa95532_0
  
matplotlib                3.3.4            py36haa95532_0
  
matplotlib-base           3.3.4            py36h49ac443_0
  
mkl                       2020.2                      256
  
mkl-service               2.3.0            py36h196d8e1_0
  
mkl_fft                   1.3.0            py36h46781fe_0
  
mkl_random                1.1.1            py36h47e9c7a_0
  
networkx                  2.5                        py_0
  
numpy                     1.19.2           py36hadc3359_0
  
numpy-base                1.19.2           py36ha3acd2a_0
  
olefile                   0.46                       py_0
  
opencv-python             4.5.1.48                  <pip>
  
openpyxl                  3.0.7              pyhd3eb1b0_0
  
openssl                   1.1.1k               h2bbff1b_0
  
pillow                    8.2.0            py36h4fa10fc_0
  
pip                       21.0.1           py36haa95532_0
  
protobuf                  3.14.0           py36hd77b12b_1
  
pyparsing                 2.4.7              pyhd3eb1b0_0
  
pyqt                      5.9.2            py36h6538335_2
  
pyreadline                2.1                      py36_1
  
python                    3.6.13               h3758d61_0
  
python-dateutil           2.8.1              pyhd3eb1b0_0
  
pywavelets                1.1.1            py36he774522_2
  
pyyaml                    5.4.1            py36h2bbff1b_1
  
qt                        5.9.7            vc14h73c81de_0
  
scikit-image              0.15.0           py36ha925a31_0
  
scipy                     1.5.2            py36h9439919_0
  
setuptools                52.0.0           py36haa95532_0
  
sip                       4.19.8           py36h6538335_0
  
six                       1.15.0             pyhd3eb1b0_0
  
sqlite                    3.35.4               h2bbff1b_0
  
tensorboard               1.11.0           py36he025d50_0
  
tensorflow                1.11.0          gpu_py36h5dc63e2_0
  
tensorflow-base           1.11.0          gpu_py36h6e53903_0
  
tensorflow-gpu            1.11.0               h0d30ee6_0
  
termcolor                 1.1.0            py36haa95532_1
  
tk                        8.6.10               he774522_0
  
toolz                     0.11.1             pyhd3eb1b0_0
  
torch                     1.8.1                     <pip>
  
torchvision               0.9.1                     <pip>
  
tornado                   6.1              py36h2bbff1b_0
  
tqdm                      4.36.1                     py_0
  
typing_extensions         3.7.4.3            pyha847dfd_0
  
vc                        14.2                 h21ff451_1
  
vs2015_runtime            14.27.29016          h5e58377_2
  
werkzeug                  1.0.1              pyhd3eb1b0_0
  
wheel                     0.36.2             pyhd3eb1b0_0
  
wincertstore              0.2              py36h7fe50ca_0
  
xz                        5.2.5                h62dcd97_0
  
yaml                      0.2.5                he774522_0
  
zipp                      3.4.1              pyhd3eb1b0_0
  
zlib                      1.2.11               h62dcd97_4
  
zstd                      1.4.9                h19a0ad4_0
