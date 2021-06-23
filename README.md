# ncnn_Micro

## introduction

none-core function to edit & MICRO ncnn model:

features in the plan:

* disassemble `ncnn_model.bin` layer by layer
* assemble `ncnn_layer.bin` into `model.bin`
* layer inference data-type convertion (f32, f16, int8, ...)
* cli ui

## developing platform

* ubuntu 20.04 
* g++ capable with c++17
* ncnn shared library
* glog (apt install)
