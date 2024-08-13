#include<torch/extension.h>
#include "utils.h"


torch::Tensor trilinear_interpolation_fw(
    torch::Tensor feats,
    torch::Tensor point
){
    CHECK_INPUT(feats);
    CHECK_INPUT(point);
    return trilinear_fw_cu(feats, point);
}


torch::Tensor trilinear_interpolation_bw(
    torch::Tensor dL_dfeat_interp,
    torch::Tensor feats,
    torch::Tensor point
){
    CHECK_INPUT(dL_dfeat_interp);
    CHECK_INPUT(feats);
    CHECK_INPUT(point);
    return trilinear_bw_cu(dL_dfeat_interp, feats, point);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("trilinear_interpolation_fw", &trilinear_interpolation_fw);
    m.def("trilinear_interpolation_bw", &trilinear_interpolation_bw);
}