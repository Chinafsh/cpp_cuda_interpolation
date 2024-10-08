#include<torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor trilinear_fw_cu(  // fw--forward,  cu--cuda function
    torch::Tensor feats,
    torch::Tensor points
);

torch::Tensor trilinear_bw_cu(  // backward
    torch::Tensor dL_dfeat_interp,
    torch::Tensor feats,
    torch::Tensor points
);