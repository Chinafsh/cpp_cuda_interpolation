#include<torch/extension.h>


template <typename scalar_t>
__global__ void trilinear_fw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 3 ,torch::RestrictPtrTraits, size_t> feats,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> points,
    torch::PackedTensorAccessor<scalar_t, 2,torch::RestrictPtrTraits, size_t> feat_interp
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int f = blockIdx.y * blockDim.y + threadIdx.y;

    if (n<feats.size(0) && f<feats.size(2)){    // 过滤掉被Block盖住的但不用计算的threads

        // point [-1, 1]
        const scalar_t u = (points[n][0]+1)/2;
        const scalar_t v = (points[n][1]+1)/2;
        const scalar_t w = (points[n][2]+1)/2;

        const scalar_t a = (1-v)*(1-w);
        const scalar_t b = (1-v)*w;
        const scalar_t c = v*(1-w);
        const scalar_t d = 1-a-b-c;
        feat_interp[n][f] = (1-u) * (a * feats[n][0][f]+
                                     b * feats[n][1][f]+
                                     c * feats[n][2][f]+
                                     d * feats[n][3][f])
                            + u *   (a * feats[n][4][f]+
                                     b * feats[n][5][f]+
                                     c * feats[n][6][f]+
                                     d * feats[n][7][f]);

    }
}

torch::Tensor trilinear_fw_cu(  // fw--forward,  cu--cuda function
    torch::Tensor feats,
    torch::Tensor points
){
    // N = feats.shape[0]
    // F = feats.shape[2]
    const int N = feats.size(0), F = feats.size(2);
    
    // feat_interp = torch.zeros(N,F, dtype=torch.float32, device='cuda')
    torch::Tensor feat_interp = torch::zeros({N,F}, feats.options());// feats.options(), 指定数据类型和仪器
    // torch::zeros({N,F}, torch::dtype(torch::kInt32).device(feats.device()));

    const dim3 threads(16,16,1);  // 256
    const dim3 blocks((N+threads.x-1)/threads.x, (F+threads.y-1)/threads.y);

    AT_DISPATCH_FLOATING_TYPES(feats.type(), "trilinear_fw_cu",
    ([&]{
        trilinear_fw_kernel<scalar_t><<<blocks, threads>>>(
            feats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            points.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            feat_interp.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
        );
    }));
    return feat_interp;

}


template <typename scalar_t>
__global__ void trilinear_bw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2 ,torch::RestrictPtrTraits, size_t> dL_dfeat_interp,
    const torch::PackedTensorAccessor<scalar_t, 3 ,torch::RestrictPtrTraits, size_t> feats,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> points,
    torch::PackedTensorAccessor<scalar_t, 3,torch::RestrictPtrTraits, size_t> dL_dfeat
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int f = blockIdx.y * blockDim.y + threadIdx.y;

    if (n<feats.size(0) && f<feats.size(2)){    // 过滤掉被Block盖住的但不用计算的threads

        // point [-1, 1]
        const scalar_t u = (points[n][0]+1)/2;
        const scalar_t v = (points[n][1]+1)/2;
        const scalar_t w = (points[n][2]+1)/2;

        const scalar_t a = (1-v)*(1-w);
        const scalar_t b = (1-v)*w;
        const scalar_t c = v*(1-w);
        const scalar_t d = 1-a-b-c;

        dL_dfeat[n][0][f] = (1-u) * a * dL_dfeat_interp[n][f];
        dL_dfeat[n][1][f] = (1-u) * b * dL_dfeat_interp[n][f];
        dL_dfeat[n][2][f] = (1-u) * c * dL_dfeat_interp[n][f];
        dL_dfeat[n][3][f] = (1-u) * d * dL_dfeat_interp[n][f];
        dL_dfeat[n][4][f] = u * a * dL_dfeat_interp[n][f];
        dL_dfeat[n][5][f] = u * b * dL_dfeat_interp[n][f];
        dL_dfeat[n][6][f] = u * c * dL_dfeat_interp[n][f];
        dL_dfeat[n][7][f] = u * d * dL_dfeat_interp[n][f];

    }
}



torch::Tensor trilinear_bw_cu(  // fw--forward,  cu--cuda function
    const torch::Tensor dL_dfeat_interp,
    const torch::Tensor feats,
    const torch::Tensor points
){
    // N = feats.shape[0]
    // F = feats.shape[2]
    const int N = feats.size(0), F = feats.size(2);
    
    torch::Tensor dL_dfeat = torch::zeros({N,8,F}, feats.options());// feats.options(), 指定数据类型和仪器
    const dim3 threads(16,16,1);  // 256
    const dim3 blocks((N+threads.x-1)/threads.x, (F+threads.y-1)/threads.y);

    AT_DISPATCH_FLOATING_TYPES(feats.type(), "trilinear_bw_cu",
    ([&]{
        trilinear_bw_kernel<scalar_t><<<blocks, threads>>>(
            dL_dfeat_interp.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            feats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            points.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            dL_dfeat.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>()
        );
    }));
    return dL_dfeat;

}




