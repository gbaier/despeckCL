float enl_nobias(float enl, float alpha, float wsum) // Eq. 20 in Deledalle's NLSAR paper
{
    return enl/(pow(1.0f-alpha, 2.0f) + (pow(alpha, 2.0f) + 2.0f*alpha*(1.0f-alpha)/wsum)*enl);
    //return enl/(pow(1.0f-alpha, 2.0f) + enl*pow((1-alpha)/wsum+alpha, 2.0f));
}

__kernel void compute_enls_nobias (__global float * enls,
                                   __global float * alphas,
                                   __global float * wsums,
                                   __global float * enls_nobias,
                                   const int height_ori,
                                   const int width_ori)
{
    const int tx = get_global_id(0);
    const int ty = get_global_id(1);

    if (tx < height_ori && ty < width_ori) {
        const int idx = tx * width_ori + ty;
        enls_nobias[idx] = enl_nobias( enls[idx], alphas[idx], wsums[idx] );
    }
}
