__kernel void compute_weights (__global float * sims, __global float * kls, __global float * weights, const int size, const float h, const float T)
{
    const int tx = get_global_id(0);

    if( tx < size ) {
        float weight = exp(sims[tx] / h  +  kls[tx] / T);
        if (isnan(weight) || isinf(weight)) {
            weight = 0.0f;
        }
        weights[tx] = weight;
    }
}
