float det_covmat_2x2(float el_00, float el_01real, float el_01imag, float el_11)
{
    return (el_00*el_11) - (el_01real*el_01real + el_01imag*el_01imag);
}

float pixel_similarity_2x2(float el_00_p1, float el_01real_p1, float el_01imag_p1, float el_11_p1,
                           float el_00_p2, float el_01real_p2, float el_01imag_p2, float el_11_p2,
                           const int nlooks)
{
    const int dimensions = 2;
    float nom1 = det_covmat_2x2(el_00_p1, el_01real_p1, el_01imag_p1, el_11_p1);
    float nom2 = det_covmat_2x2(el_00_p2, el_01real_p2, el_01imag_p2, el_11_p2);
    float det  = det_covmat_2x2(el_00_p1     + el_00_p2,
                                el_01real_p1 + el_01real_p2,
                                el_01imag_p1 + el_01imag_p2,
                                el_11_p1     + el_11_p2);

    float similarity = -nlooks*( 2*dimensions*log(2.0f) +  log(nom1) + log(nom2) - 2*log(det) );
    if (isnan(similarity)) {
        similarity = 0;
    }
    return similarity;
}
