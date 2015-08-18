float det_covmat_2x2(float el_00, float el_01real, float el_01imag, float el_11)
{
    return (el_00*el_11) - (el_01real*el_01real + el_01imag*el_01imag);
}

float pixel_similarity(float el_00_p1, float el_01real_p1, float el_01imag_p1, float el_11_p1,
                       float el_00_p2, float el_01real_p2, float el_01imag_p2, float el_11_p2)
{
    // FIXME
    const int nlooks = 9;
    const int dim = 2;
    float nom1 = det_covmat_2x2(el_00_p1, el_01real_p1, el_01imag_p1, el_11_p1);
    float nom2 = det_covmat_2x2(el_00_p2, el_01real_p2, el_01imag_p2, el_11_p2);
    float det  = det_covmat_2x2(el_00_p1     + el_00_p2,
                                el_01real_p1 + el_01real_p2,
                                el_01imag_p1 + el_01imag_p2,
                                el_11_p1     + el_11_p2);

    const float retval = -nlooks*( 2*dim*std::log(2) +  std::log(nom1) + std::log(nom2) - 2*std::log(det) );
    if ( std::isnan(retval) || std::isinf(retval) ) {
        return 0.0;
    } else {
        return retval;
    }
}
