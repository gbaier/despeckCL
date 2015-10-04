#!/usr/bin/octave
clear;
clc;
despeckcl;

load('insar_test_data.mat')

% boxcar
window_size = 5;
[amp_filt_boxcar, phi_filt_boxcar, coh_filt_boxcar] = despeckcl.boxcar(ampl_master,
                                                                       ampl_slave,
                                                                       dphase,
                                                                       window_size);

% nlinsar
search_window_size = 21;
patch_size = 5;
niter = 2;
lmin = 10;
[amp_filt_nlinsar, phi_filt_nlinsar, coh_filt_nlinsar] = despeckcl.nlinsar(ampl_master,
                                                                           ampl_slave,
                                                                           dphase,
                                                                           search_window_size,
                                                                           patch_size,
                                                                           niter,
                                                                           lmin);
% nlsar
search_window_size = 21;
patch_sizes = despeckcl.IntVector(4);
patch_sizes(0) = 3;
patch_sizes(1) = 5;
patch_sizes(2) = 7;
patch_sizes(3) = 9;
scale_sizes = despeckcl.IntVector(3);
scale_sizes(0) = 1;
scale_sizes(1) = 3;
scale_sizes(2) = 5;
[amp_filt_nlsar, phi_filt_nlsar, coh_filt_nlsar] = despeckcl.nlsar(ampl_master,
                                                                   ampl_slave,
                                                                   dphase,
                                                                   search_window_size,
                                                                   patch_sizes,
                                                                   scale_sizes);


figure('Position',[0,0,800,600]);

function scaled_img = scale_img (img)
    img = img - min(img(:));
    scaled_img = 64*img/max(img(:));
endfunction

cmaps = [gray; jet; gray];
colormap(cmaps)

subplot(331)
image(scale_img(20*log10(abs(amp_filt_boxcar))));

subplot(332)
image(64 + scale_img(phi_filt_boxcar));

subplot(333)
image(128 + scale_img(coh_filt_boxcar));

subplot(334)
image(scale_img(20*log10(abs(amp_filt_nlinsar))));

subplot(335)
image(64 + scale_img(phi_filt_nlinsar));

subplot(336)
image(128 + scale_img(coh_filt_nlinsar));

subplot(337)
image(scale_img(20*log10(abs(amp_filt_nlsar))));

subplot(338)
image(64 + scale_img(phi_filt_nlsar));

subplot(339)
image(128 + scale_img(coh_filt_nlsar));

pause
