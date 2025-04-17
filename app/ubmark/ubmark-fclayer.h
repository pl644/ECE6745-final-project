#ifndef UBMARK_FCLAYER_H
#define UBMARK_FCLAYER_H

void ubmark_fclayer(
    float* input,
    float* weights,
    float* bias,
    float* output,
    int batch,
    int channel_in,
    int channel_out
);

#endif /* UBMARK_ACCUM_H */