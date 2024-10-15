using namespace aie;
#include <math.h>

# define M_PI           3.14159265358979323846  /* pi */

vector<float, 16> twiddle16() {
    const float t = 2.0 * M_PI / 16.0;
    
    vector<float, 16> seq;
    for (unsigned i = 0; i < 16; i++) {
        seq[i] = t * i; // i-th element is updated
    }

    // vector<float, 16> real_factor = aie::sqrt(seq);
    auto complex_factor = aie::sin(seq);

    return seq;

}

vector<float, 16> fft16(vector<float, 16> real16, vector<float, 16> comp16) {
    // Getting the real and complex component of the data
    auto c = filter_even(real16, 1); // real_even16
    auto d = filter_even(comp16, 1); // comp_even16

    auto a = filter_odd(real16, 1); // real_odd16
    auto b = filter_odd(comp16, 1); // comp_odd16

    // Getting the twiddle factors
    auto t = twiddle16();

    // Result
    return t;
}

void fft_kernel(float *in_buffer, float *out_buffer, uint32_t nbytes){
    const uint8_t lanes = 16;
    const uint8_t vec_factor = 4;
     
    auto real16 = load_v<lanes>(in_buffer);
    auto comp16 = broadcast<float, lanes>(0.0);

    auto concat_result = fft16(real16, comp16);

    // Pasthrough stuff
    uint32_t loop_count = nbytes >> vec_factor;
    auto buffer = load_v<lanes>(in_buffer);
    
    for(int i=0; i<loop_count; i++) {
        store_v(out_buffer, concat_result); 
        in_buffer += lanes;
        out_buffer += lanes;
    }
}