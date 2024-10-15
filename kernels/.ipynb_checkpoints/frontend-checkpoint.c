// This kernel implements the frontend for the PFB for M=3
void frontend(float *in_buffer, float *in_buffer2, float *out_buffer, uint32_t n){
    const uint8_t vector_lanes = 16;
    const uint8_t vec_factor = 4;
    
    // a buffer to temporarily store our vector   
    ::aie::vector<float, vector_lanes> bufferA1;
    ::aie::vector<float, vector_lanes> bufferA2;
    ::aie::vector<float, vector_lanes> bufferB1;
    ::aie::vector<float, vector_lanes> bufferB2;
    ::aie::vector<float, vector_lanes> bufferC1;
    ::aie::vector<float, vector_lanes> bufferC2;
    ::aie::vector<float, vector_lanes> opA;
    ::aie::vector<float, vector_lanes> opB;
    ::aie::vector<float, vector_lanes> opC;
    ::aie::vector<float, vector_lanes> tmp;
    ::aie::vector<float, vector_lanes> sum;

    // divide by vectorization factor (64)
    uint32_t loop_count = n >> vec_factor;
    uint32_t third_count = loop_count/3;
    uint32_t third = n/3;    

    for(int i=0; i<third_count; i++) {
        // load 64 elements into the buffer
        bufferA1 = ::aie::load_v<vector_lanes>(in_buffer);
        bufferA2 = ::aie::load_v<vector_lanes>(in_buffer2);
        bufferB1 = ::aie::load_v<vector_lanes>(in_buffer+third);
        bufferB2 = ::aie::load_v<vector_lanes>(in_buffer2+third);
        bufferC1 = ::aie::load_v<vector_lanes>(in_buffer+(third<<1));
        bufferC2 = ::aie::load_v<vector_lanes>(in_buffer2+(third<<1));
        
        opA = aie::mul(bufferA1, bufferA2);
        opB = aie::mul(bufferB1, bufferB2);
        opC = aie::mul(bufferC1, bufferC2);

        tmp = aie::add(opA, opB);

        sum = aie::add(tmp, opC);
        
        aie::store_v(out_buffer, sum);
    
         // We need to increment the buffers by 64 each iteration now
        in_buffer += vector_lanes;
        in_buffer2 += vector_lanes;
        out_buffer += vector_lanes;
    }
}