void frontend(float *in_buffer, float *in_buffer2, float *out_buffer, uint32_t n){
    const uint8_t vector_lanes = 16;
    const uint8_t vec_factor = 4;
    
    // a buffer to temporarily store our vector   
    ::aie::vector<float, vector_lanes> bufferA1;
    ::aie::vector<float, vector_lanes> bufferA2;
    ::aie::vector<float, vector_lanes> bufferB1;
    ::aie::vector<float, vector_lanes> bufferB2;
    ::aie::vector<float, vector_lanes> opA;
    ::aie::vector<float, vector_lanes> opB;
    ::aie::vector<float, vector_lanes> sum;

    // divide by vectorization factor (64)
    uint32_t loop_count = n >> vec_factor;
    uint32_t half_count = loop_count >> 1;
    uint32_t half = n >> 1;

    for(int i=0; i<half_count; i++) {
        // load 64 elements into the buffer
        bufferA1 = ::aie::load_v<vector_lanes>(in_buffer);
        bufferA2 = ::aie::load_v<vector_lanes>(in_buffer2);
        bufferB1 = ::aie::load_v<vector_lanes>(in_buffer+half);
        bufferB2 = ::aie::load_v<vector_lanes>(in_buffer2+half);
        
        opA = aie::mul(bufferA1, bufferA2);
        opB = aie::mul(bufferB1, bufferB2);

        sum = aie::add(opA, opB);
        
        aie::store_v(out_buffer, sum);
    
         // We need to increment the buffers by 64 each iteration now
        in_buffer += vector_lanes;
        in_buffer2 += vector_lanes;
        out_buffer += vector_lanes;
    }
    // for(int i=0; i<loop_count; i++) {
    //      // load 64 elements into the buffer
    //     buffer = ::aie::load_v<vector_lanes>(in_buffer);
    //     buffer2 = ::aie::load_v<vector_lanes>(in_buffer2);
    //     if(i >= half) {
    //         op1 = aie::mul(buffer, buffer2);
    //         op2 = aie::load_v<vector_lanes>(out_buffer-half);
    //         sum = aie::add(op1, op2); 
    //     }
    //     else {
    //         sum = aie::mul(buffer, buffer2);
    //         aie::store_v(out_buffer, sum);
    //     }
    
    //      // We need to increment the buffers by 64 each iteration now
    //     in_buffer += vector_lanes;
    //     in_buffer2 += vector_lanes;
    //     out_buffer += vector_lanes;
    // }
}