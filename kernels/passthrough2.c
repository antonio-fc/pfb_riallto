void passthrough(float *in_buffer, float *in_buffer2, float *out_buffer, uint32_t nbytes){
    const uint8_t vector_lanes = 16;
    const uint8_t vec_factor = 4;
    
    // a buffer to temporarily store our vector   
    ::aie::vector<float, vector_lanes> buffer;
    ::aie::vector<float, vector_lanes> buffer2;

    // divide by vectorization factor (64)
    uint32_t loop_count = nbytes >> vec_factor; 
    
    for(int i=0; i<loop_count; i++) {
        
         // load 64 elements into the buffer
        buffer = ::aie::load_v<vector_lanes>(in_buffer);
        buffer2 = ::aie::load_v<vector_lanes>(in_buffer2);

        // store buffer into the out buffer
        ::aie::store_v(out_buffer, buffer); 

         // We need to increment the buffers by 64 each iteration now
        in_buffer += vector_lanes;
        in_buffer2 += vector_lanes;
        out_buffer += vector_lanes;
    }
}