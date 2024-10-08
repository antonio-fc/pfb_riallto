void pfb_fir_frontend(uint8_t *in_buffer, uint8_t *out_buffer, uint32_t nbytes){
    // a buffer to temporarily store our vector
    ::aie::vector<uint8_t, 64> buffer; 

    // divide by vectorization factor (64)
    uint32_t loop_count = nbytes >> 6; 
    
    for(int i=0; i<loop_count; i++) {
        
         // load 64 elements into the buffer
        buffer = ::aie::load_v<64>(in_buffer);

        // store buffer into the out buffer
        ::aie::store_v(out_buffer, buffer); 

         // We need to increment the buffers by 64 each iteration now
        in_buffer += 64;
        out_buffer += 64;
    }
}