module  {
   AIE.device(ipu){

   %tile00 = AIE.tile(0, 0)
   %tile01 = AIE.tile(0, 1)
   %tile02 = AIE.tile(0, 2)
   %rtp_0_2 = AIE.buffer(%tile02) { sym_name = "rtp_0_2" } : memref<4xi32>
   AIE.objectFifo @itbuffer_0___ITout___mtbuffer_0___MTin(%tile00, {%tile01}, 2 : i32) : !AIE.objectFifo<memref<2048xi32>>
   AIE.objectFifo @mtbuffer_0___MTout___frontend_0___in_buffer(%tile01, {%tile02}, 2 : i32) : !AIE.objectFifo<memref<2048xi32>>
   AIE.objectFifo @itbuffer_1___ITout___frontend_0___in_buffer2(%tile00, {%tile02}, 2 : i32) : !AIE.objectFifo<memref<2048xi32>>
   AIE.objectFifo @frontend_0___out_buffer___itbuffer_2___ITin(%tile02, {%tile00}, 2 : i32) : !AIE.objectFifo<memref<1024xi32>>

   AIE.objectFifo.link [@itbuffer_0___ITout___mtbuffer_0___MTin ] -> [@mtbuffer_0___MTout___frontend_0___in_buffer] ()

   func.func private @frontend(memref<2048xi32>, memref<2048xi32>, memref<1024xi32>, i32) -> ()

   AIE.core(%tile02) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %intmax = arith.constant 0xFFFFFFFF : index
      %c_rtpidx_3 = arith.constant 3 : index
      scf.for %arg3 = %c0 to %intmax step %c1 {
         %subview2 = AIE.objectFifo.acquire @mtbuffer_0___MTout___frontend_0___in_buffer(Consume, 1) : !AIE.objectFifoSubview<memref<2048xi32>>
         %elem2 = AIE.objectFifo.subview.access %subview2[0] : !AIE.objectFifoSubview<memref<2048xi32>> -> memref<2048xi32>
         %subview3 = AIE.objectFifo.acquire @itbuffer_1___ITout___frontend_0___in_buffer2(Consume, 1) : !AIE.objectFifoSubview<memref<2048xi32>>
         %elem3 = AIE.objectFifo.subview.access %subview3[0] : !AIE.objectFifoSubview<memref<2048xi32>> -> memref<2048xi32>
         %subview4 = AIE.objectFifo.acquire @frontend_0___out_buffer___itbuffer_2___ITin(Produce, 1) : !AIE.objectFifoSubview<memref<1024xi32>>
         %elem4 = AIE.objectFifo.subview.access %subview4[0] : !AIE.objectFifoSubview<memref<1024xi32>> -> memref<1024xi32>

         %n = memref.load %rtp_0_2[%c_rtpidx_3] : memref<4xi32>
         func.call @frontend(%elem2, %elem3, %elem4, %n) : (memref<2048xi32>, memref<2048xi32>, memref<1024xi32>, i32) -> ()

         AIE.objectFifo.release @mtbuffer_0___MTout___frontend_0___in_buffer(Consume, 1)
         AIE.objectFifo.release @itbuffer_1___ITout___frontend_0___in_buffer2(Consume, 1)
         AIE.objectFifo.release @frontend_0___out_buffer___itbuffer_2___ITin(Produce, 1)
      }
      AIE.end
   } { link_with="frontend.o" }

func.func @sequence(%itbuffer_0 : memref<2048xi32>,%itbuffer_1 : memref<2048xi32>,%itbuffer_2 : memref<1024xi32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c8192 = arith.constant 8192 : i32
    %c512 = arith.constant 512 : i32
    %c2048 = arith.constant 2048 : i32
    %c4096 = arith.constant 4096 : i32
    %c256 = arith.constant 256 : i32
    %c1024 = arith.constant 1024 : i32

    AIEX.ipu.rtp_write(0, 2, 3, 3) { buffer_sym_name = "rtp_0_2" }

    AIEX.ipu.dma_memcpy_nd(%c0, %c0,%itbuffer_2[%c0, %c0, %c0, %c0][%c1, %c1, %c1, %c1024][%c0, %c0, %c0]){ metadata= @frontend_0___out_buffer___itbuffer_2___ITin, id = 2 : i32 } :(i32, i32, memref<1024xi32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])

    AIEX.ipu.dma_memcpy_nd(%c0, %c0,%itbuffer_0[%c0, %c0, %c0, %c0][%c1, %c1, %c1, %c2048][%c0, %c0, %c0]){ metadata= @itbuffer_0___ITout___mtbuffer_0___MTin, id = 0 : i32 } :(i32, i32, memref<2048xi32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])

    AIEX.ipu.dma_memcpy_nd(%c0, %c0,%itbuffer_1[%c0, %c0, %c0, %c0][%c1, %c1, %c1, %c2048][%c0, %c0, %c0]){ metadata= @itbuffer_1___ITout___frontend_0___in_buffer2, id = 1 : i32 } :(i32, i32, memref<2048xi32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])

    AIEX.ipu.sync {column = 0 : i32, row = 0 : i32, direction = 0 : i32, channel = 0 : i32, column_num = 1 : i32, row_num = 1 : i32 }
    return
}
 }
}
