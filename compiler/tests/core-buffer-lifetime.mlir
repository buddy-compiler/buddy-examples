// Transfer aliases must be resolved before finding the last use. The two
// independent jobs remain asynchronous, and release their CPU inputs after
// their shared wait, before allocating the next layer's workspace.
// CHECK-LABEL: func.func @lifetime
// CHECK: %[[A:[a-zA-Z0-9_]+]] = memref.alloc()
// CHECK: %[[B:[a-zA-Z0-9_]+]] = memref.alloc()
// CHECK: call @stream_begin
// CHECK: call @copy0_launch
// CHECK-NOT: call @stream_wait
// CHECK: call @copy1_launch
// CHECK: call @stream_submit
// CHECK: call @stream_wait
// CHECK: memref.dealloc %[[A]]
// CHECK: memref.dealloc %[[B]]
// CHECK: call @consume
// CHECK: memref.alloc() {{.*}}memref<1024xf32>

module {
  func.func private @copy0(%input: memref<4xf32>, %output: memref<4xf32>) {
    memref.copy %input, %output : memref<4xf32> to memref<4xf32>
    return
  }
  func.func private @copy1(%input: memref<4xf32>, %output: memref<4xf32>) {
    memref.copy %input, %output : memref<4xf32> to memref<4xf32>
    return
  }
  func.func private @consume(memref<4xf32>, memref<4xf32>)
  func.func private @workspace(memref<1024xf32>)
  func.func @lifetime(%seed0: memref<4xf32>, %seed1: memref<4xf32>,
                     %out0: memref<4xf32>, %out1: memref<4xf32>) {
    %a = memref.alloc() : memref<4xf32>
    %b = memref.alloc() : memref<4xf32>
    memref.copy %seed0, %a : memref<4xf32> to memref<4xf32>
    memref.copy %seed1, %b : memref<4xf32> to memref<4xf32>
    %ta = memref.alloc() : memref<4xf32>
    %tb = memref.alloc() : memref<4xf32>
    core.transfer %a to(%ta) <to = 0> : (memref<4xf32>, memref<4xf32>) -> ()
    core.launch @copy0(%ta) outs(%out0) <thread = 0> : (memref<4xf32>, memref<4xf32>) -> ()
    core.transfer %b to(%tb) <to = 1> : (memref<4xf32>, memref<4xf32>) -> ()
    core.launch @copy1(%tb) outs(%out1) <thread = 1> : (memref<4xf32>, memref<4xf32>) -> ()
    call @consume(%out0, %out1) : (memref<4xf32>, memref<4xf32>) -> ()
    %later = memref.alloc() : memref<1024xf32>
    call @workspace(%later) : (memref<1024xf32>) -> ()
    memref.dealloc %later : memref<1024xf32>
    memref.dealloc %ta : memref<4xf32>
    memref.dealloc %tb : memref<4xf32>
    memref.dealloc %a : memref<4xf32>
    memref.dealloc %b : memref<4xf32>
    return
  }
}
