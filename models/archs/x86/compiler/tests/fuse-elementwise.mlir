// CHECK-LABEL: func.func @one_to_one
// CHECK-COUNT-1: linalg.generic
// CHECK: arith.mulf
// CHECK-NOT: linalg.generic
// CHECK: arith.addf
// CHECK-NOT: linalg.generic
// CHECK: return
func.func @one_to_one(%a: tensor<8xf32>, %b: tensor<8xf32>) -> tensor<8xf32> {
  %empty = tensor.empty() : tensor<8xf32>
  %square = linalg.generic {indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>], iterator_types = ["parallel"]}
      ins(%a : tensor<8xf32>) outs(%empty : tensor<8xf32>) {
  ^bb0(%x: f32, %out: f32):
    %value = arith.mulf %x, %x : f32
    linalg.yield %value : f32
  } -> tensor<8xf32>
  %result = linalg.generic {indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>, affine_map<(i) -> (i)>], iterator_types = ["parallel"]}
      ins(%square, %b : tensor<8xf32>, tensor<8xf32>) outs(%empty : tensor<8xf32>) {
  ^bb0(%x: f32, %y: f32, %out: f32):
    %value = arith.addf %x, %y : f32
    linalg.yield %value : f32
  } -> tensor<8xf32>
  return %result : tensor<8xf32>
}

// CHECK-LABEL: func.func @reduction_boundary
// CHECK-COUNT-2: linalg.generic
// CHECK: return
func.func @reduction_boundary(%a: tensor<8xf32>, %init: tensor<f32>) -> tensor<f32> {
  %empty = tensor.empty() : tensor<8xf32>
  %square = linalg.generic {indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>], iterator_types = ["parallel"]}
      ins(%a : tensor<8xf32>) outs(%empty : tensor<8xf32>) {
  ^bb0(%x: f32, %out: f32):
    %value = arith.mulf %x, %x : f32
    linalg.yield %value : f32
  } -> tensor<8xf32>
  %result = linalg.generic {indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> ()>], iterator_types = ["reduction"]}
      ins(%square : tensor<8xf32>) outs(%init : tensor<f32>) {
  ^bb0(%x: f32, %sum: f32):
    %value = arith.addf %sum, %x : f32
    linalg.yield %value : f32
  } -> tensor<f32>
  return %result : tensor<f32>
}

// CHECK-LABEL: func.func @broadcast_boundary
// CHECK-COUNT-2: linalg.generic
// CHECK: return
func.func @broadcast_boundary(%a: tensor<1xf32>, %b: tensor<8xf32>) -> tensor<8xf32> {
  %small = tensor.empty() : tensor<1xf32>
  %empty = tensor.empty() : tensor<8xf32>
  %square = linalg.generic {indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>], iterator_types = ["parallel"]}
      ins(%a : tensor<1xf32>) outs(%small : tensor<1xf32>) {
  ^bb0(%x: f32, %out: f32):
    %value = arith.mulf %x, %x : f32
    linalg.yield %value : f32
  } -> tensor<1xf32>
  %result = linalg.generic {indexing_maps = [affine_map<(i) -> (0)>, affine_map<(i) -> (i)>, affine_map<(i) -> (i)>], iterator_types = ["parallel"]}
      ins(%square, %b : tensor<1xf32>, tensor<8xf32>) outs(%empty : tensor<8xf32>) {
  ^bb0(%x: f32, %y: f32, %out: f32):
    %value = arith.addf %x, %y : f32
    linalg.yield %value : f32
  } -> tensor<8xf32>
  return %result : tensor<8xf32>
}
