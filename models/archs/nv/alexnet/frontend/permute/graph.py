def permute(graph):
    graph.node_table["convolution"].tensor_meta["activation_layout"] = "nhwc"
    graph.node_table["convolution_1"].tensor_meta["activation_layout"] = "nhwc"
    graph.node_table["convolution_2"].tensor_meta["activation_layout"] = "nhwc"
    graph.node_table["convolution_3"].tensor_meta["activation_layout"] = "nhwc"
    graph.node_table["convolution_4"].tensor_meta["activation_layout"] = "nhwc"
