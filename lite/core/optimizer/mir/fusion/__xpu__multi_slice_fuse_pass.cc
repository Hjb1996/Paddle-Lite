// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <memory>
#include <set>
#include <string>
#include <vector>
#include "lite/backends/xpu/math.h"
#include "lite/core/context.h"
#include "lite/core/optimizer/mir/pass_registry.h"
#include "lite/core/optimizer/mir/type_precision_cast_pass.h"  // For UpdateInputs()
#include "lite/core/optimizer/mir/xpu_pattern_matcher_high_api.h"
#include "lite/operators/subgraph_op.h"

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

/* fuse multi slice after encoder_out */
/* graph                                                 */
/*                                                       */
/*                        encoder_out                      */
/*                    /   /  |  \    \                   */
/*                /      |   |      \     \              */
/*              /        |   |       |    |              */
/*             |         |   |       |    |              */
/*             |         |   |       |    |              */
/*           slice   slice  slice   ... slice            */
/*                            |                           */
/*                            |                           */
/*                            |                           */
/*                            ↓                           */
/*                        encoder_out                      */
/*                           |                           */
/*                           |                           */
/*                           |                           */
/*                         slice                           */
/*                           |                           */
/*                           |                           */
/*                           |                           */
/*                         split                           */
/*----------------------------------------------------   */

class XPUSingleSliceFuser : public FuseBase {
 public:
  void BuildPattern() override {
    auto* input = VarNode("input")
                      ->assert_is_op_output("__xpu__multi_encoder", "Output")
                      ->assert_is_op_input("slice", "Input")
                      ->AsInput();
    auto* slice =
        OpNode("slice", "slice")
            ->assert_op_attr_satisfied<std::vector<int>>(
                "axes",
                [](const std::vector<int>& attr) {
                  return attr.size() == 1 && attr[0] == 1;
                })
            ->assert_op_attr_satisfied<std::vector<int>>(
                "starts",
                [](const std::vector<int>& attr) { return attr.size() == 1; })
            ->assert_op_attr_satisfied<std::vector<int>>(
                "ends",
                [](const std::vector<int>& attr) { return attr.size() == 1; })
            ->assert_op_two_attr_satisfied<std::vector<int>>(
                "starts",
                "ends",
                [](const std::vector<int>& attr_1, const std::vector<int>& attr_2) 
                  { return attr_2[0] - attr_1[0] == 1;})
            ->AsIntermediate();
    auto* slice_out = VarNode("slice_out")
                          ->assert_is_op_output("slice", "Out")
                          ->assert_only_one_output()
                          ->AsOutput();
                          // ->AsIntermediate();
    // auto* softmax = OpNode("softmax", "softmax")
    //                     ->assert_op_attr<int>("axis", -1)
    //                     ->AsIntermediate();
    // auto* softmax_out = VarNode("softmax_out")
    //                         ->assert_is_op_output("softmax", "Out")
    //                         ->AsOutput();

    // *input >> *slice >> *slice_out >> *softmax >> *softmax_out;
    *input >> *slice >> *slice_out ;
  }
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    auto* slice_instruct = matched.at("slice")->stmt();
    auto slice_op_desc = *slice_instruct->op_info();
    auto slice_op = matched.at("slice")->stmt()->op();
    auto* scope = slice_op->scope();

    cpp::OpDesc op_desc;
    op_desc.SetType("__xpu__multi_slice");
    auto input_name = matched.at("input")->arg()->name;
    op_desc.SetInput("Input", {input_name});
    op_desc.SetOutput("Output", {matched.at("slice_out")->arg()->name});
    // std::vector<int> lod{slice_op_desc.GetAttr<std::vector<int>>("starts")[0],
    //                      slice_op_desc.GetAttr<std::vector<int>>("ends")[0]};
    op_desc.SetAttr<std::vector<int>>("starts", slice_op_desc.GetAttr<std::vector<int>>("starts"));
    op_desc.SetAttr<std::vector<int>>("ends", slice_op_desc.GetAttr<std::vector<int>>("ends"));
    op_desc.SetAttr<std::vector<int>>("axes", slice_op_desc.GetAttr<std::vector<int>>("axes"));
    op_desc.SetAttr<int>("slice_cnt", 1);

    // auto multi_softmax_op =
    //     LiteOpRegistry::Global().Create("__xpu__multi_softmax");
    // auto& valid_places = slice_op->valid_places();
    // multi_softmax_op->Attach(op_desc, scope);
    // auto* new_op_node =
    //     graph->GraphCreateInstructNode(multi_softmax_op, valid_places);
    
    auto multi_slice_op =
        LiteOpRegistry::Global().Create("__xpu__multi_slice");
    auto& valid_places = slice_op->valid_places();
    multi_slice_op->Attach(op_desc, scope);
    auto* new_op_node =
        graph->GraphCreateInstructNode(multi_slice_op, valid_places);

    // DirectedLink(matched.at("input"), new_op_node);
    // DirectedLink(new_op_node, matched.at("softmax_out"));

    DirectedLink(matched.at("input"), new_op_node);
    DirectedLink(new_op_node, matched.at("slice_out"));
  }
};

class XPUMultiSliceFusePass : public FuseBase {
 public:
  explicit XPUMultiSliceFusePass(int slice_cnt = 1)
      : slice_cnt_(slice_cnt) {}
  void BuildPattern() override {
    auto* input = VarNode("input")
                      ->assert_is_op_output("__xpu__multi_encoder", "Output")
                      ->assert_is_op_input("slice", "Input")
                      ->AsInput();
    
    std::vector<PMNode*> slice_vec;
    std::vector<PMNode*> slice_out_vec;
    for (int i = 0; i < slice_cnt_; ++i) {
      std::string slice_op_name = "slice_" + std::to_string(i);
      std::string slice_out_name = "slice_out_" + std::to_string(i);
      auto* slice =
        OpNode(slice_op_name, "slice")
            ->assert_op_attr_satisfied<std::vector<int>>(
                "axes",
                [](const std::vector<int>& attr) {
                  return attr.size() == 1 && attr[0] == 1;
                })
            ->assert_op_attr_satisfied<std::vector<int>>(
                "starts",
                [](const std::vector<int>& attr) { return attr.size() == 1; })
            ->assert_op_attr_satisfied<std::vector<int>>(
                "ends",
                [](const std::vector<int>& attr) { return attr.size() == 1; })
            ->assert_op_attr_satisfied_specified_value<std::vector<int>, int>(
                "starts",
                i,
                [](const std::vector<int>& attr, const int val) { return attr[0] == i; })
            assert_op_attr_satisfied_specified_value<std::vector<int>, int>(
                "ends",
                i,
                [](const std::vector<int>& attr, const int val) { return attr[0] == i+1; })
            ->AsIntermediate();

        auto* slice_out = VarNode(slice_out_name)
                          ->assert_is_op_output("slice", "Out")
                          ->assert_only_one_output()
                          ->AsOutput();
        slice_vec.push_back(slice);
        slice_out_vec.push_back(slice_out)
    }

    for (int i = 0; i < slice_cnt_; ++i) {
       input >> *slice_vec[i] >> *slice_out_vec[i];
    }
   
    // auto* slice =
    //     OpNode("slice", "slice")
    //         ->assert_op_attr_satisfied<std::vector<int>>(
    //             "axes",
    //             [](const std::vector<int>& attr) {
    //               return attr.size() == 1 && attr[0] == 1;
    //             })
    //         ->assert_op_attr_satisfied<std::vector<int>>(
    //             "starts",
    //             [](const std::vector<int>& attr) { return attr.size() == 1; })
    //         ->assert_op_attr_satisfied<std::vector<int>>(
    //             "ends",
    //             [](const std::vector<int>& attr) { return attr.size() == 1; })
    //         ->assert_op_two_attr_satisfied<std::vector<int>>(
    //             "starts",
    //             "ends",
    //             [](const std::vector<int>& attr_1, const std::vector<int>& attr_2) 
    //               { return attr_2[0] - attr_1[0] == 1;})
    //         ->AsIntermediate();
    // auto* slice_out = VarNode("slice_out")
    //                       ->assert_is_op_output("slice", "Out")
    //                       ->assert_only_one_output()
    //                       ->AsOutput();
                          // ->AsIntermediate();
    // auto* softmax = OpNode("softmax", "softmax")
    //                     ->assert_op_attr<int>("axis", -1)
    //                     ->AsIntermediate();
    // auto* softmax_out = VarNode("softmax_out")
    //                         ->assert_is_op_output("softmax", "Out")
    //                         ->AsOutput();

    // *input >> *slice >> *slice_out >> *softmax >> *softmax_out;
    // *input >> *slice >> *slice_out ;

  }
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    auto* slice_0_instruct = matched.at("slice_0")->stmt();
    auto slice_op_0_desc = *slice_0_instruct->op_info();
    auto slice_op_0 = matched.at("slice_0")->stmt()->op();
    auto* scope = slice_op_0->scope();

    cpp::OpDesc slice_op_desc;
    slice_op_desc.SetType("slice");
    auto input_name = matched.at("input")->arg()->name;
    slice_op_desc.SetInput("X", {input_name});

    std::string slice_0_out_name = matched.at("slice_0")->arg()->name;
    std::string gathered_slice_node_name = slice_0_out_name + "gathered_slice_out";
    auto* gathered_slice_node =
        graph->NewArgumentNode(gathered_slice_node_name);
    gathered_slice_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kXPU), PRECISION(kInt32), DATALAYOUT(kNCHW));
    scope->NewTensor(gathered_slice_node_name);

    slice_op_desc.SetOutput("Out", {gathered_slice_node_name});
    slice_op_desc.SetAttr<std::vector<int>>("starts", {0});
    slice_op_desc.SetAttr<std::vector<int>>("ends", {slice_cnt_});
    slice_op_desc.SetAttr<std::vector<int>>("axes", {1});

    auto new_slice_op =
        LiteOpRegistry::Global().Create("slice");
    auto& valid_places = slice_op_0->valid_places();
    new_slice_op->Attach(slice_op_desc, scope);
    auto* new_slice_op_node =
        graph->GraphCreateInstructNode(new_slice_op, valid_places);

    

    cpp::OpDesc split_op_desc;
    split_op_desc.SetType("split");
    split_op_desc.SetInput("X", {gathered_slice_node_name});

    std::vector<std::string> split_out_names;
    std::vector<Node*> output_node;
    for (int i = 0; i < slice_cnt_; ++i) {
      std::string slice_out_name = "slice_out_" + std::to_string(i);
      std::string slice_op_name = "slice_" + std::to_string(i);
      split_out_names.push_back(matched.at(slice_out_name)->arg()->name);

      output_node.push_back(matched.at(slice_op_name));
    }
    split_op_desc.SetOutput("Output", split_out_names);
    split_op_desc.SetAttr<int>("axes", 1);
    split_op_desc.SetAttr<int>("num", slice_cnt_);


    auto new_split_op =
        LiteOpRegistry::Global().Create("split");
    // auto& valid_places = slice_op_0->valid_places();
    new_split_op->Attach(split_op_desc, scope);
    auto* new_split_op_node =
        graph->GraphCreateInstructNode(new_split_op, valid_places);

    


    DirectedLink(matched.at("input"), new_slice_op_node);
    DirectedLink(new_slice_op_node, gathered_slice_node);
    DirectedLink(gathered_slice_node, new_split_op_node);

    for (Node* node : output_node) {
      DirectedLink(new_split_op_node, node);
    }
    // op_desc.SetOutput("Output", {matched.at("slice_out")->arg()->name});
    // cpp::OpDesc op_desc;
    // op_desc.SetType("__xpu__multi_slice");
    // auto input_name = matched.at("input")->arg()->name;
    // op_desc.SetInput("Input", {input_name});
    // op_desc.SetOutput("Output", {matched.at("slice_out")->arg()->name});
    // // std::vector<int> lod{slice_op_desc.GetAttr<std::vector<int>>("starts")[0],
    // //                      slice_op_desc.GetAttr<std::vector<int>>("ends")[0]};
    // op_desc.SetAttr<std::vector<int>>("starts", slice_op_desc.GetAttr<std::vector<int>>("starts"));
    // op_desc.SetAttr<std::vector<int>>("ends", slice_op_desc.GetAttr<std::vector<int>>("ends"));
    // op_desc.SetAttr<std::vector<int>>("axes", slice_op_desc.GetAttr<std::vector<int>>("axes"));
    // op_desc.SetAttr<int>("slice_cnt", slice_cnt_);

    // auto multi_softmax_op =
    //     LiteOpRegistry::Global().Create("__xpu__multi_softmax");
    // auto& valid_places = slice_op->valid_places();
    // multi_softmax_op->Attach(op_desc, scope);
    // auto* new_op_node =
    //     graph->GraphCreateInstructNode(multi_softmax_op, valid_places);
    
    // auto multi_slice_op =
    //     LiteOpRegistry::Global().Create("__xpu__multi_slice");
    // auto& valid_places = slice_op->valid_places();
    // multi_slice_op->Attach(op_desc, scope);
    // auto* new_op_node =
    //     graph->GraphCreateInstructNode(multi_slice_op, valid_places);

    // DirectedLink(matched.at("input"), new_op_node);
    // DirectedLink(new_op_node, matched.at("softmax_out"));

    // DirectedLink(matched.at("input"), new_op_node);
    // DirectedLink(new_op_node, matched.at("slice_out"));
  }
private:
  bool slice_cnt_;
};



}  // namespace fusion

class XPUMultiSliceFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    // fusion::XPUSingleSliceFuser single;
    // single(graph.get());
    // fusion::XPUMultiSliceFuser multi_fuser;
    // multi_fuser(graph.get());
    std::vector<int> slice_cnts;
    for (int i = 10; i > 0; ++i) {
      slice_cnts.push_back(i);
    }
    for (auto slice_cnt : slice_cnts) {
      fusion::XPUMultiSliceFuser fuser();
      fuser(graph.get());
    }
    // fusion::XPUMultiSliceFuser fuser();
    // fuser(graph.get());
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__multi_slice_fuse_pass,
                  paddle::lite::mir::XPUMultiSliceFusePass)
    .BindTargets({TARGET(kXPU)})
