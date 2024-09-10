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

/* fuse multi slice with slice as __xpu__slice_split     */
/* graph                                                 */
/*                                                       */
/*                         in_Input                      */
/*                    /   /  |  \    \                   */
/*          [slice   slice  slice   ... slice]           */
/*             |       |     |            |              */
/*                        split                          */
/*             |       |     |            |              */
/*          slice   slice  slice   ... slice             */
/*----------------------------------------------------   */

class XPUMultiSliceSplitFuser {
 public:
  bool IsSamePredecessorOf(Node* op1, Node* op2) {
    for (auto* in1 : op1->inlinks) {
      for (auto* in2 : op2->inlinks) {
        if (in1 != in2) return false;
      }
    }
    return true;
  }
  void operator() (SSAGraph* graph) {
    std::vector<Node*> all_slice;
    std::vector<int> pred_num;;
    for (auto* node : graph->StmtTopologicalOrder()) {
      CHECK(node->IsStmt());
      if (node->stmt()->op_info()->Type() == "slice") {
        all_slice.push_back(node);
      }
    }

    std::vector<Node*> slice_split_sort;
    std::set<const Node*> to_remove;
    std::vector<bool> used(all_slice.size(), false);
    int start_num = 0;
    while (start_num < 4) {
      for (int i= 0; i < used.size(); i++) {
        if (used[i] == false) {
          if (all_slice[i]->stmt()->op_info()->GetAttr<std::vector<int>>("starts")[0] == start_num) {
            slice_split_sort.push_back(all_slice[i]);
            std::cout << "starts:" << 
            all_slice[i]->stmt()->op_info()->GetAttr<std::vector<int>>("starts")[0] << std::endl;
            used[i] = true;
            start_num++;
          } else {
            continue;
          }
        } else {
          continue;
        }
      }
    }

    for (int i = 0; i < all_slice.size(); ++i) {
      if (all_slice[i]->stmt()->op_info()->GetAttr<std::vector<int>>("starts")[0] == 5) {
        slice_split_sort.push_back(all_slice[i]);
        std::cout << "starts:" << 
          all_slice[i]->stmt()->op_info()->GetAttr<std::vector<int>>("starts")[0] << std::endl;
      }
    }
    auto first_slice = slice_split_sort[0]->stmt()->op(); // 第一个slice op算子
    auto* scope = first_slice->scope();
    auto& valid_places = first_slice->valid_places();
    std::vector<std::string> out_names;
    std::vector<std::string> in_names;
    Node* input_node = slice_split_sort[0]->inlinks.front(); // inlink是结点信息  .front()是数据信息的第一个元素
    std::vector<Node*> output_node;
    std::string in_name = 
        slice_split_sort[0]->stmt()->op_info()->Input("Input").front();
    std::vector<int> end{0};
    for (int i = 0; i < slice_split_sort.size(); i++) {
      in_names.push_back(slice_split_sort[i]->stmt()->op_info()->Input("Input").front());
      out_names.push_back(slice_split_sort[i]->stmt()->op_info()->Output("Out").front());
      // we don't have slice_4.tmp_0, so we add it manually
      if(i == 3) {
        out_names.push_back("slice_4.tmp_0");
      }
      output_node.push_back(slice_split_sort[i]->outlinks.front());
      to_remove.insert(slice_split_sort[i]);
    }
    GraphSafeRemoveNodes(graph, to_remove);
    // slice
    cpp::OpDesc op_desc;
    std::string concat_output_name = "__xpu__slice_concat_output"; // 构建一个slice 然后拿concat承接所有slice的结果
    op_desc.SetType("slice");
    op_desc.SetInput("Input", {in_name});
    // op_desc.SetOutput("Out", {concat_output_name});
    op_desc.SetAttr<std::vector<int>>("axes", {1});
    op_desc.SetAttr<std::vector<int>>("starts", {0});
    op_desc.SetAttr<std::vector<int>>("ends", {6});
    CHECK(graph->RetrieveArgument(concat_output_name) == nullptr);
    auto* concat_output_node = graph->NewArgumentNode(concat_output_name);
    concat_output_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kXPU), PRECISION(kFloat));
    scope->NewTensor(concat_output_name);
    op_desc.SetOutput("Out", {concat_output_name});
    // 构建op和节点
    auto slice_new_op = LiteOpRegistry::Global().Create(op_desc.Type());
    slice_new_op->Attach(op_desc, scope);
    auto* slice_new_op_node = graph->GraphCreateInstructNode(slice_new_op, valid_places);

    std::string slice4_name = "slice_4.tmp_0"; // 这个是新建的Tensor 这个暂且不表
    auto* slice4_name_node = graph->NewArgumentNode(slice4_name);
    slice4_name_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kXPU), PRECISION(kFloat));
    scope->NewTensor(slice4_name);

    // concat没有对应的outlinks
    auto first_split = slice_new_op_node->stmt()->op();
    auto* scope_split = first_split->scope();
    cpp::OpDesc split_op_desc;
    split_op_desc.SetType("split");
    split_op_desc.SetInput("x", {concat_output_name});
    split_op_desc.SetAttr<int>("axis", {1});
    split_op_desc.SetAttr<int>("num", {6});
    for (int i = 0; i < out_names.size(); i++) {
      if (i == 4) {
        continue;
      } else {
        std::cout << "out_name:" << out_names[i] << std::endl;
        auto* out_name_node = graph->NewArgumentNode(out_names[i]);
        out_name_node->arg()->type = LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat));
        scope_split->NewTensor(out_names[i]);
      }
    }
    split_op_desc.SetOutput("output", {out_names});
    std::cout << "----168---This is a test for this pass" << std::endl;

    DirectedLink(input_node, slice_new_op_node);
    DirectedLink(slice_new_op_node, concat_output_node);
    for (auto& node : output_node) {
      DirectedLink(slice_new_op_node, node);
    }
  }
};

}  // namespace fusion

class XPUSliceSplitFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    fusion::XPUMultiSliceSplitFuser slice_split_fuser;
    slice_split_fuser(graph.get());
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__slice_split_fuse_pass,
                  paddle::lite::mir::XPUSliceSplitFusePass)
    .BindTargets({TARGET(kXPU)});