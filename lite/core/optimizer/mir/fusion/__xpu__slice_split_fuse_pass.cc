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

    auto first_slice = slice_split_sort[0]->stmt()->op();
    auto* scope = first_slice->scope();
    auto& valid_places = first_slice->valid_places();
    std::vector<std::string> out_names;
    std::vector<std::string> in_names;
    Node* input_node = slice_split_sort[0]->inlinks.front();
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
    std::string concat_output_name =
      "__xpu__slice_concat_output";
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
    auto new_op = LiteOpRegistry::Global().Create(op_desc.Type());
    new_op->Attach(op_desc, scope);
    auto* new_op_node = graph->GraphCreateInstructNode(new_op, valid_places);

    std::string slice4_name = "slice_4.tmp_0";
    auto* slice4_name_node = graph->NewArgumentNode(slice4_name);
    slice4_name_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kXPU), PRECISION(kFloat));
    scope->NewTensor(slice4_name);
    std::cout << "----148---This is a test for this pass" << std::endl;

    // split新增代码
    // 定义名字和输出结点名字
    std::vector<Node*> split_output_node;
    std::vector<std::string> split_out_node_names;
    // 输出结点名字
    for (int i = 0; i < out_names.size(); ++i) {
      split_out_node_names.push_back("__xpu__split_" + out_names[i]);
    }
    auto first_split = new_op_node->stmt()->op();
    auto* scope_split = first_split->scope();
    cpp::OpDesc split_op_desc;
    split_op_desc.SetType("split");
    split_op_desc.SetInput("Input", {concat_output_name});
    split_op_desc.SetAttr<int>("axis", 1);
    split_op_desc.SetAttr<int>("num", 6);
    // std::string split_output_name = "__xpu__split_output";
    split_op_desc.SetOutput("Out", {split_out_node_names});

    // 目前还是有些问题，需要修改 split的参数如下，不确定上面的参数是否正确
// struct SplitParam : ParamBase {
//   const lite::Tensor* x{nullptr};
//   std::vector<lite::Tensor*> output{};
//   const lite::Tensor* axis_tensor{nullptr};
//   std::vector<lite::Tensor*> sections_tensor_list{};

//   int axis{-1};
//   int num{0};
//   std::vector<int> sections;
// };

    auto split_new_op = LiteOpRegistry::Global().Create(split_op_desc.Type());
    std::cout << "----172---This is a test for this pass" << std::endl;
    split_new_op->Attach(split_op_desc, scope_split);
    std::cout << "----174---This is a test for this pass" << std::endl;
    auto* split_node = graph->GraphCreateInstructNode(split_new_op, valid_places);

    for (int i = 0; i < split_out_node_names.size(); i++) {
      std::cout << "out_name:" << split_out_node_names[i] << std::endl;
      auto* out_name_node = graph->NewArgumentNode(split_out_node_names[i]);
      out_name_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kXPU), PRECISION(kFloat));
      // out_name_node->Attach(split_op_desc, scope_split);
      
      scope_split->NewTensor(split_out_node_names[i]);
      split_output_node.push_back(out_name_node);
    }
    std::cout << "----168---This is a test for this pass" << std::endl;



    DirectedLink(input_node, new_op_node);
    DirectedLink(new_op_node, concat_output_node);
    DirectedLink(concat_output_node, split_node);

    for (auto& node : new_op_node->outlinks) {
      std::cout << "----167--------:" << std::endl;
      std::cout << "new_op_node:" << *node << std::endl;
    }

    for (auto& node : split_node->outlinks) {
      std::cout << "----172--------:" << std::endl;
      std::cout << "concat_output_node:" << *node << std::endl;
    }

    for (auto& node : concat_output_node->outlinks) {
      std::cout << "----177--------:" << std::endl;
      std::cout << "split_node:" << *node << std::endl;
    }


    std::cout << "----157---This is a test for this pass" << std::endl;
    // 链接
    int num = 0;
    for (Node* node : output_node) {
      std::cout << "num:" << num << std::endl;
      DirectedLink(split_node, node);
      num++;
    }

    std::cout << "----166----:" << std::endl;
    for (auto& node : input_node->outlinks) {
      std::cout << "----168--------:" << std::endl;
      std::cout << "node:" << node->stmt()->op_type() << std::endl;
      // std::cout << "node input:" << node->stmt()->op_info()->Input("Input").front() << std::endl;
      // std::cout << "node output:" << node->stmt()->op_info()->Output("Out").front() << std::endl;
    }

    for (auto& node : input_node->inlinks) {
      std::cout << "----175--------:" << std::endl;
      std::cout << "in node:" << node->stmt()->op_type() << std::endl;
    }
    Node* slice_node_tmp = input_node->outlinks.front();
    std::cout << "slice_node_tmp:" << *slice_node_tmp << std::endl;
    for (auto& node : slice_node_tmp->outlinks) {
      std::cout << "----190--------:" << std::endl;
      std::cout << "slice_node_tmp:" << *node << std::endl;
    }

    Node* concat_node_tmp = slice_node_tmp->outlinks.front();
    std::cout << "concat node:" << *concat_node_tmp << std::endl;
    for (auto& node : concat_node_tmp->outlinks) {
      std::cout << "----196--------:" << std::endl;
      std::cout << "concat_node_tmp:" << *node << std::endl;
    }

    Node* split_node_tmp = concat_node_tmp->outlinks.front();
    std::cout << "split node:" << *split_node_tmp << std::endl;
    for (auto& node : split_node_tmp->outlinks) {
      std::cout << "----203--------:" << std::endl;
      std::cout << "split_node_tmp node:" << *node << std::endl;
    }
    // Node* mul_enc_node = input_node->inlinks.front();
    // // auto& mul_enc_node = input_node->inlinks;
    // for (auto& node : mul_enc_node->outlinks) {
    //   std::cout << "----193--------:" << std::endl;
    //   std::cout << "mul enc outlink node:" << *node << std::endl;
    // }
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