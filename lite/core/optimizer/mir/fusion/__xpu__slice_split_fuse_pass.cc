// 后面考虑一下multi_slice pass可以实现成 slice + split op
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

/******构造单pass******/
// class XPUSingleSliceSplitFuser : public FuseBase {
//  public:
//   void BuildPattern() override {
//     auto* input = VarNode("input")
//                       ->assert_is_op_output("__xpu__multi_encoder", "Output")
//                       ->assert_is_op_input("slice", "Input")
//                       ->AsInput();
//     auto* slice =
//         OpNode("slice", "slice")
//             ->assert_op_attr_satisfied<std::vector<int>>(
//                 "axes",
//                 [](const std::vector<int>& attr) {
//                   return attr.size() == 1 && attr[0] == 1;
//                 })
//             ->assert_op_attr_satisfied<std::vector<int>>(
//                 "starts",
//                 [](const std::vector<int>& attr) { return attr.size() == 1; })
//             ->assert_op_attr_satisfied<std::vector<int>>(
//                 "ends",
//                 [](const std::vector<int>& attr) { return attr.size() == 1; })
//             ->AsIntermediate();
//     auto* slice_out = VarNode("slice_out")
//                           ->assert_is_op_output("slice", "Out")
//                           ->assert_is_op_input("softmax", "X")
//                           ->assert_only_one_output()
//                           ->AsIntermediate();
//     // 这里可能需要重新确定一下
//     auto* split = OpNode("split", "split")
//                         ->assert_op_attr<int>("axis", 1)
//                         ->assert_op_attr<int>("num", 1)
//                         ->AsIntermediate();
//     auto* split_output = VarNode("split_out_out")
//                             ->assert_is_op_output("split", "Out")
//                             ->AsOutput();
//     *input >> *slice >> *slice_out >> *split >> *split_output;
//   }

//   void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
//     auto* slice_instruct = matched.at("slice")->stmt();
//     auto slice_op_desc = *slice_instruct->op_info();
//     auto slice_op = matched.at("slice")->stmt()->op();
//     auto* scope = slice_op->scope();

//     cpp::OpDesc op_desc;
//     op_desc.SetType("__xpu__slice_split");
//     auto input_name = matched.at("input")->arg()->name;
//     op_desc.SetInput("Input", {input_name});
//     op_desc.SetOutput("Output", {matched.at("split_out")->arg()->name});
//     op_desc.SetAttr("axis", 1);
//     op_desc.SetAttr("num", 1);
//     auto multi_slice_split_op =
//         LiteOpRegistry::Global().Create("__xpu__slice_split");
//     auto& valid_places = slice_op->valid_places();
//     multi_slice_split_op->Attach(op_desc, scope);
//     auto* new_op_node =
//         graph->GraphCreateInstructNode(multi_slice_split_op, valid_places);
//     DirectedLink(matched.at("input"), new_op_node);
//     DirectedLink(new_op_node, matched.at("slice_out"));
//   }
// };

class XPUMultiSliceSplitFuser {
 public:
  // 这些slices的输入都是同一个,判断是否是同一个父亲节点
  bool IsSamePredecessorOf(Node* op1, Node* op2) {
    for (auto* in1 : op1->inlinks) {
      for (auto* in2 : op2->inlinks) {
        if (in1 != in2) return false;
      }
    }
    return true;
  }

  // 找到相同的inlinks, 判断是否是同一个父亲节点 把最多父节点的上去
  void operator() (SSAGraph* graph) {
    std::vector<Node*> all_slice;
    std::vector<int> pred_num;
    std::vector<Node*> all_slice_split;
    // 找到所有的slice算子节点
    for (auto* node : graph->StmtTopologicalOrder()) {
      CHECK(node->IsStmt());
      // 寻找有没有符合要求的算子或者pass
      if (node->stmt()->op_info()->Type() == "slice") {
        // 先把所有的slice算子拿到
        all_slice.push_back(node);
        // 然后去找到inlinks相同且最多的node 将其筛选出来
      }
    }

    int global_max = 1;
    // 然后把in_links最高的相同的筛选出
    for (int i = 0; i < all_slice.size(); ++i) {
      int same_num = 1;
      for (int j = 0; j < all_slice.size(); ++j) {
        if (i == j) {
          continue;
        } else {
          if (IsSamePredecessorOf(all_slice[i], all_slice[j])) {
            same_num++; 
          }
        }
      }
      if (same_num > global_max) {
        global_max = same_num;
      }
      pred_num.push_back(same_num);
    }
    for (int i = 0; i < all_slice.size(); ++i) {
      if (pred_num[i] == global_max) {
        all_slice_split.push_back(all_slice[i]);
      }
    }
    VLOG(3) << "Found slice num: " << all_slice_split.size();

    std::vector<Node*> slice_split_sort;
    std::set<const Node*> to_remove;
    std::vector<int> vec_start;
    std::vector<int> vec_end;
    // 按照starts排序
    for (int i = 0; i < all_slice_split.size(); i++) {
      for (int j = 0; j < all_slice_split.size(); j++) {
        if (all_slice_split[i]->stmt()->op_info()->GetAttr<std::vector<int>>("starts")[0] == i) {
          slice_split_sort.push_back(all_slice_split[i]);
          vec_start.push_back(i);
          vec_end.push_back(i+1);
        }
      }
    }
    // 按照start和end完成排序了
    auto first_slice = slice_split_sort[0]->stmt()->op();
    auto* scope = first_slice->scope();
    auto& valid_places = first_slice->valid_places();

    std::vector<bool> used(slice_split_sort.size(), false);
    std::vector<std::string> out_names;
    Node* input_node = slice_split_sort[0]->inlinks.front();
    std::vector<Node*> output_node;
    // 记录名字
    std::string in_name = 
        slice_split_sort[0]->stmt()->op_info()->Input("X").front();
    std::vector<int> end{0};
    bool all_used = false;
    int last_remain = used.size();
    // 记录一下还有多少个节点
    int cur_remain = last_remain;
    while (all_used == false) {
      all_used = true;
      last_remain = cur_remain;
      for (int i = 0; i < used.size(); i++) {
        if (used[i] == false) {
          auto cur_lod_end =
            slice_split_sort[i]->stmt()->op_info()->GetAttr<std::vector<int>>("ends")[0];
          auto cur_lod_start =
            slice_split_sort[i]->stmt()->op_info()->GetAttr<std::vector<int>>("starts")[0];
          if (cur_lod_start == i && cur_lod_end == (i+1)) {
            out_names.push_back(
              slice_split_sort[i]->stmt()->op_info()->Output("Out").front());
            output_node.push_back(slice_split_sort[i]->outlinks.front());
            to_remove.insert(slice_split_sort[i]);
            used[i] = true;
            cur_remain = cur_remain - 1;
          }
        }
        all_used = all_used & used[i];
      }
    }

    GraphSafeRemoveNodes(graph, to_remove); // 将节点删除
    //  这里构建slice节点的数据，将结果拼接，作为split的输入
    cpp::OpDesc op_desc;
    op_desc.SetType("slice");
    op_desc.SetInput("X", {in_name});
    // 这里应该输出的是一个节点
    std::string out_name = "mul_encoder_slice_out";
    op_desc.SetOutput("Out", {out_name});
    op_desc.SetAttr<std::vector<int>>("starts", vec_start);
    op_desc.SetAttr<std::vector<int>>("ends", vec_end);
    // 需要将数据进行拼接
    std::string concat_output_name =
      "__xpu__slice_concat_output_" + in_name;
    
    CHECK(graph->RetrieveArgument(concat_output_name) == nullptr);
    auto* concat_output_node = graph->NewArgumentNode(concat_output_name);
    concat_output_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kXPU), PRECISION(kFloat));
    auto new_op = LiteOpRegistry::Global().Create(op_desc.Type());
    scope->NewTensor(concat_output_name);
    new_op->Attach(op_desc, scope);
    // 需要将进行拼接
    auto* new_op_node = graph->GraphCreateInstructNode(new_op, valid_places);
    // 上面是构建完成了slice op; 下面构建split op
    cpp::OpDesc split_op_desc;
    split_op_desc.SetType("split");
    std::string slice_output_name =
      "__xpu__slice_output_" + in_name;
    split_op_desc.SetInput("X", {slice_output_name});
    split_op_desc.SetOutput("Out", {out_names});
    split_op_desc.SetAttr<int>("axis", 1);
    split_op_desc.SetAttr<int>("num", slice_split_sort.size());

    CHECK(graph->RetrieveArgument(slice_output_name) == nullptr);
    auto* slice_output_node = graph->NewArgumentNode(slice_output_name);
    slice_output_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kXPU));
    auto split_new_op = LiteOpRegistry::Global().Create(split_op_desc.Type());
    auto* split_node = graph->GraphCreateInstructNode(split_new_op, valid_places);

    DirectedLink(input_node, new_op_node);
    DirectedLink(new_op_node, split_node);
    for (Node* node : output_node) {
      DirectedLink(split_node, node);
    }
  }
};

}  // namespace fusion

class XPUSliceSplitFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    // fusion::XPUSingleSliceSplitFuser single;
    // single(graph.get());
    fusion::XPUMultiSliceSplitFuser slice_split_fuser;
    slice_split_fuser(graph.get());
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__slice_split_fuse_pass,
                  paddle::lite::mir::XPUSliceSplitFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__slice_split");


