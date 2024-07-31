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

class XPUSliceSplitFuser : public FuseBase {
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

  // 将slice的节点选择出来
  void SelectNode(SSAGraph* graph, std::vector<Node*> all_slice_split) {
    std::vector<Node*> all_slice;
    std::vector<const Node*> to_remove;
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
      int max_num = 1;
      for (int j = 0; j !=i && j < all_slice.size(); ++j) {
        if (IsSamePredecessorOf(all_slice[i], all_slice[j])) {
          max_num++; 
        }
        if (max_num > global_max) {

          global_max = max_num;
        }
        pred_num[i] = max_num;
      }

      for (int i = 0; i < all_slice.size(); ++i) {
        if (pred_num[i] == global_max) {
          all_slice_split.push_back(all_slice[i]);
        }
      }
    }
    VLOG(3) << "Found slice num: " << all_slice_split.size();

    if (all_slice_split.size() == 0) {
      return;
    }
  }

  // 找到相同的inlinks, 判断是否是同一个父亲节点 把最多父节点的上去
  void operator() (SSAGraph* graph) {
    std::vector<Node*> all_slice_split;
    std::vector<Node*> slice_split_sort;
    std::vector<int> vec_start;
    std::vector<int> vec_end;
    // 挑选节点 但是还没有排序
    SelectNode(graph, all_slice_split);
    // 按照starts排序
    for (int i = 0; i < all_slice_split.size(); i++) {
      for (int j = 0; j < all_slice_split.size(); j++) {
        if (all_split_slice[i]->stmt()->op_info()->GetAttr<std::vector<int>>("starts")[0] == i) {
          slice_split_sort.push_back(all_split_slice[i]);
          vec_start.push_back(i);
          vec_end.push_back(i+1);
        }
      }
    }

    auto first_slice = slice_split_sort[0]->stmt()->op();
    auto* scope = first_slice->scope();
    auto& valid_places = first_slice->valid_places();

    std::vector<bool> used(all_split_slice.size(), false);
    std::vector<std::string> out_names;
    Node* input_node = all_split_slice[0]->inlinks.front();
    std::vector<Node*> output_node;
    // 记录名字
    std::string in_name = 
        all_split_slice[0]->stmt()->op_info()->Input("X").front();
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
            all_split_slice[i]->stmt()->op_info()->GetAttr<std::vector<int>>("ends")[0];
          auto cur_lod_start =
            all_split_slice[i]->stmt()->op_info()->GetAttr<std::vector<int>>("starts")[0];
          if (cur_lod_start == i && cur_lod_end == (i+1)) {
            out_names.push_back(
              slice_split_sort[i]->stmt()->op_info()->Output("Out").front());
            output_node.push_back(slice_split_sort[i]->outlinks.front());
            used[i] = true;
            cur_remain = cur_remain - 1;
          }
        }
        all_used = all_used & used[i];
      }
    }

    GraphSafeRemoveNodes(graph, to_remove);
    cpp::OpDesc op_desc;
    op_desc.SetType("__xpu__slice_split");
    op_desc.SetInput("X", {in_name});
    op_desc.SetOutput("Out", {out_names});
    // 需要将axis_tensor进行填充 给到split
    op_desc.SetAttr<std::vector<int>>("starts", vec_start);
    op_desc.SetAttr<std::vector<int>>("ends", vec_end);

    // 需要将数据进行拼接
    std::string concat_output_name =
      "__xpu__slice_split_concat_output_" + in_name;
    CHECK(graph->RetrieveArgument(concat_output_name) == nullptr);
    auto* concat_output_node = graph->NewArgumentNode(concat_output_name);
    concat_output_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kXPU), PRECISION(kFloat));
    auto* new_op = LiteOpRegistry::Global().Create(op_desc.Type());
    scope->NewTensor(concat_output_name);

    new_op->Attach(op_desc, scope);

    // 需要将进行拼接
    auto* new_op_node = graph->GraphCreateInstructNode(new_op, valid_places);
    DirectedLink(new_op_node, concat_output_node);
    DirectedLink(input_node, new_op_node);
    for (Node* node : output_node) {
      DirectedLink(new_op_node, node);
    }
  }
};

}  // namespace fusion

class XPUSliceSplitFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    fusion::XPUSliceSplitFuser slice_split_fuser;
    slice_split_fuser(graph.get());
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__multi_softmax_fuse_pass,
                  paddle::lite::mir::XPUSliceSplitFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__slice_split");


