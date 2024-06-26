import json
import os
import random
from collections import namedtuple

import dgl
import torch as th
from dgl.data import DGLDataset
from dgl.data.utils import load_graphs, save_graphs

binOps_training_data_paths = "./binOps_training_150.json"
binOps_validation_data_paths = "./binOps_eval_150.json"
calls_training_data_paths = "./calls_training_150.json"
calls_validation_data_paths = "./calls_eval_150.json"


token_vectors = None
type_vectors = None
node_type_vectors = None

binOps_training = None
binOps_eval = None
calls_training = None

with open("./token_to_vector_150.json", encoding="utf-8") as f:
    token_vectors = json.load(f)

with open("./type_to_vector.json", encoding="utf-8") as f:
    type_vectors = json.load(f)

with open("./node_type_to_vector.json", encoding="utf-8") as f:
    node_type_vectors = json.load(f)

with open(binOps_validation_data_paths, encoding="utf-8") as f:
    binOps_eval = json.load(f)

with open(calls_validation_data_paths, encoding="utf-8") as f:
    calls_eval = json.load(f)

with open(binOps_training_data_paths, encoding="utf-8") as f:
    binOps_training = json.load(f) + binOps_eval[398000:]
    binOps_eval = binOps_eval[0:398000]

with open(calls_training_data_paths, encoding="utf-8") as f:
    calls_training = json.load(f) + calls_eval[220000:]
    calls_eval = calls_eval[0:220000]

print("calls_training length", len(calls_training))
print("calls_eval length", len(calls_eval))
print("binOps_training length", len(binOps_training))
print("binOps_eval length", len(binOps_eval))


binOps_graph = ([0, 1, 1, 1, 3, 4, 2, 5], [1, 2, 3, 5, 4, 2, 6, 6])
incorrect_bin_operand_graph = (
    [0, 1, 1, 1, 3, 4, 2, 5, 0, 7],
    [1, 2, 3, 5, 4, 2, 6, 6, 7, 8],
)
calls_graph = ([0, 1, 1, 2, 2, 3, 5, 6, 4], [1, 2, 5, 5, 3, 4, 6, 7, 7])
swapped_calls_graph = ([0, 1, 1, 2, 3, 5, 6, 5, 7], [1, 2, 5, 3, 4, 6, 7, 2, 4])
operator_embedding_size = 30
name_embedding_size = 200
type_embedding_size = 5
Operand = namedtuple("Operand", ["op", "type"])
LABELS = {
    "correct_binary_op": 0,
    "incorrect_binary_operand": 1,
    "incorrect_binary_operator": 2,
    "correct_args": 3,
    "swapped_args": 4,
}


class Dataset(DGLDataset):
    def __init__(self, is_training=True):
        self.file_to_operands = dict()
        self.all_operators = None
        self.graphs = []
        self.labels = []
        self.is_training = is_training
        self.bug_type = "all"

        super().__init__(name="synthetic")

    def pre_scan_binOps(self, first_data_paths, second_data_paths=[]):
        all_operators_set = set()
        for bin_op in first_data_paths:
            file = bin_op["src"].split(" : ")[0]
            operands = self.file_to_operands.setdefault(file, set())
            left_operand = Operand(bin_op["left"], bin_op["leftType"])
            right_operand = Operand(bin_op["right"], bin_op["rightType"])
            operands.add(left_operand)
            operands.add(right_operand)

            all_operators_set.add(bin_op["op"])
        if second_data_paths == []:
            self.all_operators = list(all_operators_set)
            return

        for bin_op in second_data_paths:
            file = bin_op["src"].split(" : ")[0]
            operands = self.file_to_operands.setdefault(file, set())
            left_operand = Operand(bin_op["left"], bin_op["leftType"])
            right_operand = Operand(bin_op["right"], bin_op["rightType"])
            operands.add(left_operand)
            operands.add(right_operand)

            all_operators_set.add(bin_op["op"])
        self.all_operators = list(all_operators_set)

    def generate_random_embedding(self, num_nodes):
        return th.randn(num_nodes, name_embedding_size)

    def get_tensor_feature(self, data):
        max_len = max([x.squeeze().numel() for x in data])
        data = [
            th.nn.functional.pad(
                x, pad=(0, max_len - x.numel()), mode="constant", value=0
            )
            for x in data
        ]
        return th.stack(data)

    def generate_graphs_from_binOps_ast(self):
        num_nodes = 7

        dataset = binOps_training if self.is_training else binOps_eval
        for data in dataset:
            left = data["left"]
            right = data["right"]
            operator = data["op"]
            left_type = data["leftType"]
            right_type = data["rightType"]
            parent = data["parent"]
            grand_parent = data["grandParent"]
            src = data["src"]

            if not (left in token_vectors):
                continue
            if not (right in token_vectors):
                continue

            operator_vector = [0] * operator_embedding_size
            operator_vector[self.all_operators.index(operator)] = 1
            correct_vector = [
                th.tensor(node_type_vectors[grand_parent]),
                th.tensor(node_type_vectors[parent]),
                th.tensor(operator_vector),
                th.tensor(type_vectors[left_type]),
                th.tensor(token_vectors[left]),
                th.tensor(type_vectors[right_type]),
                th.tensor(token_vectors[right]),
            ]

            g = dgl.graph(binOps_graph, num_nodes=len(correct_vector))
            g.ndata["features"] = self.get_tensor_feature(correct_vector)
            self.graphs.append(g)
            self.labels.append(
                LABELS["correct_binary_op"] if self.bug_type == "all" else 0
            )

            if self.bug_type in ["incorrect_binary_operator", "all"]:
                other_operator = None
                other_operator_vector = None

                while other_operator_vector == None:
                    other_operator = random.choice(self.all_operators)
                    if other_operator != operator:
                        other_operator_vector = [0] * operator_embedding_size
                        other_operator_vector[
                            self.all_operators.index(other_operator)
                        ] = 1

                incorrect_bin_ops_vector = [
                    th.tensor(node_type_vectors[grand_parent]),
                    th.tensor(node_type_vectors[parent]),
                    th.tensor(other_operator_vector),
                    th.tensor(type_vectors[left_type]),
                    th.tensor(token_vectors[left]),
                    th.tensor(type_vectors[right_type]),
                    th.tensor(token_vectors[right]),
                ]

                g = dgl.graph(binOps_graph, num_nodes=num_nodes)
                g.ndata["features"] = self.get_tensor_feature(incorrect_bin_ops_vector)
                self.graphs.append(g)
                self.labels.append(
                    LABELS["incorrect_binary_operator"] if self.bug_type == "all" else 1
                )

            if self.bug_type in ["incorrect_binary_operand", "all"]:
                replace_left = random.random() < 0.5
                if replace_left:
                    to_replace_operand = left
                else:
                    to_replace_operand = right
                file = src.split(" : ")[0]
                all_operands = self.file_to_operands[file]
                tries_left = 100
                found = False
                while (not found) and tries_left > 0:
                    other_operand = random.choice(list(all_operands))
                    if (
                        other_operand.op in token_vectors
                        and other_operand.op != to_replace_operand
                    ):
                        found = True
                    tries_left -= 1

                if not found:
                    return

                other_operand_vector = token_vectors[other_operand.op]
                other_operand_type_vector = type_vectors[other_operand.type]

                if replace_left:
                    incorrect_bin_operands_vector = [
                        th.tensor(node_type_vectors[grand_parent]),
                        th.tensor(node_type_vectors[parent]),
                        th.tensor(operator_vector),
                        th.tensor(other_operand_type_vector),
                        th.tensor(other_operand_vector),
                        th.tensor(type_vectors[right_type]),
                        th.tensor(token_vectors[right]),
                    ]
                else:
                    incorrect_bin_operands_vector = [
                        th.tensor(node_type_vectors[grand_parent]),
                        th.tensor(node_type_vectors[parent]),
                        th.tensor(operator_vector),
                        th.tensor(type_vectors[left_type]),
                        th.tensor(token_vectors[left]),
                        th.tensor(other_operand_type_vector),
                        th.tensor(other_operand_vector),
                    ]

                g = dgl.graph(binOps_graph, num_nodes=num_nodes)

                g.ndata["features"] = self.get_tensor_feature(
                    incorrect_bin_operands_vector
                )
                self.graphs.append(g)
                self.labels.append(
                    LABELS["incorrect_binary_operand"] if self.bug_type == "all" else 1
                )

    def generate_graphs_from_calls_ast(self):
        num_nodes = 8

        dataset = calls_training if self.is_training else calls_eval
        for call in dataset:
            arguments = call["arguments"]
            if len(arguments) != 2:
                continue

            callee_string = call["callee"]
            argument_strings = call["arguments"]

            if not (callee_string in token_vectors):
                continue
            not_found = False
            for argument_string in argument_strings:
                if not (argument_string in token_vectors):
                    not_found = True
            if not_found:
                continue

            callee_vector = token_vectors[callee_string]
            argument0_vector = token_vectors[argument_strings[0]]
            argument1_vector = token_vectors[argument_strings[1]]

            base_string = call["base"]
            base_vector = token_vectors.get(base_string, [0] * name_embedding_size)

            argument_type_strings = call["argumentTypes"]
            argument0_type_vector = type_vectors.get(
                argument_type_strings[0], [0] * type_embedding_size
            )
            argument1_type_vector = type_vectors.get(
                argument_type_strings[1], [0] * type_embedding_size
            )

            parameter_strings = call["parameters"]
            parameter0_vector = token_vectors.get(
                parameter_strings[0], [0] * name_embedding_size
            )
            parameter1_vector = token_vectors.get(
                parameter_strings[1], [0] * name_embedding_size
            )

            correct_vector = [
                th.tensor(base_vector),
                th.tensor(callee_vector),
                th.tensor(parameter0_vector),
                th.tensor(argument0_type_vector),
                th.tensor(argument0_vector),
                th.tensor(parameter1_vector),
                th.tensor(argument1_type_vector),
                th.tensor(argument1_vector),
            ]

            g = dgl.graph(calls_graph, num_nodes=len(correct_vector))
            g.ndata["features"] = self.get_tensor_feature(correct_vector)
            self.graphs.append(g)
            self.labels.append(LABELS["correct_args"] if self.bug_type == "all" else 0)
            swapped_args_vector = [
                th.tensor(base_vector),
                th.tensor(callee_vector),
                th.tensor(parameter0_vector),
                th.tensor(argument1_type_vector),
                th.tensor(argument1_vector),
                th.tensor(parameter1_vector),
                th.tensor(argument0_type_vector),
                th.tensor(argument0_vector),
            ]

            g = dgl.graph(swapped_calls_graph, num_nodes=len(swapped_args_vector))
            g.ndata["features"] = self.get_tensor_feature(swapped_args_vector)
            self.graphs.append(g)
            self.labels.append(LABELS["swapped_args"] if self.bug_type == "all" else 1)

    @property
    def dataset_type(self):
        return "training" if self.is_training else "eval"

    def process(self):
        filepath = "./data/large_homo_graph_data_{}_{}_{}.bin".format(
            self.dataset_type,
            "deepbugs",
            self.bug_type,
        )
        if os.path.exists(filepath):
            print("----Loading {} graph data----".format(self.dataset_type))
            self.graphs, label_dict = load_graphs(filepath)
            self.labels = label_dict["labels"]
        else:
            print("----Saving {} graph data----".format(self.dataset_type))
            if self.bug_type in [
                "incorrect_binary_operator",
                "incorrect_binary_operand",
            ]:
                self.pre_scan_binOps(binOps_training, binOps_eval)
                self.generate_graphs_from_binOps_ast()
            elif self.bug_type == "swapped_args":
                self.generate_graphs_from_calls_ast()
            else:
                self.pre_scan_binOps(binOps_training, binOps_eval)
                self.generate_graphs_from_binOps_ast()
                self.generate_graphs_from_calls_ast()

            random.shuffle(self.graphs)
            self.labels = th.LongTensor(self.labels)
            save_graphs(filepath, self.graphs, {"labels": self.labels})

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)

    @property
    def num_classes(self):
        """Number of classes."""
        return len(LABELS)
