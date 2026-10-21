"""Independent checks for a manually reviewed experimental BFS.py.

Not a Python sandbox. Run only after inspecting the entire generated file and
passing its SHA256 with --reviewed-sha256; the comparison driver never imports it.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random


def verify(module):
    node, invert = module.TreeNode, module.invert_tree_bfs
    if invert(None) is not None:
        raise AssertionError("Empty root must return None")
    sample = node(1)
    if sample.val != 1 or sample.left is not None or sample.right is not None:
        raise AssertionError("TreeNode default children")
    right, left = node(3), node(2)
    root = node(1, left=left, right=right)
    if root.left is not left or root.right is not right:
        raise AssertionError("TreeNode child constructor")

    def check(root, nodes):
        before = [(n, n.val, n.left, n.right) for n in nodes]
        if invert(root) is not root:
            raise AssertionError("Return value changed root identity")
        for n, value, left, right in before:
            if n.val != value or n.left is not right or n.right is not left:
                raise AssertionError("Per-node value/edge identity mirror mismatch")
        if invert(root) is not root:
            raise AssertionError("Second inversion changed root")
        for n, value, left, right in before:
            if n.val != value or n.left is not left or n.right is not right:
                raise AssertionError("Double inversion did not restore original graph")

    check(root, [root, left, right])
    check(sample, [sample])
    # Asymmetric golden tree: original level order 4,2,7,1,3,6,9 -> 4,7,2,9,6,3,1.
    nodes = [node(v) for v in (4, 2, 7, 1, 3, 6, 9)]
    for i in range(3):
        nodes[i].left, nodes[i].right = nodes[2*i+1], nodes[2*i+2]
    invert(nodes[0])
    actual = [nodes[0].val, nodes[0].left.val, nodes[0].right.val,
              nodes[0].left.left.val, nodes[0].left.right.val,
              nodes[0].right.left.val, nodes[0].right.right.val]
    if actual != [4, 7, 2, 9, 6, 3, 1]:
        raise AssertionError("Golden mirrored level order")
    check(nodes[0], nodes)
    rng = random.Random(20261020)
    for _ in range(50):
        nodes = [node(rng.randrange(7)) for _ in range(rng.randrange(1, 250))]
        available = [(nodes[0], "left"), (nodes[0], "right")]
        for current in nodes[1:]:
            parent, edge = available.pop(rng.randrange(len(available)))
            setattr(parent, edge, current)
            available.extend([(current, "left"), (current, "right")])
        check(nodes[0], nodes)
    nodes = [node(i) for i in range(10000)]
    for parent, child in zip(nodes, nodes[1:]):
        parent.left = child
    check(nodes[0], nodes)
    return {"pass": True, "empty": True, "golden": True, "random_trees": 50,
            "deep_chain_nodes": 10000, "per_node_identity_edges": True,
            "double_inversion": True, "breadth_first": "requires source review"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--reviewed-sha256", required=True)
    args = parser.parse_args()
    digest = hashlib.sha256(args.file.read_bytes()).hexdigest()
    if digest != args.reviewed_sha256:
        parser.error("File differs from manually reviewed SHA256")
    spec = importlib.util.spec_from_file_location("reviewed_bfs", args.file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print(json.dumps({"sha256": digest, **verify(module)}))


if __name__ == "__main__":
    main()
