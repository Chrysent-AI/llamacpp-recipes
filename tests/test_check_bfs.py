from collections import deque
from types import SimpleNamespace
import unittest

from check_bfs import verify


class Node:
    def __init__(self, val, left=None, right=None):
        self.val, self.left, self.right = val, left, right


def correct(root):
    if root is None:
        return None
    queue = deque([root])
    while queue:
        node = queue.popleft()
        node.left, node.right = node.right, node.left
        queue.extend(child for child in (node.left, node.right) if child is not None)
    return root


class CheckerTests(unittest.TestCase):
    def test_correct_iterative_inversion_passes(self):
        result = verify(SimpleNamespace(TreeNode=Node, invert_tree_bfs=correct))
        self.assertTrue(result["pass"])
        self.assertEqual(result["deep_chain_nodes"], 10000)

    def test_root_only_swap_is_rejected(self):
        def root_only(root):
            if root:
                root.left, root.right = root.right, root.left
            return root
        with self.assertRaisesRegex(AssertionError, "Golden"):
            verify(SimpleNamespace(TreeNode=Node, invert_tree_bfs=root_only))

    def test_replacement_root_is_rejected(self):
        def copied(root):
            return Node(root.val, root.right, root.left) if root else None
        with self.assertRaisesRegex(AssertionError, "identity"):
            verify(SimpleNamespace(TreeNode=Node, invert_tree_bfs=copied))

    def test_recursive_inversion_is_rejected_on_deep_chain(self):
        def recursive(root):
            if root:
                root.left, root.right = recursive(root.right), recursive(root.left)
            return root
        with self.assertRaises(RecursionError):
            verify(SimpleNamespace(TreeNode=Node, invert_tree_bfs=recursive))


if __name__ == "__main__":
    unittest.main()
