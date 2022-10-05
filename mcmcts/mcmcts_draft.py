from typing import List

from data_structures.data_structures import ImmutableBoard

class ValueAccumulator:
    def __init__(self):



class MCMCTSNode:
    def __init__(self, state : ImmutableBoard, parent: "MCMCTSNode" =None):
        self.state = state
        self.parent = parent
        self.children : List["MCMCTSNode"] = []
        self.N_visits = 0
        self.value = 0
        #other stats

    @property
    def score


    def add_child(self, child_state):
        child_node = MCMCTSNode(child_state, self)
        self.children.append(child_node)
        return child_node

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_actions())

    def is_terminal(self):
        return self.state.is_terminal()

    def __repr__(self):
        return "Node(state={}, parent={}, children={}, visits={}, value={})".format(
            self.state, self.parent, self.children, self.visits, self.value
        )


class MCTS:
    def __init__(self, root_state : ImmutableBoard, max_depth : int):
        self.root = MCMCTSNode(root_state)
        self.max_depth = max_depth

    def search(self, n_iterations : int):
        for i in range(n_iterations):
            node = self.select_node()
            reward = self.simulate(node)
            self.backpropagate(node, reward)

    def select_node(self):
        node = self.root
        while not node.is_terminal():
            if node.is_fully_expanded():
                node = self.select_child(node)
            else:
                return self.expand(node)
        return node

    def select_child(self, node):
        raise NotImplementedError

    def expand(self, node):
        raise NotImplementedError

