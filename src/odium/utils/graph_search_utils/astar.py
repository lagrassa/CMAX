import numpy as np
import heapq


class Node:
    def __init__(self, simple_state):
        self.simple_state = simple_state
        self._g = None
        self._h = None
        self._came_from = None
        self._action = None

    def __eq__(self, other):
        if np.array_equal(self.simple_state, other.simple_state):
            return True
        return False

    def __hash__(self):
            return hash(tuple(self.simple_state))


class Astar:
    def __init__(self, heuristic_fn, successors_fn, check_goal_fn, num_expansions, actions):
        self.heuristic_fn = heuristic_fn
        self.successors_fn = successors_fn
        self.check_goal_fn = check_goal_fn
        self.num_expansions = num_expansions
        self.actions = actions

    def act(self, start_node):
        closed_set = set()
        open = []

        if hasattr(start_node, '_came_from'):
            # Ensure start node has no parent
            del start_node._came_from

        reached_goal = False
        start_node._g = 0
        h = start_node._h = self.heuristic_fn(start_node)
        f = start_node._g + start_node._h

        count = 0
        start_triplet = [f, h, count, start_node]
        heapq.heappush(open, start_triplet)
        count += 1
        open_d = {start_node: start_triplet}

        for _ in range(self.num_expansions):
            f, h, _, node = heapq.heappop(open)
            del open_d[node]
            closed_set.add(node)
            # Check if expanded state is goal
            if self.check_goal_fn(node):
                reached_goal = True
                best_node = node
                break

            for action in self.actions:
                neighbor, cost = self.successors_fn(node, action)
                if neighbor in closed_set:
                    continue

                tentative_g = node._g + cost
                if neighbor not in open_d:
                    neighbor._came_from = node
                    neighbor._action = action
                    neighbor._g = tentative_g
                    h = neighbor._h = self.heuristic_fn(
                        neighbor)
                    f = neighbor._g + neighbor._h
                    d = open_d[neighbor] = [tentative_g +
                                            h, h, count, neighbor]
                    heapq.heappush(open, d)
                    count += 1
                else:
                    neighbor = open_d[neighbor][3]
                    if tentative_g < neighbor._g:
                        neighbor._came_from = node
                        neighbor._action = action
                        neighbor._g = tentative_g
                        open_d[neighbor][0] = tentative_g + neighbor._h
                        heapq.heapify(open)

        if not reached_goal:
            # Find the node with the least f value in the open list
            best_node = None
            best_node_cost = np.inf
            for n in open_d.keys():
                if n._h + n._g < best_node_cost:
                    best_node = n
                    best_node_cost = n._h + n._g
                elif n._h + n._g == best_node_cost:
                    # Then choose the one with least heuristic
                    if n._h < best_node._h:
                        best_node = n
                        best_node_cost = n._h + n._g

        info = {'start_node_f': start_node._g + start_node._h,
                'best_node_f': best_node._g + best_node._h,
                'start_node_h': start_node._h,
                'best_node': best_node,
                'open': open_d,
                'closed': closed_set
                }

        best_action, path = self.get_best_action(start_node, best_node)
        info['path'] = path
        return best_action, info

    def get_best_action(self, start_node, best_node):
        node = best_node._came_from
        action = best_node._action
        path = [best_node.simple_state]
        while True:
            path.append(node.simple_state)
            if hasattr(node, '_came_from'):
                # Not the start node
                next_node = node._came_from
                action = node._action
                if not hasattr(next_node, '_came_from'):
                    # Next node is start node
                    break
                else:
                    node = next_node
            else:
                # Start node
                break

        path.append(start_node.simple_state)
        path = path[::-1]
        return action, path
