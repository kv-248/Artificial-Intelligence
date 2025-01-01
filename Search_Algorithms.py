import numpy as np
import pickle
from collections import deque
import heapq
import time
import tracemalloc
import csv
import pandas as pd
import matplotlib.pyplot as plt



# General Notes:
# - Update the provided file name (code_<RollNumber>.py) as per the instructions.
# - Do not change the function name, number of parameters or the sequence of parameters.
# - The expected output for each function is a path (list of node names)
# - Ensure that the returned path includes both the start node and the goal node, in the correct order.
# - If no valid path exists between the start and goal nodes, the function should return None.


# Algorithm: Iterative Deepening Search (IDS)

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 2, 9, 8, 5, 97, 98, 12]

def depth_limited_search(adj_matrix, current_node, goal_node, depth, visited_nodes):
    
    if current_node == goal_node:
        return [current_node]
    if depth <= 0:
        return None
     
    for neighbor in range(len(adj_matrix[current_node])):
        if adj_matrix[current_node][neighbor] > 0 and neighbor not in visited_nodes:
            visited_nodes.add(neighbor)  
            result = depth_limited_search(adj_matrix, neighbor, goal_node, depth - 1, visited_nodes)
            if result is not None:
                return [current_node] + result

    return None

def get_ids_path(adj_matrix, start_node, goal_node):
    for limit in range(len(adj_matrix)):
        visited_nodes = {start_node}
        path = depth_limited_search(adj_matrix, start_node, goal_node, limit, visited_nodes)
        if path is not None:
            return path
    return None

# Algorithm: Bi-Directional Search

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 2, 9, 8, 5, 97, 98, 12]

def get_bidirectional_search_path(adj_matrix, start_node, goal_node):
    if start_node == goal_node:
        return [start_node]
    forward_queue = deque([(start_node, [start_node])])
    backward_queue = deque([(goal_node, [goal_node])])
    forward_visited = {start_node: [start_node]}
    backward_visited = {goal_node: [goal_node]}
    
    while forward_queue and backward_queue:
        path = bfs_step(adj_matrix, forward_queue, forward_visited, backward_visited, False)
        if path:
            return path
        
        path = bfs_step(adj_matrix, backward_queue, backward_visited, forward_visited, True)
        if path:
            return path
    
    return None

def bfs_step(adj_matrix, queue, visited, other_visited, reverse):
    current, path = queue.popleft()
    
    for neighbor in range(len(adj_matrix)):
        if (adj_matrix[current][neighbor] if not reverse else adj_matrix[neighbor][current]) and neighbor not in visited:
            new_path = path + [neighbor]
            visited[neighbor] = new_path
            queue.append((neighbor, new_path))
            
            if neighbor in other_visited:
                if reverse:
                    return other_visited[neighbor] + new_path[::-1][1:]
                else:
                    return new_path + other_visited[neighbor][::-1][1:]
    
    return None







# Algorithm: A* Search Algorithm

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - node_attributes: Dictionary of node attributes containing x, y coordinates for heuristic calculations.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 28, 10, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 27, 9, 8, 5, 97, 28, 10, 12]

def dist(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def heuristic(start_node, node, goal_node, node_attributes):
    x1, y1 = node_attributes[start_node]['x'], node_attributes[start_node]['y']
    x2, y2 = node_attributes[node]['x'], node_attributes[node]['y']
    x3, y3 = node_attributes[goal_node]['x'], node_attributes[goal_node]['y']
    return dist(x1, y1, x2, y2) + dist(x2, y2, x3, y3)

def get_astar_search_path(adj_matrix, node_attributes, start_node, goal_node):
    node_names = list(node_attributes.keys())
    node_indices = {name: index for index, name in enumerate(node_names)}
    
    open_list = []
    heapq.heappush(open_list, (heuristic(start_node, start_node, goal_node, node_attributes), 0, node_indices[start_node], [start_node]))
    
    closed_set = set()
    
    g_scores = {index: float('inf') for index in range(len(node_names))}
    g_scores[node_indices[start_node]] = 0
    
    while open_list:
        f, g, current_index, path = heapq.heappop(open_list)
        current_node = node_names[current_index]
        
        if current_node == goal_node:
            return path
        
        closed_set.add(current_index)
        for neighbor_index, weight in enumerate(adj_matrix[current_index]):
            if weight != 0 and neighbor_index not in closed_set:
                neighbor_node = node_names[neighbor_index]
                tentative_g = g + weight
                
                if tentative_g < g_scores[neighbor_index]:
                    g_scores[neighbor_index] = tentative_g
                    new_f = tentative_g + heuristic(start_node, neighbor_node, goal_node, node_attributes)
                    heapq.heappush(open_list, (new_f, tentative_g, neighbor_index, path + [neighbor_node]))
    
    return None



# Algorithm: Bi-Directional Heuristic Search

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - node_attributes: Dictionary of node attributes containing x, y coordinates for heuristic calculations.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 34, 33, 11, 32, 31, 3, 5, 97, 28, 10, 12]

def get_bidirectional_heuristic_search_path(adj_matrix, node_attributes, start_node, goal_node):
    forward_open_list = []
    reverse_open_list = []
    
    forward_g_scores = {vertex: float('inf') for vertex in range(len(adj_matrix))}
    reverse_g_scores = {vertex: float('inf') for vertex in range(len(adj_matrix))}
    
    forward_g_scores[start_node] = 0
    reverse_g_scores[goal_node] = 0

    heapq.heappush(forward_open_list, (heuristic(start_node, start_node, goal_node, node_attributes), 0, start_node, [start_node]))
    heapq.heappush(reverse_open_list, (heuristic(goal_node, goal_node, start_node, node_attributes), 0, goal_node, [goal_node]))

    forward_closed_set = {start_node: [start_node]}
    reverse_closed_set = {goal_node: [goal_node]}

    while forward_open_list and reverse_open_list:
        f_fwd, g_fwd, current_fwd_node, forward_path = heapq.heappop(forward_open_list)

        if current_fwd_node in reverse_closed_set:
            return forward_path + reverse_closed_set[current_fwd_node][1:]

        for neighbor_node, weight in enumerate(adj_matrix[current_fwd_node]):
            if weight and neighbor_node not in forward_closed_set:
                tentative_forward_cost = g_fwd + weight
                if tentative_forward_cost < forward_g_scores.get(neighbor_node, float('inf')):
                    forward_g_scores[neighbor_node] = tentative_forward_cost
                    new_fwd_f = tentative_forward_cost + heuristic(start_node, neighbor_node, goal_node, node_attributes)
                    heapq.heappush(forward_open_list, (new_fwd_f, tentative_forward_cost, neighbor_node, forward_path + [neighbor_node]))
                    forward_closed_set[neighbor_node] = forward_path + [neighbor_node]
        f_rev, g_rev, current_rev_node, reverse_path = heapq.heappop(reverse_open_list)

        if current_rev_node in forward_closed_set:
            return forward_closed_set[current_rev_node] + reverse_path[1:]

        for neighbor_node, weight in enumerate(adj_matrix[current_rev_node]):
            if weight and neighbor_node not in reverse_closed_set:
                tentative_reverse_cost = g_rev + weight
                if tentative_reverse_cost < reverse_g_scores.get(neighbor_node, float('inf')):
                    reverse_g_scores[neighbor_node] = tentative_reverse_cost
                    new_rev_f = tentative_reverse_cost + heuristic(goal_node, neighbor_node, start_node, node_attributes)
                    heapq.heappush(reverse_open_list, (new_rev_f, tentative_reverse_cost, neighbor_node, [neighbor_node] + reverse_path))
                    reverse_closed_set[neighbor_node] = [neighbor_node] + reverse_path

    return None





# def measure_algorithm_performance(algorithm_func, adj_matrix, node_attributes, start_node, goal_node):
#     start_time = time.time()
#     tracemalloc.start()
#     if node_attributes:
#         path = algorithm_func(adj_matrix, node_attributes, start_node, goal_node)
#     else:
#         path = algorithm_func(adj_matrix, start_node, goal_node)
    
#     end_time = time.time()
#     current, peak = tracemalloc.get_traced_memory()
#     tracemalloc.stop()
#     path_cost = 0
#     if path:
#         for i in range(len(path) - 1):
#             path_cost += adj_matrix[path[i]][path[i + 1]]

#     return round((end_time - start_time) * 1000, 2), round(peak / 1024, 2), path_cost  


# def compare_algorithms(adj_matrix, node_attributes=None):
#     results = []

#     for start_node in range(len(adj_matrix)):
#         for goal_node in range(len(adj_matrix)):
#             if start_node != goal_node:
#                 ids_time, ids_memory, ids_cost = measure_algorithm_performance(get_ids_path, adj_matrix, None, start_node, goal_node)
#                 bidi_time, bidi_memory, bidi_cost = measure_algorithm_performance(get_bidirectional_search_path, adj_matrix, None, start_node, goal_node)
#                 astar_time, astar_memory, astar_cost = measure_algorithm_performance(get_astar_search_path, adj_matrix, node_attributes, start_node, goal_node)
#                 bidi_heur_time, bidi_heur_memory, bidi_heur_cost = measure_algorithm_performance(get_bidirectional_heuristic_search_path, adj_matrix, node_attributes, start_node, goal_node)
#                 results.append({
#                     'start_node': start_node,
#                     'goal_node': goal_node,
#                     'Iterative Deepening Search (ms)': ids_time,
#                     'Iterative Deepening Search (KB)': ids_memory,
#                     'Iterative Deepening Search (cost)': ids_cost,
#                     'Bidirectional Search (ms)': bidi_time,
#                     'Bidirectional Search (KB)': bidi_memory,
#                     'Bidirectional Search (cost)': bidi_cost,
#                     'A* Search (ms)': astar_time,
#                     'A* Search (KB)': astar_memory,
#                     'A* Search (cost)': astar_cost,
#                     'Bidirectional Heuristic Search (ms)': bidi_heur_time,
#                     'Bidirectional Heuristic Search (KB)': bidi_heur_memory,
#                     'Bidirectional Heuristic Search (cost)': bidi_heur_cost
#                 })
#                 print(results[-1])

#     with open('results.csv', mode='w', newline='') as file:
#         fieldnames = ['start_node', 'goal_node', 
#                       'Iterative Deepening Search (ms)', 'Iterative Deepening Search (KB)', 'Iterative Deepening Search (cost)',
#                       'Bidirectional Search (ms)', 'Bidirectional Search (KB)', 'Bidirectional Search (cost)',
#                       'A* Search (ms)', 'A* Search (KB)', 'A* Search (cost)',
#                       'Bidirectional Heuristic Search (ms)', 'Bidirectional Heuristic Search (KB)', 'Bidirectional Heuristic Search (cost)']
        
#         writer = csv.DictWriter(file, fieldnames=fieldnames)
#         writer.writeheader()
#         writer.writerows(results)

#     return results


# file_path = 'results.csv'  
# data = pd.read_csv(file_path)


# ids_time = data['Iterative Deepening Search (ms)']
# ids_space = data['Iterative Deepening Search (KB)']
# ids_cost = data['Iterative Deepening Search (cost)']

# bidirectional_time = data['Bidirectional Search (ms)']
# bidirectional_space = data['Bidirectional Search (KB)']
# bidirectional_cost = data['Bidirectional Search (cost)']

# astar_time = data['A* Search (ms)']
# astar_space = data['A* Search (KB)']
# astar_cost = data['A* Search (cost)']

# bhs_time = data['Bidirectional Heuristic Search (ms)']
# bhs_space = data['Bidirectional Heuristic Search (KB)']
# bhs_cost = data['Bidirectional Heuristic Search (cost)']

# def scatter_plot(x_data, y_data, labels, title, xlabel, ylabel, colors):
#     plt.figure(figsize=(10, 6))
    
#     for x, y, label, color in zip(x_data, y_data, labels, colors):
#         plt.scatter(x, y, label=label, color=color, alpha=0.7, s=40)  # Set size with 's' for smaller dots
    
#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# alg_labels = ['IDS', 'Bidirectional Search', 'A* Search', 'Bidirectional Heuristic Search']
# alg_colors = ['red', 'green', 'blue', 'purple']  


# x_data_time_space = [ids_time, bidirectional_time, astar_time, bhs_time]
# y_data_time_space = [ids_space, bidirectional_space, astar_space, bhs_space]

# scatter_plot(x_data_time_space, y_data_time_space, alg_labels, 'Time vs Space', 'Time (ms)', 'Space (KB)', alg_colors)

# x_data_time_cost = [ids_time, bidirectional_time, astar_time, bhs_time]
# y_data_time_cost = [ids_cost, bidirectional_cost, astar_cost, bhs_cost]

# scatter_plot(x_data_time_cost, y_data_time_cost, alg_labels, 'Time vs Cost', 'Time (ms)', 'Cost', alg_colors)

# x_data_space_cost = [ids_space, bidirectional_space, astar_space, bhs_space]
# y_data_space_cost = [ids_cost, bidirectional_cost, astar_cost, bhs_cost]

# scatter_plot(x_data_space_cost, y_data_space_cost, alg_labels, 'Space vs Cost', 'Space (KB)', 'Cost', alg_colors)





def bonus_problem(adj_matrix):
    n = len(adj_matrix)
    graph = [[] for _ in range(n)]
    
    for i in range(n) :
        for j in range(n) :
            if adj_matrix[i][j] > 0 :
                graph[i].append(j)
                graph[j].append(i)

    for i in range(n) :
        graph[i] = list(set(graph[i]))
        
    disc = [-1] * n
    low = [-1] * n
    parent = [-1] * n
    bridges = []
    time = [0]
    
    def dfs(u):
        disc[u] = low[u] = time[0]
        time[0] += 1
        
        for v in graph[u]:
            if disc[v] == -1:
                parent[v] = u
                dfs(v)
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]:
                    bridges.append((min(u, v), max(u, v)))  
            elif v != parent[u]:
                low[u] = min(low[u], disc[v])
    
    for i in range(n):
        if disc[i] == -1:
            dfs(i)
    
    return bridges




if __name__ == "__main__":
  adj_matrix = np.load('IIIT_Delhi.npy')
  with open('IIIT_Delhi.pkl', 'rb') as f:
    node_attributes = pickle.load(f)

  start_node = int(input("Enter the start node: "))
  end_node = int(input("Enter the end node: "))

  print(f'Iterative Deepening Search Path: {get_ids_path(adj_matrix,start_node,end_node)}')
  print(f'Bidirectional Search Path: {get_bidirectional_search_path(adj_matrix,start_node,end_node)}')
  print(f'A* Path: {get_astar_search_path(adj_matrix,node_attributes,start_node,end_node)}')
  print(f'Bidirectional Heuristic Search Path: {get_bidirectional_heuristic_search_path(adj_matrix,node_attributes,start_node,end_node)}')
  print(f'Bonus Problem: {bonus_problem(adj_matrix)}')
