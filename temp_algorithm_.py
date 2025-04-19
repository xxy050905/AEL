def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    import random
    if random.random() < 0.1 and len(unvisited_nodes) > 1:
        return random.choice(unvisited_nodes)
    min_dist = float('inf')
    nearest = unvisited_nodes[0]
    for node in unvisited_nodes:
        if distance_matrix[current_node][node] < min_dist:
            min_dist = distance_matrix[current_node][node]
            nearest = node
    return nearest