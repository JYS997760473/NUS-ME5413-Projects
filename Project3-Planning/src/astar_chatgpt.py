import heapq

# Define a function to calculate the heuristic distance
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Define the A* search function
def astar(graph, start, goal):
    # Create the open and closed sets
    open_set = []
    closed_set = set()

    # Initialize the start node
    heapq.heappush(open_set, (0, start))
    came_from = {}

    # Initialize the g and f scores
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while len(open_set) > 0:
        # Get the node with the lowest f score
        current = heapq.heappop(open_set)[1]

        # If we have reached the goal, reconstruct the path and return it
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        # Add the current node to the closed set
        closed_set.add(current)

        # Loop through the neighbors of the current node
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)

            # If the neighbor is out of bounds or unwalkable, skip it
            if neighbor[0] < 0 or neighbor[0] >= len(graph) or neighbor[1] < 0 or neighbor[1] >= len(graph[0]) or graph[neighbor[0]][neighbor[1]] == 0:
                continue

            # If the neighbor is in the closed set, skip it
            if neighbor in closed_set:
                continue

            # Calculate the tentative g score for the neighbor
            tentative_g_score = g_score[current] + heuristic(current, neighbor)

            # If the neighbor is not in the open set, add it
            if neighbor not in [n[1] for n in open_set]:
                heapq.heappush(open_set, (tentative_g_score + heuristic(neighbor, goal), neighbor))

            # If the tentative g score is greater than or equal to the neighbor's g score, skip it
            elif tentative_g_score >= g_score.get(neighbor, float('inf')):
                continue

            # Otherwise, update the neighbor's g score, f score, and came from values
            came_from[neighbor] = current
            g_score[neighbor] = tentative_g_score
            f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)

    # If we reach here, there is no path to the goal
    return None

def main():

    board = [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]

    start = (0, 0)
    end = (6, 6)

    path = astar(board, start, end)
    print(path)


if __name__ == '__main__':
    main()