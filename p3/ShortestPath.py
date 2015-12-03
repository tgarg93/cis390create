import Graph
import numpy as np
import heapq

def straightLineDistance(g,a,b):
    """ Returns the distance between nodes a and b in Graph g """
    return np.linalg.norm(g.nodes[a]-g.nodes[b])

def astar(g,start,goal):
    """Implements the A-star algorithm with distance to the goal as a heuristic
    Input: g: Graph as specified in the Graph class, with N nodes, and edges for each node
           start: start node, a number from 0 to N-1
           goal: goal node, a number from 0 to N-1 (not necessarily distict from start)
    Output: a tuple (path, numopened)
            path: list of the indices of the nodes on the shortest path found
                starting with "start" and ending with "end"
            numopened: number of nodes opened while searching for the shortest path"""

    numOfNodes = g.nodes.shape[0]

    # If start or goal nodes are invalid then return empty path and zero opened nodes
    if start < 0 or start >= numOfNodes or goal < 0 or goal >= numOfNodes:
        return (list([]), 0)

    #If start is the same as goal then return start and one opened node
    if start == goal:
        return (list([start]), 1)

    g_score = []
    f_score = []
    #list to make it easy to pull the minimum f_score of nodes in open set
    open_f_score = []
    closedSet = []
    openSet = []
    prev = [-1 for x in xrange(numOfNodes)]
    path = []
    numOpened = 0

    openSet.append(start)

    g_score.append([0, start])

    # f_score of start node is g_score[start] + distance from start to goal
    for x in g_score:
        if x[1] == start:
            f_score_start = x[0] + straightLineDistance(g, start, goal)
            f_score.append([f_score_start, start])
            open_f_score.append([f_score_start, start])

    for x in xrange(numOfNodes):
        if x != start:
            g_score.append([float("inf"), x])
            f_score.append([float("inf"), x])

    # While there are still nodes to be explored
    while len(openSet) > 0:
        # pick from the nodes in the open set the one with the minimum f_score
        currentNodeElement = min(open_f_score)
        current = currentNodeElement[1]
        numOpened += 1
        # If we have reached the goal then popular the path from prev references and then reverse
        if current == goal:
            predecessor = goal
            path.append(goal)
            while prev[predecessor] != start:
                path.append(prev[predecessor])
                predecessor = prev[predecessor]
            path.append(start)
            path.reverse()
            return (path, numOpened)

        # Remove the current node from the open set
        openSet.remove(current)
        open_f_score.remove(currentNodeElement)
        # Add the current node to the closed set
        closedSet.append(current)

        # Loop through all the adjacent
        for adj in g.edges[current]:
            #print "Adjacent vertices of " + str(current) + " are " + str(adj)
            if adj in closedSet:
                continue
            g_score_current = -1
            g_score_adj = -1
            for x in g_score:
                if x[1] == current:
                    g_score_current = x[0]
                    #print "g_score_current is " + str(g_score_current)
                elif x[1] == adj:
                    g_score_adj = x[0]
                    #print "g_score_adj is " + str(g_score_adj)

            tentative_g_score = g_score_current + straightLineDistance(g, adj, current)

            if tentative_g_score < g_score_adj:
                prev[adj] = current
                for x in g_score:
                    if x[1] == adj:
                        g_score.remove(x)
                        g_score.append([tentative_g_score, adj])
                for x in f_score:
                    if x[1] == adj:
                        f_score.remove(x)
                        f_score.append([tentative_g_score + straightLineDistance(g, adj, goal), adj])
                for x in open_f_score:
                    if x[1] == adj:
                        open_f_score.remove(x)
                        open_f_score.append([tentative_g_score + straightLineDistance(g, adj, goal), adj])
                if adj not in openSet:
                    openSet.append(adj)
                    open_f_score.append([tentative_g_score + straightLineDistance(g, adj, goal), adj])

    return (list([]), numOpened)

    """Implements Dijkstra's shortest path algorithm
    Input: g: Graph as specified in the Graph class, with N nodes, and edges for each node
           start: start node, a number from 0 to N-1
           goal: goal node, a number from 0 to N-1 (not necessarily distict from start)
    Output: a tuple (path, numopened)
            path: list of the indices of the nodes on the shortest path found
                starting with "start" and ending with "end"
            numopened: number of nodes opened while searching for the shortest path"""

def dijkstras(g,start,goal):

    numOfNodes = g.nodes.shape[0]

    # If start or goal nodes are invalid then return empty path and zero opened nodes
    if start < 0 or start >= numOfNodes or goal < 0 or goal >= numOfNodes:
        return (list([]), 0)

    #If start is the same as goal then return start and one opened node
    if start == goal:
        return (list([start]), 1)

    priorityQueue = []
    visitedNodes = []
    prev = [-1 for x in xrange(numOfNodes)]
    path = []
    numopened = 0

    priorityQueue.append([0,start])

    for x in xrange(numOfNodes):
        if x != start:
            priorityQueue.append([float("inf"),x])

    # While the heap is not empty
    while len(priorityQueue) > 0:
        #Pop the element with minimum distance
        minNode = min(priorityQueue)
        priorityQueue.remove(minNode)

        #print "min node popped is: " + str(minNode[1]) + " with distance: " + str(minNode[0])
        numopened += 1
        # if found goal then terminate
        if minNode[1] == goal:
            #print "distance is " + str(minNode[0])
            if minNode[0] == float("inf"):
                return (list([]), numopened)
            predecessor = goal
            path.append(goal)
            while prev[predecessor] != start:
                path.append(prev[predecessor])
                predecessor = prev[predecessor]
            path.append(start)
            path.reverse()
            return (path, numopened)
        elif minNode[1] not in visitedNodes:
            #Add to visited nodes
            visitedNodes.append(minNode[1])
            #Loop through all adjacent edges of the node
            for adj in g.edges[minNode[1]]:
                #print "Adjacent vertices of " + str(minNode[1]) + " are " + str(adj)
                #Find the adjacent elements in the heap
                for n in priorityQueue:
                    if n[1] == adj:
                        #discardedNode =
                        dist = np.linalg.norm(g.nodes[adj]-g.nodes[minNode[1]]) + minNode[0]
                        if dist < n[0]:
                            #print "Updating distance to " + str(dist)
                            priorityQueue.remove(n)
                            priorityQueue.append([dist, adj])
                            prev[adj] = minNode[1]


def testInvalidStartDijkstraMain():
    nodes = np.array([[0.0, 0.0],  # Node 0
                  [0.0, 1.0],  # Node 1
                  [1.0, 0.0],  # Node 2
                  [1.0, 1.0]]) # Node 3

    # Create edges
    edges = [ [1, 2], # Node 0
             [0, 3], # Node 1
             [0, 3], # Node 2
             [1, 2]] # Node 3

    g = Graph.Graph(nodes,edges) # A simple graph with nodes on the unit square
    result = dijkstras(g, -1, 1)
    print result[0] == [] and result[1] == 0

def testStartEqualsGoalDijkstraMain():
    nodes = np.array([[0.0, 0.0],  # Node 0
                  [0.0, 1.0],  # Node 1
                  [1.0, 0.0],  # Node 2
                  [1.0, 1.0]]) # Node 3

    # Create edges
    edges = [ [1, 2], # Node 0
             [0, 3], # Node 1
             [0, 3], # Node 2
             [1, 2]] # Node 3

    g = Graph.Graph(nodes,edges) # A simple graph with nodes on the unit square
    result = dijkstras(g, 1, 1)
    print result[0] == [1] and result[1] == 1

def testUnitSquareDijkstra1Main():
    nodes = np.array([[0.0, 0.0],  # Node 0
                  [0.0, 1.0],  # Node 1
                  [1.0, 0.0],  # Node 2
                  [1.0, 1.0]]) # Node 3

    # Create edges
    edges = [ [1, 2], # Node 0
             [0, 3], # Node 1
             [0, 3], # Node 2
             [1, 2]] # Node 3

    g = Graph.Graph(nodes,edges) # A simple graph with nodes on the unit square
    result = dijkstras(g, 0, 1)
    print result[0] == [0,1] and result[1] == 2

def testUnitSquareDijkstra2Main():
    nodes = np.array([[0.0, 0.0],  # Node 0
                  [0.0, 1.0],  # Node 1
                  [1.0, 0.0],  # Node 2
                  [1.0, 1.0]]) # Node 3

    # Create edges
    edges = [ [1, 2], # Node 0
             [0, 3], # Node 1
             [0, 3], # Node 2
             [1, 2]] # Node 3

    g = Graph.Graph(nodes,edges) # A simple graph with nodes on the unit square
    result = dijkstras(g, 0, 2)
    print result[0] == [0,2] and result[1] == 3

def testUnitSquareDijkstra3Main():
    nodes = np.array([[0.0, 0.0],  # Node 0
                  [0.0, 1.0],  # Node 1
                  [1.0, 0.0],  # Node 2
                  [1.0, 1.0]]) # Node 3

    # Create edges
    edges = [ [1, 2], # Node 0
             [0, 3], # Node 1
             [0, 3], # Node 2
             [1, 2]] # Node 3

    g = Graph.Graph(nodes,edges) # A simple graph with nodes on the unit square
    result = dijkstras(g, 0, 3)
    print result[0] == [0,1,3] and result[1] == 4

def testUnitSquareDijkstra4Main():
    nodes = np.array([[0.0, 0.0],  # Node 0
                  [0.0, 1.0],  # Node 1
                  [1.0, 0.0],  # Node 2
                  [1.0, 1.0]]) # Node 3

    # Create edges
    edges = [ [1, 2, 3], # Node 0
             [0, 3], # Node 1
             [0, 3], # Node 2
             [1, 2]] # Node 3

    g = Graph.Graph(nodes,edges) # A simple graph with nodes on the unit square
    result = dijkstras(g, 0, 3)
    print result[0] == [0,3] and result[1] == 4


def testUnitSquareDijkstra5Main():
    nodes = np.array([[0.0, 0.0],  # Node 0
                  [1.0, 0.0],  # Node 1
                  [2.0, 0.0],  # Node 2
                  [2.0, 4.0],  # Node 3
                  [2.0, 4.0],  # Node 4
                  [4.0, 4.0 ]]) # Node 5

    # Create edges
    edges = [ [1], # Node 0
             [0, 2], # Node 1
             [1, 3], # Node 2
             [2, 4], # Node 3
              [3]] # Node 4

    g = Graph.Graph(nodes,edges) # A simple graph with nodes on the unit square
    result = dijkstras(g, 0, 4)
    print result[0] == [0,1,2,3,4] and result[1] == 5

def testUnitSquareDijkstra6Main():
    nodes = np.array([[0.0, 0.0],  # Node 0
                  [1.0, 0.0],  # Node 1
                  [2.0, 0.0],  # Node 2
                  [2.0, 4.0],  # Node 3
                  [4.0, 4.0 ]]) # Node 4

    # Create edges
    edges = [ [1,4], # Node 0
             [0, 2], # Node 1
             [1, 3], # Node 2
             [2, 4], # Node 3
              [0, 3]] # Node 4

    g = Graph.Graph(nodes,edges) # A simple graph with nodes on the unit square
    result = dijkstras(g, 0, 4)
    print result
    print result[0] == [0,4] and result[1] == 4


def testUnreachableDijkstraMain():
    nodes = np.array([[0.0, 0.0],  # Node 0
                  [0.0, 1.0],  # Node 1
                  [1.0, 0.0],  # Node 2
                  [1.0, 1.0]]) # Node 3

    # Create edges
    edges = [ [1, 2], # Node 0
             [0], # Node 1
             [0], # Node 2
             []] # Node 3

    g = Graph.Graph(nodes,edges) # A simple graph with nodes on the unit square
    result = dijkstras(g, 0, 3)
    print result
    print result[0] == [] and result[1] == 4

def testInvalidStartAStarMain():
    nodes = np.array([[0.0, 0.0],  # Node 0
                  [0.0, 1.0],  # Node 1
                  [1.0, 0.0],  # Node 2
                  [1.0, 1.0]]) # Node 3

    # Create edges
    edges = [ [1, 2], # Node 0
             [0, 3], # Node 1
             [0, 3], # Node 2
             [1, 2]] # Node 3

    g = Graph.Graph(nodes,edges) # A simple graph with nodes on the unit square
    result = astar(g, -1, 1)
    print result[0] == [] and result[1] == 0

def testStartEqualsGoalAStarMain():
    nodes = np.array([[0.0, 0.0],  # Node 0
                  [0.0, 1.0],  # Node 1
                  [1.0, 0.0],  # Node 2
                  [1.0, 1.0]]) # Node 3

    # Create edges
    edges = [ [1, 2], # Node 0
             [0, 3], # Node 1
             [0, 3], # Node 2
             [1, 2]] # Node 3

    g = Graph.Graph(nodes,edges) # A simple graph with nodes on the unit square
    result = astar(g, 1, 1)
    print result[0] == [1] and result[1] == 1

def testUnitSquareAStar1Main():
    nodes = np.array([[0.0, 0.0],  # Node 0
                  [0.0, 1.0],  # Node 1
                  [1.0, 0.0],  # Node 2
                  [1.0, 1.0]]) # Node 3

    # Create edges
    edges = [ [1, 2], # Node 0
             [0, 3], # Node 1
             [0, 3], # Node 2
             [1, 2]] # Node 3

    g = Graph.Graph(nodes,edges) # A simple graph with nodes on the unit square
    result = astar(g, 0, 1)
    print result[0] == [0,1] and result[1] == 2

def testUnitSquareAStar2Main():
    nodes = np.array([[0.0, 0.0],  # Node 0
                  [0.0, 1.0],  # Node 1
                  [1.0, 0.0],  # Node 2
                  [1.0, 1.0]]) # Node 3

    # Create edges
    edges = [ [1, 2], # Node 0
             [0, 3], # Node 1
             [0, 3], # Node 2
             [1, 2]] # Node 3

    g = Graph.Graph(nodes,edges) # A simple graph with nodes on the unit square
    result = astar(g, 0, 2)
    print result[0] == [0,2] and result[1] == 2

def testUnitSquareAStar3Main():
    nodes = np.array([[0.0, 0.0],  # Node 0
                  [0.0, 1.0],  # Node 1
                  [1.0, 0.0],  # Node 2
                  [1.0, 1.0]]) # Node 3

    # Create edges
    edges = [ [1, 2], # Node 0
             [0, 3], # Node 1
             [0, 3], # Node 2
             [1, 2]] # Node 3

    g = Graph.Graph(nodes,edges) # A simple graph with nodes on the unit square
    result = astar(g, 0, 3)
    print result[0] == [0,1,3] and result[1] == 4

def testUnitSquareAStar4Main():
    nodes = np.array([[0.0, 0.0],  # Node 0
                  [0.0, 1.0],  # Node 1
                  [1.0, 0.0],  # Node 2
                  [1.0, 1.0]]) # Node 3

    # Create edges
    edges = [ [1, 2, 3], # Node 0
             [0, 3], # Node 1
             [0, 3], # Node 2
             [1, 2]] # Node 3

    g = Graph.Graph(nodes,edges) # A simple graph with nodes on the unit square
    result = astar(g, 0, 3)
    print result[0] == [0,3] and result[1] == 2


def testUnitSquareAStar5Main():
    nodes = np.array([[0.0, 0.0],  # Node 0
                  [1.0, 0.0],  # Node 1
                  [2.0, 0.0],  # Node 2
                  [2.0, 4.0],  # Node 3
                  [2.0, 4.0],  # Node 4
                  [4.0, 4.0 ]]) # Node 5

    # Create edges
    edges = [ [1], # Node 0
             [0, 2], # Node 1
             [1, 3], # Node 2
             [2, 4], # Node 3
              [3]] # Node 4

    g = Graph.Graph(nodes,edges) # A simple graph with nodes on the unit square
    result = astar(g, 0, 4)
    print result
    print result[0] == [0,1,2,3,4] and result[1] == 5

def testUnitSquareAStar6Main():
    nodes = np.array([[0.0, 0.0],  # Node 0
                  [1.0, 0.0],  # Node 1
                  [2.0, 0.0],  # Node 2
                  [2.0, 4.0],  # Node 3
                  [4.0, 4.0 ]]) # Node 4

    # Create edges
    edges = [ [1,4], # Node 0
             [0, 2], # Node 1
             [1, 3], # Node 2
             [2, 4], # Node 3
              [0, 3]] # Node 4

    g = Graph.Graph(nodes,edges) # A simple graph with nodes on the unit square
    result = astar(g, 0, 4)
    print result
    print result[0] == [0,4] and result[1] == 2


def testUnreachableAStarMain():
    nodes = np.array([[0.0, 0.0],  # Node 0
                  [0.0, 1.0],  # Node 1
                  [1.0, 0.0],  # Node 2
                  [1.0, 1.0]]) # Node 3

    # Create edges
    edges = [ [1, 2], # Node 0
             [0], # Node 1
             [0], # Node 2
             []] # Node 3

    g = Graph.Graph(nodes,edges) # A simple graph with nodes on the unit square
    result = astar(g, 0, 3)
    print result[0] == [] and result[1] == 3

#if __name__ == "__main__":
#    testUnreachableAStarMain()
#    testUnitSquareAStar1Main()
#    testUnitSquareAStar2Main()
#    testUnitSquareAStar3Main()
#    testUnitSquareAStar4Main()
#    testUnitSquareAStar5Main()
#    testUnitSquareAStar6Main()

