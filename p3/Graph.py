# Graph class
import numpy as np
import matplotlib.pyplot as plt
import heapq

# The Graph class is orgainzed into 2 fields: "nodes" and "edges"
# The "nodes" field is a 2 by N numpy array specifiying the positions of the N vertices/nodes
# The "edges" field is an array of length N of arrays of nodes, with list i giving the list
#          of nodes that node i is connected to. For the graphs we are dealing with, you may assume
#          that if i is connected to j then j is connected to i.
class Graph(object):
    """Graph to search through using Dijkstra's or A*. Stores each node as a 2-dimensional point,
    and the edges as a map from node number to a list of node numbers it is connected to"""
    def __init__(self, nodes, edges):
        super(Graph, self).__init__()
        # Nodes are stored as x-y values
        self.nodes = np.array(nodes)
        # Edges are a map from a node to list of nodes
        self.edges = edges

    def drawPath(self,path):
        # Draw nodes
        plt.scatter(self.nodes[:,0],self.nodes[:,1])
        # Draw edges
        linesx = []
        linesy = []
        for i in xrange(len(self.edges)):
            for j in self.edges[i]:
                plt.plot([self.nodes[i][0], self.nodes[j][0]],
                         [self.nodes[i][1], self.nodes[j][1]],'b-',linewidth=0.5)
        # Don't draw path if there is none
        if len(path) == 0:
            plt.show()
            return True
        # Draw start and end nodes
        plt.scatter( \
            self.nodes[[path[0],path[-1]],0], \
            self.nodes[[path[0],path[-1]],1],s=90,c='k',marker='x') # ,linewidths='1'
        # Highlight chosen path nodes
        plt.scatter(self.nodes[path,0],self.nodes[path,1],s=30,c='r')
        # Highlight chosen path edges
        pathx = []
        pathy = []
        for i in xrange(len(path)-1):
            if path[i] >= len(self.nodes) or path[i+1] not in self.edges[path[i]]:
                # Invalid path
                plt.show()
                return False
            else:
                pathx.append(self.nodes[path[i]][0])
                pathx.append(self.nodes[path[i+1]][0])
                pathx.append(None)
                pathy.append(self.nodes[path[i]][1])
                pathy.append(self.nodes[path[i+1]][1])
                pathy.append(None)
        plt.plot(pathx,pathy,'r',linewidth=2)
        # Done, all successful
        plt.show()
        return True

    def draw(self):
        self.drawPath([])

def GridGraph(N,M=None):
    if M is None:
        M = N
    nodes = []
    edges = []
    for i in xrange(N):
        for j in xrange(M):
            # Add node
            nodes.append([float(i),float(j)])
            # Add nodes edges
            # Corner cases
            nodeEdges = []
            if i != 0:
                nodeEdges.append(N*(i-1) + j)
            if i != N-1:
                nodeEdges.append(N*(i+1) + j)
            if j != 0:
                nodeEdges.append(N*i + j-1)
            if j != M-1:
                nodeEdges.append(N*i + j+1)
            edges.append(nodeEdges)
    return Graph(nodes=nodes,edges=edges)

def UniformNodesUniformEdges(N,p=1.7):
    """Generate the nodes at points uniformly at random in the rectangle [0,N] x [0,N]"""
    nodes = N*np.random.rand(N,2);
    adjacency = np.random.binomial(1,p/N,size=(N,N))
    adjacency = (adjacency + adjacency.T) > 0
    edges = []
    for e in adjacency:
        edges.append([x for x in range(len(e)) if e[x]])
    return Graph(nodes=nodes, edges=edges)

def UniformNodesClosestEdges(N,k=5):
    """Generate the nodes uniformly at random, but create the edges to the closest k nodes"""
    nodes = N*np.random.rand(N,2);
    edges = [[] for x in xrange(N)]
    for i in xrange(N):
        h = []
        for j in xrange(N):
            if j == i:
                continue
            if len(h) < k:
                heapq.heappush(h,(-np.linalg.norm(nodes[i]-nodes[j]),j))
            else:
                heapq.heappushpop(h,(-np.linalg.norm(nodes[i]-nodes[j]),j))
        for j in h:
            if i not in edges[j[1]] and j[1] not in edges[i]:
                edges[i].append(j[1])
                edges[j[1]].append(i)

    return Graph(nodes=nodes, edges=edges)


def graphMain():
    # Create nodes
    nodes = np.array([[0.0, 0.0],  # Node 0
                  [0.0, 1.0],  # Node 1
                  [1.0, 0.0],  # Node 2
                  [1.0, 1.0]]) # Node 3
                  
    # Create edges
    edges = [ [1, 2], # Node 0
             [0, 3], # Node 1
             [0, 3], # Node 2
             [1, 2]] # Node 3
             
    g = Graph(nodes,edges) # A simple graph with nodes on the unit square
    g.draw()

#    N = 20
#    g = UniformNodesClosestEdges(N)
#    g.draw()

if __name__ == "__main__":
    graphMain()
