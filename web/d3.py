from d3graph import d3graph, vec2adjmat
import numpy as np

# Initialize
def make_d3(df):
     d3 = d3graph()
     # Load karate example
     adjmat = vec2adjmat(df.source, df.target, weight=df.weight)
     label = np.array([10]*len(adjmat))
     node_size = np.array([10]*len(label))

     d3.graph(adjmat)
     d3.set_node_properties(color=label, label=label)
     d3.show()

     d3.set_node_properties(label=label, color=label, cmap='Set1')
     d3.show()

     d3.set_node_properties(size=node_size)
     d3.show()

     d3.set_node_properties(color=label, size=node_size, label=label)
     d3.show()

     d3.set_edge_properties(edge_distance=100)
     d3.set_node_properties(color=node_size, size=node_size, label=label)
     d3.show()

     d3 = d3graph(charge=1000)
     d3.graph(adjmat)
     d3.set_node_properties(color=node_size, size=node_size, label=label)
     d3.show()

     d3 = d3graph(collision=1, charge=250)
     d3.graph(adjmat)
     d3.set_node_properties(color=label, size=node_size, edge_size=node_size, cmap='Set1', label=label)
     d3.show()

     d3 = d3graph(collision=1, charge=250)
     d3.graph(adjmat)
     d3.set_node_properties(color=label, size=node_size, edge_size=node_size, edge_color='#00FFFF', cmap='Set1', label=label)
     d3.show()