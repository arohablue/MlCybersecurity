class MyClass:
    def __init__(self, cluster_node_list):
        self._cluster_node_list = cluster_node_list

    @property
    def cluster_nodes(self):
        for cluster_node in self._cluster_node_list:
            yield cluster_node

# Example usage:
cluster_nodes_list = [1, 2, 3, 4, 5]
my_instance = MyClass(cluster_nodes_list)

# Accessing the property
for node in my_instance.cluster_nodes:
    print(node)