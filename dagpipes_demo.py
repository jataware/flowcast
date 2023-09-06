from __future__ import annotations

import json
from collections import defaultdict, deque
from typing import TypedDict
from dynamic_insights import Pipeline, Realization, Scenario, CMIP6Data, OtherData, Threshold, ThresholdType, Model
from matplotlib import pyplot as plt
import pdb



def main():
    # load the DAG from the json file
    with open('input.json') as f:
        graph = json.loads(f.read())

    # sort the nodes (ensures pipeline execution order is correct)
    topological_sort(graph)

    # create the pipeline
    pipe = Pipeline(realizations=Realization.r1i1p1f1, scenarios=Scenario.ssp585)

    #keep track of nodes to plot
    nodes_to_plot: list[tuple[str,str]] = []

    # insert each step into the pipeline
    for node in graph['nodes']:
        print(node['id'], node['type'])
        
        if node['type'] == 'load':
            try:
                var = CMIP6Data(node['data']['input'].replace(' ', '_'))
            except ValueError:
                var = OtherData(node['data']['input'].replace(' ', '_'))
            
            pipe.load(node['id'], var, Model.CAS_ESM2_0)

            #HACK for setting target geo/temporal resolution demo
            if var == OtherData.land_cover:
                pipe.set_geo_resolution(node['id'])
            elif var == OtherData.population:
                pipe.set_time_resolution(node['id'])

            continue
                
        if node['type'] == 'threshold':
            parent, = get_node_parents(node['id'], graph, num_expected=1)
            pipe.threshold(node['id'], parent, Threshold(float(node['data']['input']['value']), ThresholdType[node['data']['input']['type']]))
            continue

        if node['type'] == 'multiply':
            left, right = get_node_parents(node['id'], graph, num_expected=2)
            pipe.multiply(node['id'], left, right)
            continue    

        if node['type'] == 'sum':
            parent, = get_node_parents(node['id'], graph, num_expected=1)
            dims = [dim for dim in  ['lat', 'lon', 'time', 'country', 'scenario', 'realization'] if node['data']['input'][dim]]
            pipe.sum(node['id'], parent, dims)
            continue

        if node['type'] == 'save':
            parent, = get_node_parents(node['id'], graph, num_expected=1)

            # skip saving and plot instead
            nodes_to_plot.append((parent, node['data']['input'][:-3]))
            continue

        
        
    # run the pipeline
    pipe.execute()

    # plot each on the same figure
    for node_id, _ in nodes_to_plot:
        node = pipe.get_value(node_id)
        node.data.plot()

    plt.title('Exposure to extreme heat')
    plt.legend([name for _, name in nodes_to_plot])
    plt.ylabel('People (Hundreds of Millions)')
    plt.show()
    

def get_node_parents(node_id: str, graph: Graph, num_expected:int=None) -> list[str]:
    parents = [edge['source'] for edge in graph['edges'] if edge['target'] == node_id]
    if num_expected is not None:
        assert len(parents) == num_expected, f'Node {node_id} has incorrect number of parents: {parents}. Expected {num_expected}.'
    return parents

class Graph(TypedDict):
    nodes: list[dict[str, str|dict]]
    edges: list[dict[str, str|dict]]

def topological_sort(graph: Graph):
    nodes = {node['id']: node for node in graph['nodes']}
    edges = [(edge['source'], edge['target']) for edge in graph['edges']]
    
    in_degree = defaultdict(int)

    for _, target in edges:
        in_degree[target] += 1

    # Create a queue and enqueue all nodes with in-degree 0
    queue = deque([node_id for node_id in nodes if in_degree[node_id] == 0])

    result = []

    while queue:
        current = queue.popleft()
        result.append(nodes[current])  # you can replace with current if you only need node_id

        for src, tgt in edges:
            if src == current:
                in_degree[tgt] -= 1
                if in_degree[tgt] == 0:
                    queue.append(tgt)

    # update the graph with the sorted nodes
    graph['nodes'] = result






if __name__ == '__main__':
    main()