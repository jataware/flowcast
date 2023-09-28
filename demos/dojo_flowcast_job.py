from __future__ import annotations

import json
from collections import defaultdict, deque
from typing import TypedDict
from flowcast.pipeline import Pipeline, Threshold, ThresholdType
from data import Realization, Scenario, CMIP6Data, OtherData, Model
from matplotlib import pyplot as plt
import pdb

from contextlib import contextmanager, ExitStack

@contextmanager
def net_netcdf():
    #collect the data header
    #download the netcdf file
    #open the netcdf file as an xarray dataset
    pdb.set_trace()
    try:
        yield xr.open_dataset('data.nc'), header
    finally:
        #close the netcdf file    
        pdb.set_trace()


def net_dag():
    pdb.set_trace()

def get_netcdf():
    pdb.set_trace()
    ...

def get_data_header():
    pdb.set_trace()
    ...


def main():

    #TODO: there will be multiple netcdf's to handle...
    with ExitStack() as stack:
        files, headers = zip(*[stack.enter_context(net_netcdf(name)) for name in names])
        #create loader functions for each dataset/variable (or do as they are loaded by the graph?)
        pdb.set_trace()
        graph = net_dag()
        ...

    # see example for dealing with multiple files:
    # with ExitStack() as stack:
    #     files = [stack.enter_context(open(fname)) for fname in filenames]
    #     # All opened files will automatically be closed at the end of
    #     # the with statement, even if attempts to open files later
    #     # in the list raise an exception



####################################################################3



    # load the DAG from the json file
    with open('input.json') as f:
        graph = json.loads(f.read())

    # sort the nodes (ensures pipeline execution order is correct)
    topological_sort(graph)

    # create the pipeline
    pipe = Pipeline()

    #keep track of nodes to plot
    nodes_to_plot: list[tuple[str,str]] = []

    # insert each step into the pipeline
    for node in graph['nodes']:
        print(node['id'], node['type'])
        
        if node['type'] == 'load':
            name = node['data']['input'].replace(' ', '_')
            if name == 'land_cover':
                loader = OtherData.land_cover()
            elif name == 'population':
                loader = OtherData.population(scenario=scenario)
            elif name == 'tasmax':
                loader = CMIP6Data.tasmax(realization=realization, scenario=scenario, model=Model.CAS_ESM2_0)
            elif name == 'tas':
                loader = CMIP6Data.tas(realization=realization, scenario=scenario, model=Model.FGOALS_f3_L)
            elif name == 'pr':
                loader = CMIP6Data.pr(realization=realization, scenario=scenario, model=Model.FGOALS_f3_L)
            else:
                raise ValueError(f'Unknown data type {name}')

            # add the step to the pipeline            
            pipe.load(node['id'], loader)

            #HACK for setting target geo/temporal resolution demo
            if name == 'land_cover':
                pipe.set_geo_resolution(node['id'])
            elif name == 'population':
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
            pipe.sum_reduce(node['id'], parent, dims)
            continue

        if node['type'] == 'save':
            parent, = get_node_parents(node['id'], graph, num_expected=1)

            # skip saving and plot instead
            nodes_to_plot.append((parent, node['data']['input'][:-3]))
            continue

        raise NotImplementedError(f'Parsing of node type {node["type"]} not implemented.')

        
        
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