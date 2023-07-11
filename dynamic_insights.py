"""
[CMIP6 generalization tasks]
1. ability to dynamically construct the data pipeline for a scenario
2. ability to dynamically/automatically get new data from cmip6 portal
3. including past 80 years in data with ERA5




---------------------------------------------------------------------------



[1. ability to dynamically construct the data pipeline for a scenario]

framework notes:
- functions that take in one or more datasets/etc., and return a result
  - can have prerequisites on the input datasets, e.g. resolutions must match, etc.



[Dataflow pipeline operations]


rescale to target
- other dataset
- specified resolution


arithmetic between datasets
- add
- subtract
- multiply
- divide

arithmetic over single dataset
- threshold

aggregation over single dataset
- sum
- mean
- min
- max

temporal interpolation
- e.g. decadal to yearly

crop data by country





[SCENARIOS]

--------------- heat exposure ----------------
pop = load pop data [decadal]
tasmax = load tasmax data [monthly]

heat = tasmax > 35°C
heat = #TODO: to yearly. currently it is group by year and mean of heat...

pop = interpolate to yearly

pop, heat = crop (pop, heat) to overlapping time period

heat = regrid to pop

exposure = (heat > 0) * pop

result = split by country (exposure, ['US', 'China', 'India', ...])

--------------- crop suitability --------------
...

---------------------------------------------------------------------------



[2. ability to dynamically/automatically get new data from cmip6 portal]
# web scraper for downloading data, then gpt4 for combining into a single dataframe/xarray
#  ---> process for converting to single xarray is a function defined by the llm the gets saved with processed data + original data



---------------------------------------------------------------------------


[3. including past 80 years in data with ERA5]
TBD



"""


from enum import Enum
from typing import Any
from types import MethodType

import pdb


class Scenario(str, Enum):
    ssp126 = 'ssp126'
    ssp245 = 'ssp245'
    ssp370 = 'ssp370'
    ssp585 = 'ssp585'


class Realization(str, Enum):
    r1i1p1f1 = 'r1i1p1f1'
    #TODO: other realizations as needed

class Model(str, Enum):
    CAS_ESM2_0 = 'CAS-ESM2-0'
    #TODO: other models as needed

class Data(str, Enum):
    population = 'population'
    tasmax = 'tasmax'
    tas = 'tas'
    pr = 'pr'
    #TODO: other variables as needed

# Operations
# - load
# - regrid
# - crop
# - rescale
# - add
# - subtract
# - multiply
# - divide
# - threshold
# - sum
# - mean
# - min
# - max
# - interpolate
# - split


class Pipeline:


    def __init__(self):
        
        # settings for data to be used in pipeline
        self.realizations: list[Realization] = []
        self.scenarios: list[Scenario] = []
        self.models: list[Model] = []
        #target resolution?
        #target time frequency?
        #target country split?
        
        # list of steps in the pipeline DAG
        self.steps: list[tuple[MethodType, tuple[Any, ...]]] = []

        # keep track of whether pipeline has been compiled
        self.compiled = False
        self.last_set_identifier: str|None = None

        # namespace for binding operation results to identifiers 
        self.env: dict[str, Any] = {}

    def set_realizations(self, *realizations: Realization):
        """Set the cmip6 realization(s) to use for selecting data"""
        self.realizations = list(realizations)

    def set_scenarios(self, *scenarios: Scenario):
        """Set the cmip6 scenario(s) to use for selecting data"""
        self.scenarios = list(scenarios)

    def set_models(self, *models: Model):
        """Set the climate model(s) to pull data from in the pipeline"""
        self.models = list(models)

    def bind_value(self, identifier:str, value: Any):
        """Bind a value to an identifier in the pipeline namespace"""
        self.last_set_identifier = identifier
        self.env[identifier] = value

    def get_value(self, identifier:str):
        """Get a value from the pipeline namespace"""
        return self.env[identifier]
    
    def get_last_value(self):
        """Get the last value that was set in the pipeline namespace"""
        assert self.last_set_identifier is not None, 'No value has been set yet'
        return self.env[self.last_set_identifier]

    
    def load(self, identifier:str, data: Data):
        """Append data load step to the pipeline"""
        self.steps.append((self._do_load, (identifier, data,)))

    def _do_load(self, identifier:str, data: Data):
        """Perform execution of a data load step"""

        #special case variables separate from cmip6 data
        if data == Data.population:
            pdb.set_trace()
            ...

        pdb.set_trace()
        ...

    def compile(self):
        """Check that the pipeline is valid, insert inferred steps, etc."""
        assert not self.compiled, 'Pipeline has already been compiled'
        assert len(self.realizations) > 0, 'Must specify at least one realization with .set_realizations()'
        assert len(self.scenarios) > 0, 'Must specify at least one scenario with .set_scenarios()'
        assert len(self.models) > 0, 'Must specify at least one model with .set_models()'

        #TODO 
        # - type/size/shape checking, etc.
        # - insert any necessary steps in between to make sure data operands match shape for operations
        # - etc.

        self.compiled = True


    def execute(self):
        """Execute the pipeline"""

        assert self.compiled, 'Pipeline must be compiled before execution'

        for func, args in self.steps:
            func(*args)

#topological ordering (could be reconstructed back into a DAG)
pipe = Pipeline()
pipe.set_realizations(Realization.r1i1p1f1)
pipe.set_scenarios(Scenario.ssp585)
pipe.set_models(Model.CAS_ESM2_0)
pipe.load('pop', Data.population)
pipe.load('tasmax', Data.tasmax)
#...

#TBD on inferring needed transformations, e.g. regridding, etc.

pipe.compile()
pipe.execute()

res = pipe.get_last_value()
pop = pipe.get_value('pop')
tasmax = pipe.get_value('tasmax')

pdb.set_trace()
...


#every step print out: data type and data dims