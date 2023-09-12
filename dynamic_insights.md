# Dynamic Insights
A framework for dynamically allowing users to easily string together operations with CMIP6/related data, and create custom insights into the data.

## Overview
To create a dynamic insight, the user will:
1. create an empty pipeline
2. add steps to the pipeline
    - while steps are added, several compile-time checks are performed to validate the pipeline
3. execute the pipeline


## Compilation vs Execution
Pipelines separate out compilation/building the pipeline from execution because the operations on datasets may be expensive, and ideally we would like to know before we run the pipeline if there is a problem that would cause it to crash.


## Pipeline steps
A step in a pipeline is usually some operation on a dataset. Most steps will produce a new result dataset, which the user must name, and can be used in subsequent steps.

### Static Single Assignment
No steps should modify data in place--all data are treated as constant, and names may not be reused. This is called Static Single Assignment (SSA), and is a common technique in compilers for reducing complexity. SSA is used here mainly to simplify regridding logic:
- Regridding can target matching existing data's resolution. If that existing data were to later change resolution, it would be more difficult to bookkeep and make sure any dependent data gets updated as well

In practice, what this means is that instead of a pipeline looking like this: 
```
x = load_data(...)
y = load_data(...)
x = x > 100
y = y * x
y = y.sum(dims=['lat', 'lon'])
y = y.split_by_country(['China', 'India', 'United States', 'Canada', 'Mexico'])
y.save('result.nc')
```
new names are needed for each step:
```
x1 = load_data(...)
y1 = load_data(...)
x2 = x1 > 100
y2 = y * x2
y3 = y2.sum(dims=['lat', 'lon'])
y4 = y3.split_by_country(['China', 'India', 'United States', 'Canada', 'Mexico'])
y4.save('result.nc')
```

### Pipeline operations
steps include things like:
- loading data
- geo/temporal regridding
- unary operations on a single dataset (threshold, etc.)
- binary operations between two datasets (add, multiply, divide, etc.)
- aggregating over a dataset's dimensions
- reverse geocoding a dataset to countries
- save data to a file



### Compile-time checks
The main compile-time checks performed on a pipeline are:
- ensuring variable names are not reused (enforcing SSA)
- ensuring geo/temporal regridding targets are set before operations that will require regridding
- ensuring that 


## Example
An example pipeline for the extreme heat scenario:

```
# extreme heat scenario
pipe = Pipeline(
    realizations=Realization.r1i1p1f1, 
    scenarios=[Scenario.ssp126, Scenario.ssp245, Scenario.ssp370, Scenario.ssp585],
)

# load the data
pipe.load('pop', OtherData.population)
pipe.load('tasmax', CMIP6Data.tasmax, Model.CAS_ESM2_0)

# set target geo/temporal resolution of existing data in pipeline
pipe.set_geo_resolution('tasmax')
pipe.set_time_resolution('tasmax')

# operations on the data to perform the scenario
pipe.threshold('heat', 'tasmax', Threshold(308.15, ThresholdType.greater_than))
pipe.multiply('exposure0', 'heat', 'pop')
pipe.country_split('exposure1', 'exposure0', ['China', 'India', 'United States', 'Canada', 'Mexico'])
pipe.sum('exposure2', 'exposure1', dims=['lat', 'lon'])

# save the results
pipe.save('exposure2', 'exposure.nc')

# run the pipeline
pipe.execute()
```

which basically gets converted to the following operations:
```
pop = load_population_data()
tasmax = load_tasmax_data(Model.CAS_ESM2_0)
heat = tasmax > 35°C
__tmp_0__ = regrid(pop, match=heat) #automatically inserted by the pipeline
exposure0 = heat * __tmp_0__
exposure1 = split_by_country(exposure0, ['China', 'India', 'United States', 'Canada', 'Mexico'])
exposure2 = exposure1.sum(dims=['lat', 'lon'])
exposure2.to_netcdf('exposure.nc')
```





user can build a pipeline of steps by sequentially calling operations (e.g. load data, threshold, multiply, save data, etc.), and specifying input variable names and names for the results
as the user builds the pipeline, the Pipeline class runs several "compile-time" checks, such as ensuring variable names are valid, ensuring data will be regridded if necessary, etc.
Once the pipeline is built, it can be executed, which performs all of the accumulated steps in sequence
separate compile/execute phases are so that you don't have to wait for expensive data operations to know if something's invalid

## Compilation
The [`Pipeline`](dynamic_insights.py#L252) class provides an [`@compile`](dynamic_insights.py#L291) decorator which is used to annotate methods that will be used as pipeline operations. The decorator will perform several compile-time checks, and then saves the method in a list of pipeline steps to be executed later.


## Dynamic Regridding
Data frequently is not at a common geo/temporal resolution and requires regridding to be used together with other data. During pipeline execution, the current data's resolution/frequency is stored with the data as part of the [`Variable`](dynamic_insights.py#L215) class. When an operation is to be performed with one or more variables, their current frequency/resolution is compared to the current target frequency/resolution of the pipeline, and if they do not match, a regridding operation is called. The result is given a temporary identifier, which is used in the operation in place of the original un-regridded data.

## Pipeline Operation Signatures
Here are some example signatures for pipeline operations:
```
def set_geo_resolution(self, target:Resolution|str): ...
def set_time_resolution(self, target:Frequency|str): ...
def load(self, name:ResultID, /, data:CMIP6Data|OtherData, model:Model|None=None): ...
def threshold(self, y:ResultID, x:OperandID, /, threshold:Threshold): ...
def multiply(self, y:ResultID, x1:OperandID, x2:OperandID, /): ...
def country_split(self, y:ResultID, x:OperandID, /, countries:list[str]): ...
def sum(self, y:ResultID, x:OperandID, /, dims:list[str]): ...
def save(self, x:OperandID, /, path:str): ...
```

Some things to note:
- All of these are annotated with `@compile` which means when they are called, they are added to the pipeline's list of steps, and their execution is deferred to when `pipeline.execute()` is called
- `set_geo_resolution` and `set_time_resolution` don't operate on any data, and instead manage global pipeline settings. They just look like normal methods with no special considerations (other than being annotated with `@compile`).
- `load`, `threshold`, `multiply`, `country_split`, and `sum` all produce a result value, which is named by the user. Resulting values should use the `ResultID` type annotation, and be position-only values (positional only is achieved with the `/` in the signature). By convention, the result value is the first argument in the method signature, and also I've been calling it `y`.
- `threshold`, `multiply`, `country_split`, `sum`, and `save` all take input arguments, which must refer to existing data in the pipeline namespace. Input values should use the `OperandID` type annotation, and be position-only values (positional only is achieved with the `/` in the signature). By convention, the input values follow the result value (if any), and I've been calling them `x`, `x1`, `x2`, etc.
- any relevant non-identifier arguments can appear wherever make sense. Typically I put them after the `/` making them positional or keyword arguments.


## Misc Notes:
- to call an `@compile` decorated function immediately (rather than add it to the pipeline, which is the default behavior), methods have a `.unwrapped` property, which returns the original unwrapped version of the function. This is useful if you want to use one of the methods inside a pipeline method at runtime, e.g. as is done in the [`auto_regrid()`](dynamic_insights.py#L683) method.

## Current progress

### Extreme Heat Scenario
See the original [extreme heat scenario](test.py#L87) for comparison.
See current [extreme heat scenario](dynamic_insights.py#L797) for building the pipeline

For the goal of recreating the extreme heat scenario, most of the pipeline framework is complete.

The biggest incomplete component is the handling of regridding data. Currently you can set up a pipeline that will need to regrid data, and when the execution gets to a step that needs to perform the actual regridding on the data, it is marked with a `NotImplementedError`. Namely
- [fixed_time_regrid()](dynamic_insights.py#L630)
- [matched_time_regrid()](dynamic_insights.py#L642)
- [fixed_geo_regrid()](dynamic_insights.py#L655)
- [matched_geo_regrid()](dynamic_insights.py#L667)

`fixed` vs `matched` refers to which type of target is used for the regridding:
- fixed targets a static resolution/frequency, e.g. 1°x1°, 1 month
- matched targets (by name) the resolution/frequency of another dataset in the pipeline

I think this is **HARD** difficulty

Additionally for the extreme heat scenario, two of the necessary runtime operations are not yet implemented:
- [country_split()](dynamic_insights.py#L743). This will largely pull from the version implemented in [test.py](test.py#L133), but needs to be adapted to work in the pipeline framework. I think this is **MEDIUM** difficulty
- [sum()](dynamic_insights.py#L756). This will be basically just the xarray sum() method using the specified dimensions. I think this is **EASY** difficulty

### Crop viability scenario
See the original [crop viability scenario](test.py#L190) for comparison.
See current [crop viability scenario](dynamic_insights.py#L835) for stub of building the pipeline

In terms of the crop viability scenario, not much of the specifics are implemented. 
- CMIP6 data for surface air temperature (tas) and precipitation (pr) can be loaded, but modis crop use data loading is not implemented
- most of the operations used such as taking the baseline temperature/pr values, computing z-scores, etc. will need to be added as methods available in the pipeline