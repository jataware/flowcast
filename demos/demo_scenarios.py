from flowcast.pipeline import Pipeline, Threshold, ThresholdType 
from flowcast.spacetime import Resolution, Frequency
from matplotlib import pyplot as plt
from data import Realization, Scenario, Model, CMIP6Data, OtherData

import pdb


def country_heat_scenario():

    pipe = Pipeline(low_memory=True)
    
    # set geo/temporal resolution targets for operations in the pipeline
    pipe.set_geo_resolution('pop')
    pipe.set_time_resolution(Frequency.yearly)

    # load the data
    pipe.load('pop', OtherData.population(scenario=Scenario.ssp585))
    pipe.load('tasmax', CMIP6Data.tasmax(model=Model.CAS_ESM2_0, scenario=Scenario.ssp585, realization=Realization.r1i1p1f1))
    
    # operations on the data to perform the scenario
    pipe.threshold('heat', 'tasmax', Threshold(308.15, ThresholdType.greater_than))
    pipe.multiply('exposure0', 'heat', 'pop')
    pipe.reverse_geocode('exposure1', 'exposure0', ['China', 'India', 'United States', 'Canada', 'México', 'Brazil', 'Australia'])
    pipe.sum_reduce('exposure2', 'exposure1', dims=['lat', 'lon'])
    pipe.save('exposure2', 'exposure.nc')

    # run the pipeline
    pipe.execute()

    # e.g. extract any live results
    res = pipe.get_last_value()
    # pop = pipe.get_value('pop')
    # tasmax = pipe.get_value('tasmax')

    # plot all the countries on a single plot
    for country in res.data['admin0'].values:
        res.data.sel(admin0=country).plot(label=country)

    plt.title('People Exposed to Heatwaves by Country')
    plt.legend()
    plt.show()


def state_heat_scenario():

    pipe = Pipeline(low_memory=True)
    
    # set geo/temporal resolution targets for operations in the pipeline
    pipe.set_geo_resolution('pop')
    pipe.set_time_resolution(Frequency.yearly)

    # load the data
    pipe.load('pop', OtherData.population(scenario=Scenario.ssp585))
    pipe.load('tasmax', CMIP6Data.tasmax(model=Model.CAS_ESM2_0, scenario=Scenario.ssp585, realization=Realization.r1i1p1f1))
    
    # operations on the data to perform the scenario
    pipe.threshold('heat', 'tasmax', Threshold(308.15, ThresholdType.greater_than))
    pipe.multiply('exposure0', 'heat', 'pop')
    pipe.reverse_geocode('exposure1', 'exposure0', ['California', 'Texas', 'Virginia', 'Pennsylvania', 'Florida'], admin_level=1)
    pipe.sum_reduce('exposure2', 'exposure1', dims=['lat', 'lon'])
    pipe.save('exposure2', 'exposure.nc')

    # run the pipeline
    pipe.execute()

    # e.g. extract any live results
    res = pipe.get_last_value()
    # pop = pipe.get_value('pop')
    # tasmax = pipe.get_value('tasmax')

    # plot all the states on a single plot
    for state in res.data['admin1'].values:
        res.data.sel(admin1=state).plot(label=state)

    plt.title('People Exposed to Heatwaves by State')
    plt.legend()
    plt.show()
    

def crop_scenario():
    pipe = Pipeline(low_memory=True)
    pipe.set_geo_resolution('modis')
    pipe.set_time_resolution(Frequency.monthly)
    
    pipe.load('raw_modis', OtherData.land_cover())
    pipe.fixed_geo_regrid('modis', 'raw_modis', Resolution(0.2, 0.2))
    pipe.load('tas', CMIP6Data.tas(realization=Realization.r1i1p1f1, scenario=Scenario.ssp585, model=Model.FGOALS_f3_L))
    pipe.load('pr', CMIP6Data.pr(realization=Realization.r1i1p1f1, scenario=Scenario.ssp585, model=Model.FGOALS_f3_L))


    #tas_mean = tas


    pipe.execute()

    pdb.set_trace()


    #TODO: rest of scenario


def demo_scenario():
    pipe = Pipeline()
    pipe.set_geo_resolution('modis')
    pipe.set_time_resolution('pop')
    pipe.load('modis', OtherData.land_cover())
    pipe.load('pop', OtherData.population(scenario=Scenario.ssp585))
    pipe.load('tasmax', CMIP6Data.tasmax(realization=Realization.r1i1p1f1, scenario=Scenario.ssp585, model=Model.CAS_ESM2_0))
    pipe.threshold('heat', 'tasmax', Threshold(308.15, ThresholdType.greater_than))
    pipe.multiply('exposure0', 'heat', 'pop')
    pipe.threshold('urban_mask', 'modis', Threshold(13, ThresholdType.equal))
    pipe.threshold('not_urban_mask', 'modis', Threshold(13, ThresholdType.not_equal))
    pipe.multiply('urban_exposure', 'exposure0', 'urban_mask')
    pipe.multiply('not_urban_exposure', 'exposure0', 'not_urban_mask')
    pipe.sum_reduce('global_urban_exposure', 'urban_exposure', dims=['lat', 'lon'])
    pipe.sum_reduce('global_not_urban_exposure', 'not_urban_exposure', dims=['lat', 'lon'])
    pipe.save('global_urban_exposure', 'global_urban_exposure.nc')
    pipe.save('global_not_urban_exposure', 'global_not_urban_exposure.nc')

    pipe.execute()

    pipe.get_value('global_urban_exposure').data.plot()
    pipe.get_value('global_not_urban_exposure').data.plot()
    plt.legend(['urban', 'not urban'])
    plt.show()

    pdb.set_trace()
    ...



if __name__ == '__main__':
    # test_admin1()
    # country_heat_scenario()
    state_heat_scenario()
    # crop_scenario()
    # demo_scenario()