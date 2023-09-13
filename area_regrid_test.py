from __future__ import annotations

import numpy as np
from dynamic_insights import Pipeline, Realization, Scenario, CMIP6Data, OtherData, Model
from spacetime import DatetimeNoLeap, LongitudeConvention, inplace_set_longitude_convention
from regrid import regrid_1d, RegridType
from matplotlib import pyplot as plt
import pdb



def main():
    pipe = Pipeline(
        realizations=Realization.r1i1p1f1, 
        scenarios=[Scenario.ssp585]#[Scenario.ssp126, Scenario.ssp245, Scenario.ssp370, Scenario.ssp585]
    )

    pipe.load('pop', OtherData.population)
    pipe.load('land_cover', OtherData.land_cover)
    pipe.load('tasmax', CMIP6Data.tasmax, Model.CAS_ESM2_0)
    pipe.load('pr', CMIP6Data.pr, Model.FGOALS_f3_L)
    pipe.load('tas', CMIP6Data.tas, Model.FGOALS_f3_L)
    pipe.execute()
    pop = pipe.get_value('pop').data
    modis = pipe.get_value('land_cover').data
    tasmax = pipe.get_value('tasmax').data
    pr = pipe.get_value('pr').data
    tas = pipe.get_value('tas').data


    #ensure the lon conventions match
    inplace_set_longitude_convention(pop, LongitudeConvention.neg180_180)
    inplace_set_longitude_convention(modis, LongitudeConvention.neg180_180)
    inplace_set_longitude_convention(tasmax, LongitudeConvention.neg180_180)
    inplace_set_longitude_convention(pr, LongitudeConvention.neg180_180)
    inplace_set_longitude_convention(tas, LongitudeConvention.neg180_180)

    
    # convert modis to the goe resolution of tasmax
    new_modis = regrid_1d(modis, tasmax['lat'].data, 'lat', aggregation=RegridType.mode)
    new_modis = regrid_1d(new_modis, tasmax['lon'].data, 'lon', aggregation=RegridType.mode)
    plt.figure()
    new_modis.plot()
    plt.figure()
    modis.plot()
    plt.show()



    # convert pr to yearly, and then the geo resolution of population
    time_axis = np.array([DatetimeNoLeap(year, 1, 1) for year in range(2015, 2101)])
    new_pr = regrid_1d(pr, time_axis, 'time', aggregation=RegridType.interp_mean)
    new_pr = regrid_1d(new_pr, pop['lat'].data, 'lat', aggregation=RegridType.interp_mean)
    new_pr = regrid_1d(new_pr, pop['lon'].data, 'lon', aggregation=RegridType.interp_mean)
    plt.figure()
    new_pr.isel(time=0,ssp=-1, realization=0).plot()
    plt.figure()
    pr.isel(time=0,ssp=-1, realization=0).plot()
    plt.show()



    #convert tas to yearly, and then the geo resolution of population
    time_axis = np.array([DatetimeNoLeap(year, 1, 1) for year in range(2015, 2101)])
    new_tas = regrid_1d(tas, time_axis, 'time', aggregation=RegridType.interp_mean)
    new_tas = regrid_1d(new_tas, pop['lat'].data, 'lat', aggregation=RegridType.interp_mean)
    new_tas = regrid_1d(new_tas, pop['lon'].data, 'lon', aggregation=RegridType.interp_mean)
    plt.figure()
    new_tas.isel(time=0,ssp=-1, realization=0).plot()
    plt.figure()
    tas.isel(time=0,ssp=-1, realization=0).plot()
    plt.show()



    #convert tasmax to yearly, and then the geo resolution of population
    time_axis = np.array([DatetimeNoLeap(year, 1, 1) for year in range(2015, 2101)])
    new_tasmax = regrid_1d(tasmax, time_axis, 'time', aggregation=RegridType.interp_mean)
    new_tasmax = regrid_1d(new_tasmax, pop['lat'].data, 'lat', aggregation=RegridType.interp_mean)
    new_tasmax = regrid_1d(new_tasmax, pop['lon'].data, 'lon', aggregation=RegridType.interp_mean)
    plt.figure()
    new_tasmax.isel(time=0,ssp=-1, realization=0).plot()
    plt.figure()
    tasmax.isel(time=0,ssp=-1, realization=0).plot()
    plt.show()

    
    #conservatively convert population to the resolution of modis
    new_pop = regrid_1d(pop, modis['lat'].data, 'lat', aggregation=RegridType.conserve)
    new_pop = regrid_1d(new_pop, modis['lon'].data, 'lon', aggregation=RegridType.conserve)

    #plot old vs new population (clip old population to match bounds of new population)
    plt.figure()
    old_pop = pop.isel(time=0,ssp=-1).sel(lat=slice(new_pop['lat'].max(), new_pop['lat'].min()), lon=slice(new_pop['lon'].min(), new_pop['lon'].max()))
    plt.imshow(np.log(old_pop + 1e-6))
    # plt.imshow(old_pop)

    plt.figure()
    plt.imshow(np.log(new_pop.isel(time=0,ssp=-1) + 1e-6))
    # plt.imshow(new_pop.isel(time=0,ssp=-1))
    
    old_pop_count = np.nansum(old_pop)
    new_pop_count = np.nansum(new_pop.isel(time=0,ssp=-1))
    print(f'comparing pop count: {old_pop_count=} vs {new_pop_count=}. Error: {np.abs(old_pop_count - new_pop_count) / old_pop_count:.4%}')

    plt.show()
    

    #conservatively convert population to the resolution of modis
    new_pop = regrid_1d(pop, tasmax['lat'].data, 'lat', aggregation=RegridType.conserve)
    new_pop = regrid_1d(new_pop, tasmax['lon'].data, 'lon', aggregation=RegridType.conserve)

    #plot old vs new population (clip old population to match bounds of new population)
    plt.figure()
    old_pop = pop.isel(time=0,ssp=-1).sel(lat=slice(new_pop['lat'].min(), new_pop['lat'].max(),-1), lon=slice(new_pop['lon'].min(), new_pop['lon'].max()))
    plt.imshow(np.log(old_pop + 1e-6), origin='lower')
    # plt.imshow(old_pop, origin='lower')

    plt.figure()
    plt.imshow(np.log(new_pop.isel(time=0,ssp=-1) + 1e-6), origin='lower')
    # plt.imshow(new_pop.isel(time=0,ssp=-1), origin='lower')

    old_pop_count = np.nansum(old_pop)
    new_pop_count = np.nansum(new_pop.isel(time=0,ssp=-1))
    print(f'comparing pop count: {old_pop_count=} vs {new_pop_count=}. Error: {np.abs(old_pop_count - new_pop_count) / old_pop_count:.4%}')

    plt.show()




    pdb.set_trace()
    ...





if __name__ == '__main__':
    main()