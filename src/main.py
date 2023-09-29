import os
import logging

from extract_ee_data import extract_ee_data
from dineof_to_tif import dineof_to_tif
from tif_to_eecollection import cogs_to_collection
from dineof_runner import exec_dineof
from config import config

def run_job(start_time, end_time, variable, out_collection, min_modes=15, max_modes=25):

    # outuri = f'{config.BUCKET}/dineof_staging/dineof_in_{variable}.nc'
    outuri = f'dineof_in_{variable}.nc'

    print('Starting data extraction...')
    extract_ee_data(
        outuri, 
        config.REGION, 
        start_time, 
        end_time,
        collection=variable,
        crs=config.CRS,
        scale=config.SCALE, 
        date_chunks=100
    )
    print('Data extraction complete!')

    dineof_in = outuri # NOTE: need to update data transfer if output to GCS
    dineof_out = f'{variable}_dineof_out.nc'

    print('Running DINEOF...')
    exec_dineof(
        dineof_in,
        variable,
        min_modes=min_modes,
        max_modes=max_modes,
    )
    print('DINEOF complete!')

    bucket_path = f'experimental/dineof_{variable}'
    outbucket = f'{config.OUTPUT_BUCKET}{bucket_path}'

    print('Unpacking DINEOF output as tifs...')
    # unpack the DINEOF output as COGs
    dineof_to_tif(
        dineof_out, 
        dineof_in, 
        outbucket,
        variable,
    )
    print('Finished writing DINEOF output to tifs!')

    prefix = f'{bucket_path}/'
    print(prefix)

    print('Starting to add tifs to collection...')
    # add the output COGs to an EE ImageCollection
    cogs_to_collection(config.OUTPUT_BUCKET, prefix, config.EE_PROJECT, out_collection)
    print('Done adding tifs to collection...')

    print('ALL DONE!')

    return

def main():
    
    starttime_ = os.getenv('STARTTIME', default = '2018-10-01')
    endtime_ = os.getenv('ENDTIME', default = '2019-10-01')
    variable_ = os.getenv('VARIABLE', default = 'sst')
    out_collection = os.getenv('OUTCOLLECTION', default = 'test_out_collection')
    min_modes = int(os.getenv('MINMODES', default = '10'))
    max_modes = int(os.getenv('MAXMODES', default = '20'))
    task_index_ = int(os.getenv('CLOUD_RUN_TASK_INDEX', default = '1'))
    logging.debug(f'task index: {task_index_}')

    run_job(
        starttime_, 
        endtime_, 
        variable_, 
        out_collection,
        min_modes=min_modes,
        max_modes=max_modes
    )


if __name__ == "__main__":
    main()