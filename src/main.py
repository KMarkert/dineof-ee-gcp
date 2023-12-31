import os
import logging

from extract_ee_data import extract_ee_data
from dineof_to_tif import dineof_to_tif
from tif_to_eecollection import cogs_to_collection
from dineof_runner import exec_dineof
from ee_auth import ee_initialize

def run_job(
    project, 
    output_bucket, 
    region, 
    start_time, 
    end_time, 
    band, 
    out_collection, 
    min_modes=15, 
    max_modes=25, 
    crs='epsg:4326', 
    scale=None
):

    ee_initialize(project)

    # outuri = f'{config.BUCKET}/dineof_staging/dineof_in_bande}.nc'
    outuri = f'dineof_in_{band}.nc'

    print('Starting data extraction...')
    extract_ee_data(
        outuri, 
        region, 
        start_time, 
        end_time,
        band=band,
        crs=crs,
        scale=scale, 
        date_chunks=100
    )
    print('Data extraction complete!')

    dineof_in = outuri # NOTE: need to update data transfer if output to GCS
    dineof_out = f'{band}_dineof_out.nc'

    print('Running DINEOF...')
    exec_dineof(
        dineof_in,
        band,
        min_modes=min_modes,
        max_modes=max_modes,
    )
    print('DINEOF complete!')

    bucket_path = f'ee-dineof-demo/dineof_{band}'
    outbucket = f'{output_bucket}{bucket_path}'

    print('Unpacking DINEOF output as tifs...')
    # unpack the DINEOF output as COGs
    dineof_to_tif(
        dineof_out, 
        dineof_in, 
        outbucket,
        band,
    )
    print('Finished writing DINEOF output to tifs!')

    prefix = f'{bucket_path}/'
    print(prefix)

    print('Starting to add tifs to collection...')
    # add the output COGs to an EE ImageCollection
    cogs_to_collection(output_bucket, prefix, project, out_collection)
    print('Done adding tifs to collection...')

    print('ALL DONE!')

    return
    

def main():
    
    project_ = os.getenv('PROJECT', default=None)
    output_bucket_ = os.getenv('OUTPUT_BUCKET', default=None)
    region_ = os.getenv('REGION', default=None)
    starttime_ = os.getenv('STARTTIME', default = '2018-10-01')
    endtime_ = os.getenv('ENDTIME', default = '2019-10-01')
    band_ = os.getenv('BAND', default = 'sst')
    scale_ = os.getenv('SCALE', default=None)
    out_collection_ = os.getenv('OUTCOLLECTION', default = 'test_out_collection')
    min_modes = int(os.getenv('MINMODES', default = '10'))
    max_modes = int(os.getenv('MAXMODES', default = '20'))
    task_index_ = int(os.getenv('CLOUD_RUN_TASK_INDEX', default = '1'))
    logging.debug(f'task index: {task_index_}')

    region_ = [float(x.strip()) for x in region_.split(',')]

    run_job(
        project_,
        output_bucket_,
        region_,
        starttime_, 
        endtime_, 
        band_, 
        out_collection_,
        min_modes=min_modes,
        max_modes=max_modes,
        scale=scale_
    )


if __name__ == "__main__":
    main()