import datetime
import json
import requests
from functools import partial
import logging
from concurrent.futures import ThreadPoolExecutor

import ee
from google.cloud import storage
from google.auth.transport.requests import AuthorizedSession

from ee_auth import get_ee_credentials

def validate_ee_out_collection(project, collection):
    ee_initialize(project)

    collection_asset = f'projects/{project}/assets/{collection}'
    
    # create a new image collection
    # this is not necessary if one already exists
    try:
        ee.data.createAsset({'type':'ImageCollection'}, collection_asset)
        logging.info(f"created asset '{collection_asset}'")

    except ee.EEException as e:
        if 'Cannot overwrite asset' in str(e):
            logging.warn(f"Asset '{collection_asset}' already exists, skipping creation...")
        else:
            logging.error(e)

    return


def cog_to_eecollection(uri, project=None, collection=None):

    credentials = get_ee_credentials()
    session = AuthorizedSession(credentials.with_quota_project(project))

    cog_asset_endpoint = 'https://earthengine.googleapis.com/v1/projects/{}/assets?assetId={}'

    collection_asset = f'projects/{project}/assets/{collection}'
    
    img_name = uri.split('/')[-1].split('.')[0]

    asset_id = f'{collection}/{img_name}'
    print(f'making asset {asset_id}')

    start_time = datetime.datetime.strptime(img_name.split('_')[-1],'%Y%m%d')
    end_time = start_time + datetime.timedelta(seconds=86399)

    # Request body as a dictionary.
    request = {
        'type': 'IMAGE',
        'cloudStorageLocation': {
            'uris': [uri]
        },
        'properties': {
            'processing_version': 'experimental',
            'source': 'dineof',
        },
        'startTime': start_time.strftime('%Y-%m-%dT%H:%M:%SZ'), # can programmatically change date information
        'endTime': end_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
    }

    response = session.post(
        url = cog_asset_endpoint.format(project, asset_id),
        data = json.dumps(request)
    )

    logging.debug(json.loads(response.content))

    return

def list_objects(bucket, prefix):
    """Function to list tif data on GCS"""

    storage_client = storage.Client()
    print("list objects: ,", prefix)

    blobs = storage_client.list_blobs(bucket, prefix=prefix, delimiter='/')
    blob_names = [f'gs://{bucket}/{blob.name}' for blob in blobs if blob.name.endswith('.tif')]

    return blob_names


def cogs_to_collection(inbucket, prefix, project, collection):
    if inbucket.startswith('gs://'):
       inbucket = inbucket.replace('gs://','') 

    if inbucket.endswith('/'):
       inbucket = inbucket[:-1]

    sources = list_objects(inbucket, prefix)

    request_func = partial(cog_to_eecollection, project=project, collection=collection)

    validate_ee_out_collection(project, collection)

    # create a multithreading object and apply the function to request data
    with ThreadPoolExecutor(max_workers=4) as executor:
        gen = executor.map(request_func, sources)

        _ = tuple(gen)
   #  cog_to_eecollection(source, project=project, collection=collection)

    return


def main():

#   inbucket = 'sfwmd_ml-ai'
#   prefix = f'experimental/sst/'

#   for source in sources:
#     cog_to_eecollection(source, project='sfwmd-gee-dev-gis', collection='ML/experimental_sst_dineof')

  return


if __name__ == "__main__":
  main()