from pathlib import Path
import logging

import numpy as np
import xarray as xr
import rioxarray
import fsspec


def netcdf_to_tif(da, outbucket, outprefix='dineof_out', dim='time', dim_labels=None):

  if dim_labels is None:
    dim_labels = da[dim].values


  for i in range(len(dim_labels)):
    uri = f"{outbucket}/{outprefix}_{dim_labels[i]}.tif"
    with fsspec.open(uri, 'wb') as f:
      # save the data as a cog
      da.isel(**{dim: i}).rio.to_raster(
          f,
          driver="COG",  # set the driver to be Cloud-Optimized GeoTIFF
          tiled=True, # GDAL: By default striped TIFF files are created. This option can be used to force creation of tiled TIFF files.
          windowed=False,  # rioxarray: read & write one window at a time,
          overviews='auto',  # auto generate internal overviews if not available
          blocksize=256,  # set size of tiles to 256x256
          compress='DEFLATE',  # LZW compression, use LZW or DEFLATE
          level=9,  # level 9 compression (highest)
      )

  return

def dineof_to_tif(dineof_out, dineof_in, outbucket, variable, x_dim='dim001', y_dim='dim002', time_dim='dim003', ):
  dineof_out = xr.open_dataset(dineof_out)
  original = xr.open_dataset(dineof_in)

  dim_labels = original['time'].dt.strftime('%Y%m%d').values

  da = dineof_out[variable]
  # update the DataArray coordinate info to contain geographic information
  da = da.rename({y_dim:'y', x_dim:'x'})
  da['y'] = original['y']
  da['x'] = original['x']
  #set the CRS, use the domain info to set
  da.rio.write_crs('EPSG:4326', inplace=True)
  da.rio.write_nodata(9999, inplace=True)

  netcdf_to_tif(da, outbucket, dim=time_dim, dim_labels=dim_labels)

def main():

  return


if __name__ == "__main__":
  main()