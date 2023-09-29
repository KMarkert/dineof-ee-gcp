import ee
import io
import math
import google.auth
import numpy as np
import xarray as xr
# import apache_beam as beam
import logging
import fsspec

from config import config

credentials, project_id = google.auth.default()
ee.Initialize(credentials, project=config.EE_PROJECT)

def get_geo_info(region, crs, scale):
    coords = region.coordinates().getInfo()
    # upper left point
    ul_pt = coords[0][3]
    lr_pt = coords[0][1]

    # Make a projection to discover the scale in degrees.
    proj = ee.Projection(crs).atScale(scale).getInfo()

    # Get scales out of the transform.
    scale_x = proj['transform'][0]
    scale_y = -proj['transform'][4]

    x_coords = np.arange(ul_pt[0], lr_pt[0], scale_x)
    y_coords = np.arange(ul_pt[1], lr_pt[1], scale_y)

    # adjust the coordinates to center of the pixel
    # netcdf convention is to have the coordinates at the center of the pixel
    x_coords = x_coords + (scale_x / 2)
    y_coords = y_coords - (scale_y / 2)

    grid = {
        'dimensions': {
            'width': x_coords.size,
            'height': y_coords.size,
        },
        'affineTransform': {
            'scaleX': scale_x,
            'shearX': 0,
            'translateX': ul_pt[0],
            'shearY': 0,
            'scaleY': scale_y,
            'translateY': ul_pt[1]
        },
        'crsCode': proj['crs'],
    }

    return x_coords, y_coords, grid

def get_land_mask(region, crs, scale):
    land_mask = (
        ee.Image('projects/sfwmd-gee-dev-gis/assets/ML/LakeO_Mask')
    )
    x_coords, y_coords, grid = get_geo_info(region, crs, scale)
    request = {
            'expression': land_mask,
            'fileFormat': 'NPY',
            'grid': grid,
        }

    response = ee.data.computePixels(request)
    data = (
        np.load(io.BytesIO(response))
        .view(np.uint8)
        .reshape(len(y_coords), len(x_coords),)
    )

    da = xr.DataArray(
        data=data,
        dims=["y", "x",],
        coords=dict(
            x=(["x",], x_coords),
            y=(["y",], y_coords),
        ),
        name = "land_mask",
    )

    da.attrs = dict(long_name = "Land mask", units="N/A", description="1 = Run Grid Cell, 0 = Do Not Run",)
    da.x.attrs = dict(units = "degrees_east", long_name = "Longitude")
    da.y.attrs = dict(units = "degrees_north", long_name = "Latitude")

    return da


def filter_poor_observations(da, threshold=0.05):
    x = (da > 0).sum(dim=["x", "y"])
    y = x / (da.shape[1] * da.shape[2])
    keep_dates = y.where(y > threshold).dropna(dim="time").time
    return da.sel(time=keep_dates)


def mask_cicyano(img):
    return img.updateMask(img.lt(250))


def get_cicyano(start_time, end_time, region):
    olci_a = (
        ee.ImageCollection("projects/ce-datasets/assets/ce-noaa-nccos-hab/sentinel-3a")
        .filterDate(start_time, end_time)
        .filterBounds(region)
    )
            
    olci_b = (
        ee.ImageCollection("projects/ce-datasets/assets/ce-noaa-nccos-hab/sentinel-3b")
        .filterDate(start_time, end_time)
        .filterBounds(region)
    )

    return (
        olci_a.merge(olci_b)
        .map(mask_cicyano)
        .select('cicyano')
    )

def extract_bits(image, start, end=None, new_name=None):
    """Function to convert qa bits to binary flag image

    args:
        image (ee.Image): qa image to extract bit from
        start (int): starting bit for flag
        end (int | None, optional): ending bit for flag, if None then will only use start bit. default = None
        new_name (str | None, optional): output name of resulting image, if None name will be {start}Bits. default = None

    returns:
        ee.Image: image with extract bits
    """

    newname = new_name if new_name is not None else f"{start}_bits"

    if (start == end) or (end is None):
        # perform a bit shift with bitwiseAnd
        return image.select([0], [newname]).bitwiseAnd(1 << start)
    else:
        # Compute the bits we need to extract.
        pattern = 0
        for i in range(start, end):
            pattern += int(math.pow(2, i))

        # Return a single band image of the extracted QA bits, giving the band
        # a new name.
        return image.select([0], [newname]).bitwiseAnd(pattern).rightShift(start)


def preprocess_sst(image):
    qa_band = image.select("QC_Day")

    mask = extract_bits(qa_band, start=2, end=3).eq(0)

    return image.multiply(0.02).subtract(273.15).updateMask(mask).copyProperties(image,["system:time_start"])


def get_sst(start_time, end_time, region):
    # NOTE: using Aqua LST product with 1:30PM local equatorial overpass
    return (
        ee.ImageCollection("MODIS/061/MYD11A1") 
        .filterDate(start_time, end_time)
        .filterBounds(region)
        .map(preprocess_sst)
        .select(["LST_Day_1km"],['sst'])
    )


def get_data_tile(region, start_time, end_time, collection='cicyano', crs="EPSG:4326", scale=1000, date_chunks=5):
    def daily_mosaic(date):
        date = ee.Date(date)
        return (
            dataset
            .filterDate(date, date.advance(1, 'day'))
            .mean()
            .unmask(-999)
            .set('system:time_start', date.millis())
        )

    if collection not in ['cicyano', 'sst']:
        raise NotImplementedError('available options for ')

    if collection == 'cicyano':
        dataset = get_cicyano(start_time, end_time, region)

    elif collection == 'sst':
        dataset = get_sst(start_time, end_time, region)

    dates = (
        dataset.aggregate_array('system:time_start')
        .map(lambda x: ee.Date(x).format('YYYY-MM-dd'))
        .distinct()   
    ).getInfo()

    # get the tile for the first date
    batch_dates = [dates[i::date_chunks] for i in range(date_chunks)]

    x_coords, y_coords, grid = get_geo_info(region, crs, scale)

    tile_batches = []
    ref_date = min(dates)
    for batch in batch_dates:

        request_image = (
            ee.ImageCollection(ee.List(batch).map(daily_mosaic))
            .toBands()
            .toFloat()
            .rename(batch)
        )

        # Make a request object.
        batch_request = {
            'expression': request_image,
            'fileFormat': 'NPY',
            'grid': grid
        }

        response = ee.data.computePixels(batch_request)
        data = (
            np.load(io.BytesIO(response))
            .view(np.float32)
            # .astype(np.float32)
            .reshape(len(y_coords), len(x_coords), len(batch))
            .transpose(2, 0, 1)
        )

        da = xr.DataArray(
            data=data,
            dims=["time", "y", "x",],
            coords=dict(
                x=(["x",], x_coords),
                y=(["y",], y_coords),
                time=np.array(batch, dtype=np.datetime64),
                reference_time=ref_date,
            ),
            name = collection,
        )
        da.attrs = dict(_FillValue = -999)
        da.x.attrs = dict(units = "degrees_east", long_name = "Longitude")
        da.y.attrs = dict(units = "degrees_north", long_name = "Latitude")

        tile_batches.append(da)

    land_da = get_land_mask(region, crs, scale)

    out_ds = xr.concat(tile_batches, dim="time")
    out_ds = out_ds.sortby("time")
    out_ds = filter_poor_observations(out_ds, threshold=0.02)

    out_ds = xr.merge([out_ds, land_da])

    return out_ds

def extract_ee_data(outpath, region, start_time, end_time, collection='cicyano', crs="EPSG:4326", scale=300, date_chunks=50):

    if not isinstance(region, ee.Geometry):
        region = ee.Geometry.Rectangle(region)
    
    da = get_data_tile(region, start_time, end_time, collection=collection, scale=scale, date_chunks=date_chunks)
    print(da)

    # with fsspec.open(outpath, 'w') as f:
    da.to_netcdf(outpath, engine="netcdf4")

    return


def main():
    bbox = (-81.106114, 26.681248, -80.611270, 27.207278)
    region = ee.Geometry.Rectangle(bbox)
    start_time = '2016-01-01'
    end_time = '2023-12-31'




if __name__ == "__main__":
    main()