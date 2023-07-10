import ee
import io
import google.auth
import numpy as np
import xarray as xr
# import apache_beam as beam

credentials, project_id = google.auth.default()
ee.Initialize(credentials, project=project_id)

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

def get_mask(region, crs, scale):
    land_mask = (
        ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
        .select('occurrence')
        .unmask(0)
        .lt(75)
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
        name = "mask",
    )

    da.attrs = dict(long_name = "Land mask")
    da.x.attrs = dict(units = "degrees_east", long_name = "Longitude")
    da.y.attrs = dict(units = "degrees_north", long_name = "Latitude")

    return da

def filter_poor_observations(da):
    x = (da > 0).sum(dim=["x", "y"])
    y = x / (da.NDVI.shape[1] * da.NDVI.shape[2])
    keep_dates = y.where(y > 0.05).dropna(dim="time").time
    return da.sel(time=keep_dates)

def mask_lc8(image):
    # Bit 0 - Fill
    # Bit 1 - Dilated Cloud
    # Bit 2 - Cirrus
    # Bit 3 - Cloud
    # Bit 4 - Cloud Shadow
    qaMask = image.select('QA_PIXEL').bitwiseAnd(int('11111', 2)).eq(0)
    saturationMask = image.select('QA_RADSAT').eq(0)

    # Apply the scaling factors to the appropriate bands.
    opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0)

    # calculate NDVI
    ndvi = (
        opticalBands.select('SR_B5').subtract(opticalBands.select('SR_B4'))
        .divide(opticalBands.select('SR_B5').add(opticalBands.select('SR_B4')))
        .rename('NDVI')
    )

    ndvi = ndvi.updateMask(ndvi.gte(-1).And(ndvi.lte(1)))

    # Replace the original bands with the scaled ones and apply the masks.
    return (
        image.addBands(opticalBands, None, True)
        # .addBands(thermalBands, None, True)
        .addBands(ndvi, None, True)
        .updateMask(qaMask)
        .updateMask(saturationMask)
    ).select(['SR_B.*', 'NDVI'])

def get_data_tile(region, start_time, end_time, crs="EPSG:4326", scale=30, date_chunks=5):
    def daily_mosaic(date):
        date = ee.Date(date)
        return (
            dataset
            .filterDate(date, date.advance(1, 'day'))
            .mean()
            .unmask(-999)
            .set('system:time_start', date.millis())
        )

    dataset = (
        # Map the function over one year of data.
        ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
        .filterDate(start_time, end_time)
        .filterBounds(region.centroid())
        .map(mask_lc8)
    )

    dates = (
        dataset.aggregate_array('system:time_start')
        .map(lambda x: ee.Date(x).format('YYYY-MM-dd'))
        .distinct()   
    ).getInfo()

    x_coords, y_coords, grid = get_geo_info(region, crs, scale)

    if crs == 'EPSG:4326':
        x_name, y_name = "x", "y"
        x_long, y_long = "Longitude", "Latitude"
        x_units, y_units = "degrees_east", "degrees_north"

    else:
        x_name, y_name = (
            "x",
            "y",
        )
        x_long, y_long = "Eastings", "Northings"
        # assumes all non-geographic projections have m units...
        x_units, y_units = "meters", "meters"

    bandnames = dataset.first().bandNames().getInfo()

    tiles = []
    ref_date = min(dates)
    for date in dates:

        request_image = daily_mosaic(date)

        # Make a request object.
        batch_request = {
            'expression': request_image,
            'fileFormat': 'NPY',
            'grid': grid
        }

        response = ee.data.computePixels(batch_request)
        data = (
            np.load(io.BytesIO(response))
        )

        # CF conventions are coordinates for center pixels
        # assign domain coordinates and shift to center
        coords = {
            x_name: (
                [x_name],
                x_coords,
                {"units": x_units, "long_name": x_long},
            ),
            y_name: (
                [y_name],
                y_coords,
                {"units": y_units, "long_name": y_long},
            ),
        }

        data_dict = {band: ([y_name, x_name], data[band]) for band in bandnames}

        ds = xr.Dataset(data_dict, coords=coords)
        for band in bandnames:
            ds[band].attrs = dict(_FillValue = -999.0)

        tiles.append(ds)

    out_ds = xr.concat(tiles, dim="time")
    out_ds = out_ds.update({"time": dates})
    out_ds = out_ds.sortby("time").astype(np.float32)
    out_ds = filter_poor_observations(out_ds)

    mask = get_mask(region, crs, scale)

    out_ds = xr.merge([out_ds, mask])

    return out_ds


def main():
    bbox = [-111.958046, 40.014990, -111.496620, 40.374794]
    region = ee.Geometry.Rectangle(bbox)
    start_time = '2017-01-01'
    end_time = '2022-01-01'

    da = get_data_tile(region, start_time, end_time, scale=90, date_chunks=50)

    da.to_netcdf('test_landsat.nc')

if __name__ == "__main__":
    main()