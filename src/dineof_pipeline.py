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

def get_land_mask(region, crs, scale):
    land_mask = (
        ee.Image('MODIS/MOD44W/MOD44W_005_2000_02_24')
        .select('water_mask')
        .unmask(1)
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

    da.attrs = dict(long_name = "Land mask")
    da.x.attrs = dict(units = "degrees_east", long_name = "Longitude")
    da.y.attrs = dict(units = "degrees_north", long_name = "Latitude")

    return da

def filter_poor_observations(da):
    x = (da > 0).sum(dim=["x", "y"])
    y = x / (da.shape[1] * da.shape[2])
    keep_dates = y.where(y > 0.05).dropna(dim="time").time
    return da.sel(time=keep_dates)

def get_data_tile(region, start_time, end_time, crs="EPSG:4326", scale=1000, date_chunks=5):
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
        ee.ImageCollection("NASA/OCEANDATA/MODIS-Terra/L3SMI")
        .filterDate(start_time, end_time)
        .filterBounds(region)
        .select('chlor_a')
    )

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
            name = "chlor_a",
        )
        da.attrs = dict(units = "mg m^-3", long_name = "Chlorophyll a concentration", _FillValue = -999)
        da.x.attrs = dict(units = "degrees_east", long_name = "Longitude")
        da.y.attrs = dict(units = "degrees_north", long_name = "Latitude")

        tile_batches.append(da)

    land_da = get_land_mask(region, crs, scale)

    out_ds = xr.concat(tile_batches, dim="time")
    out_ds = out_ds.sortby("time")
    out_ds = filter_poor_observations(out_ds)

    out_ds = xr.merge([out_ds, land_da])

    return out_ds


def main():
    bbox = [-91.8017, 23.5289, -79.8266,30.7750]
    region = ee.Geometry.Rectangle(bbox)
    start_time = '2010-01-01'
    end_time = '2011-12-31'

    da = get_data_tile(region, start_time, end_time, scale=5000, date_chunks=50)

    da.to_netcdf('test.nc')

if __name__ == "__main__":
    main()