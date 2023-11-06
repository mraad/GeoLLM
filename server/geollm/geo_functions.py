import geofunctions as S
import pyspark.sql.functions as F
import geopandas as gp
from pyspark.sql import DataFrame

geo_function_definitions = [
    {
        'name': 'create_square_bins',
        'description': 'Create a square bin grid from the given data in a dataframe with given parameters',
        'parameters': {
            'type': 'object',
            'properties': {
                'lon_column': {
                    'type': 'string',
                    'description': 'longitude (x) column'
                },
                'lat_column': {
                    'type': 'string',
                    'description': 'latitude (y) column'
                },
                'cell_size': {
                    'type': 'integer',
                    'description': 'binning square size'
                },
                'max_count': {
                    'type': 'integer',
                    'description': 'maximum count of data points in a cell'
                },
                'output_file': {
                    'type': 'string',
                    'description': 'GeoJson output file name'
                }
            }
        }
    }
]


class GeoFunctions:

    def __init__(self, data_frame: DataFrame):
        self._df = data_frame

    def create_square_bins(self, lon_column: str, lat_column: str, cell_size: int,
                           max_count: int, output_file: str) -> str:
        # Perform Spatial Binning
        df = (
            self._df.select(S.st_lontoq(lon_column, cell_size), S.st_lattor(lat_column, cell_size))
            .groupBy("q", "r")
            .count()
            .select(
                S.st_qtox("q", cell_size),
                S.st_rtoy("r", cell_size),
                "count",
            )
            .select(
                S.st_cell("x", "y", cell_size).alias("geometry"),
                F.least("count", F.lit(max_count)).alias("count"),
            )
            .orderBy("count")
        )

        # Create geodataframe to get GeoJSON document
        df = df.toPandas()
        df.geometry = df.geometry.apply(lambda _: bytes(_))
        df.geometry = gp.GeoSeries.from_wkb(df.geometry)
        gdf = gp.GeoDataFrame(df, crs="EPSG:3857")
        gdf = gdf.to_crs(crs="EPSG:4326")
        geo_json = gdf.to_json()

        # Write GeoJSON document to local disk
        with open(output_file, 'w') as f:
            f.write(geo_json)

        return output_file

    def available_functions(self):
        return {
            "create_square_bins": self.create_square_bins,
        }
