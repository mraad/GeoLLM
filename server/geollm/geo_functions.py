import geofunctions as S
import pyspark.sql.functions as F
import geopandas as gp
import json

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
        # the function names listed here must match the names in the geo_function_definitions dictionary!
        return {
            "create_square_bins": self.create_square_bins,
        }

    def available_validation_functions(self):
        # the function names listed here must match the names in the geo_function_definitions dictionary!
        return {
            "create_square_bins": self.validate_lat_lon_columns,
        }

    def validate_lat_lon_columns(self, parameters_json: str) -> str | None:
        # first check if the lat/lon columns extracted columns from Dataframe
        column_json = json.loads(parameters_json)
        lat_col = column_json['lat_column']
        lon_col = column_json['lon_column']
        lat_found = False
        lon_found = False
        for name, dtype in self._df.dtypes:
            if name == lat_col and (dtype == 'double' or dtype == 'float'):
                lat_found = True
            if name == lon_col and (dtype == 'double' or dtype == 'float'):
                lon_found = True
        if lat_found and lon_found:
            return parameters_json

        print(f'try to direct extracting x/y columns from column names and types!')
        # extracted lat/lon columns not found from Dataframe based on column name and type
        x_col_candidate = None
        x_col_candidate_priority = 10000000
        y_col_candidate = None
        y_col_candidate_priority = 10000000

        for col_name, col_type in self._df.dtypes:
            if col_type == 'double' or col_type == 'float':
                if col_name == 'X' or col_name == 'x':
                    x_col_candidate = col_name
                    x_col_candidate_priority = 0
                elif col_name.startswith("X") or col_name.startswith("x"):
                    current_priority = len(col_name)
                    if x_col_candidate is None or current_priority < x_col_candidate_priority:
                        x_col_candidate = col_name
                        x_col_candidate_priority = current_priority
                elif col_name == 'Longitude' or col_name == 'longitude':
                    current_priority = 1
                    if x_col_candidate is None or current_priority < x_col_candidate_priority:
                        x_col_candidate = col_name
                        x_col_candidate_priority = current_priority
                elif col_name.startswith("Lon") or col_name.startswith("lon"):
                    current_priority = len(col_name)
                    if x_col_candidate is None or current_priority < x_col_candidate_priority:
                        x_col_candidate = col_name
                        x_col_candidate_priority = current_priority

                if col_name == 'Y' or col_name == 'y':
                    y_col_candidate = col_name
                    y_col_candidate_priority = 0
                elif col_name.startswith("Y") or col_name.startswith("y"):
                    current_priority = len(col_name)
                    if y_col_candidate is None or current_priority < y_col_candidate_priority:
                        y_col_candidate = col_name
                        y_col_candidate_priority = current_priority
                elif col_name == 'Latitude' or col_name == 'latitude':
                    current_priority = 1
                    if y_col_candidate is None or current_priority < y_col_candidate_priority:
                        y_col_candidate = col_name
                        y_col_candidate_priority = current_priority
                elif col_name.startswith("Lat") or col_name.startswith("lat"):
                    current_priority = len(col_name)
                    if y_col_candidate is None or current_priority < y_col_candidate_priority:
                        y_col_candidate = col_name
                        y_col_candidate_priority = current_priority

        if x_col_candidate is not None and y_col_candidate is not None:
            column_json['lon_column'] = x_col_candidate
            column_json['lat_column'] = y_col_candidate
            return json.dumps(column_json)

        return None
