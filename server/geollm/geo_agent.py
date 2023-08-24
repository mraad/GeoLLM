import os
import re
from pyspark.sql import DataFrame

from .geo_llm import GeoLLM


class GeoAgent:

    def __init__(self, geo_llm: GeoLLM):
        self._df = None
        self._geo_llm = geo_llm
        self._tools = ["transform_df", "create_geo_bins", "plot_df_geo_bins", "analyze_s3_dataset"]

    @staticmethod
    def _extract_s3_url(desc: str) -> str:
        # Regular expression pattern for S3 URL extraction
        pattern = r's3a://[a-zA-Z0-9\-\.]+/[a-zA-Z0-9\-_\/\.]+'
        # Extract the S3 URL
        s3_url = re.search(pattern, desc)
        if s3_url:
            s3_url = s3_url.group(0)
            if s3_url[-1] in '.!@#$%^&*()':
                s3_url = s3_url[:-1]
        else:
            s3_url = None

        return s3_url

    def _quick_check_transformed_df(self, df: DataFrame) -> bool:
        # compare heads records of 2 DataFrames
        head1 = self._df.take(1)[0]
        head2 = df.take(1)[0]
        for name, dtype in df.dtypes:
            if head1[name] is not None and head2[name] is None:
                return False
        return True

    def chat(self, message: str) -> str:
        tool = self._geo_llm.select_tool(message)
        print(tool)
        match tool:
            case 'analyze_s3_dataset':
                s3_url = self._extract_s3_url(message)
                if s3_url is None:
                    return f'{{"success":"False", "error_msg":"Required s3 url is invalid: {s3_url}"}}'
                else:
                    print(s3_url)
                    df = self._geo_llm.create_s3_df(s3_url)
                    if df is None:
                        return \
                            f'{{"success":"False", "error_msg":"Failed to load CSV data from the given url: {s3_url}"}}'
                    else:
                        self._df = df
                        return self._geo_llm.analyze_s3_dataset(df)

            case 'transform_df':
                try:
                    loop = 0
                    quick_check = False
                    df = None
                    while not quick_check and loop < 3:
                        loop += 1
                        df = self._geo_llm.transform_df(self._df, message)
                        quick_check = self._quick_check_transformed_df(df)

                    if quick_check and df is not None:
                        self._df = df
                        return f'{{"success": "True", "error_msg":""}}'
                    else:
                        return (f'{{"success": "False", "error_msg":"Could not transform the data. Please refine your '
                                f'message and try again!"}}')
                except Exception as e:
                    return (f'{{"success": "False", "error_msg":"Could not transform the data: {e}. Please refine your '
                            f'message and try again!"}}')

            case 'create_geo_bins':
                try:
                    self._geo_llm.create_geo_bins(self._df, message)
                    file_name = '../output/geo_bins.json'
                    if not os.path.isfile(file_name):
                        file_name = './geo_bins.json'
                        if not os.path.isfile(file_name):
                            return (f'{{"success": "False", "error_msg":"geo-binning data may have failed '
                                    f'since there is no JSON output. Please refine your message and try again!"}}')

                    with open(file_name, 'r') as f:
                        geojson = f.read()
                        print(len(geojson))
                        print(geojson[0:500])
                        return geojson

                except Exception as e:
                    return (f'{{"success": "False", "error_msg":"Could not perform geo-binning the data: {e}. Please '
                            f'refine your message and try again!"}}')

            case _:
                return (f'{{"success": "False", "error_msg":"Could not find a tool matching your description. Please '
                        f'refine your message and try again!"}}')
