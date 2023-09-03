import os
import re
import time
import math
import json

from pyspark.sql import DataFrame
from .geo_llm import GeoLLM


class GeoAgent:

    def __init__(self, geo_llm: GeoLLM, output_dir: str = "./"):
        self._df = None
        self._geo_llm = geo_llm
        self._output_dir = output_dir
        self._tools = ["transform_df", "create_geo_bins", "plot_df_geo_bins", "analyze_s3_dataset"]

    @staticmethod
    def _extract_s3_url(desc: str) -> str | None:
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
        if self._df is None:
            return False

        head1 = self._df.take(1)[0]
        head2 = df.take(1)[0]
        for name, dtype in df.dtypes:
            if head1[name] is not None and head2[name] is None:
                return False
        return True

    def chat(self, message: str, server_url: str) -> str:
        tool = self._geo_llm.select_tool(message)
        print(tool)
        match tool:
            case 'analyze_s3_dataset':
                s3_url = self._extract_s3_url(message)
                if s3_url is None:
                    return f'{{"success":"False", "agent_message":"Required s3 url is invalid: {s3_url}"}}'
                else:
                    print(s3_url)
                    df = self._geo_llm.create_s3_df(s3_url)
                    if df is None:
                        return \
                            (f'{{"success":"False", "agent_message": '
                             f'"Failed to load CSV data from the given url: {s3_url}"}}')
                    else:
                        self._df = df
                        response = self._geo_llm.analyze_s3_dataset(df)
                        print(f'analyze_s3_dataset: {response}')
                        response_json = json.loads(response)
                        # to change JSON item quote from single to double
                        col_names_types = json.dumps(response_json["col_data_types"])
                        samples = json.dumps(response_json["samples"])
                        col_descriptions = json.dumps(response_json["col_descriptions"])
                        if response_json['success'] == 'True':
                            message = (f'A PySpark Dataframe has been successfully created '
                                       f'based on the given S3 data source: {s3_url}. '
                                       f'Following is a brief description of each column and a few sample records.')
                            return (f'{{"success":"True", "agent_message":"{message}",'
                                    f'"column_descriptions":{col_descriptions},'
                                    f'"column_data_types":{col_names_types},'
                                    f'"samples":{samples}}}')
                        else:
                            return f'{{"success":"False", "agent_message":"{response_json["error_msg"]}"}}'

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
                        return (f'{{"success": "True", "agent_message":"Your request for transforming the data '
                                f'has been successfully executed."}}')
                    else:
                        return (f'{{"success": "False", '
                                f'"agent_message":"Could not transform the data. Quick check failed!'
                                f' Please refine your message and try again!"}}')
                except Exception as e:
                    print(type(e))
                    print(e)
                    return (f'{{"success": "False", "agent_message":"Could not transform the data: {e}. '
                            f'Please refine your message and try again!"}}')

            case 'create_geo_bins':
                try:
                    ts = math.trunc(time.time())
                    file_name = f'geojson_{ts}.json'
                    geojson_output_file = f'{self._output_dir}/{file_name}'
                    self._geo_llm.create_geo_bins(self._df, geojson_output_file, message)
                    if not os.path.isfile(geojson_output_file):
                        return (f'{{"success": "False", "agent_message":"geo-binning data may have failed '
                                f'since there is no JSON output. Please refine your message and try again!"}}')

                    return (f'{{"success": "True", '
                            f'"agent_message":"Your request for performing geo-binning has been successfully executed.",'
                            f'"url":"{server_url}/output_data/{file_name}"}}')
                    # with open(geojson_output_file, 'r') as f:
                    #     geojson = f.read()
                    #     print(len(geojson))
                    #     print(geojson[0:500])  # print out few records
                    #     return geojson

                except Exception as e:
                    return (f'{{"success": "False", "agent_message":"Could not perform geo-binning the data: {e}. '
                            f'Please refine your message and try again!"}}')

            case _:
                return (f'{{"success": "False", "agent_message":"Could not find a tool matching your description. '
                        f'Please refine your message and try again!"}}')
