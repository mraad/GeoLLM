import os
import re
import time
import math
import json

import urllib.parse
from pyspark.sql import DataFrame
from pydantic import BaseModel
from .geo_llm import GeoLLM


class Message(BaseModel):
    id: str
    message: str


class GeoAgent:

    def __init__(self, geo_llm: GeoLLM, output_dir: str = "./"):
        self._df = None
        self._geo_llm = geo_llm
        self._output_dir = output_dir
        self._tools = ["transform_dataframe", "create_square_bins", "create_dataframe_from_s3"]

    @staticmethod
    def _extract_s3_url(desc: str) -> str | None:
        # Regular expression pattern for S3 URL extraction
        pattern = r's3a://[a-zA-Z0-9\-\.]+/[a-zA-Z0-9\-_\/\.]+'
        # Extract the S3 URL
        s3_url = re.search(pattern, desc)
        if s3_url:
            s3_url = s3_url.group(0)
            print(s3_url)
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

    def _execute_transform_df(self, message: str) -> str:
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
            print(e)
            return (f'{{"success": "False", "agent_message":"Could not transform the data: {e}. '
                    f'Please refine your message and try again!"}}')

    def _execute_create_dataframe(self, decoded_message: str, generate_column_description: bool = False) -> str:
        s3_url = self._extract_s3_url(decoded_message)
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
                # print(self._geo_llm._get_col_info_json(df.dtypes))
                if generate_column_description:
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
                else:
                    return f'{{"success":"True", "agent_message":"Dataframe from S3 successfully created!"}}'

    def _execute_create_geo_bins(self, decoded_message: str, server_url: str) -> str:
        try:
            ts = math.trunc(time.time())
            file_name = f'geojson_{ts}.json'
            geojson_output_file = f'{self._output_dir}/{file_name}'
            self._geo_llm.create_square_bins(self._df, geojson_output_file, decoded_message)
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

    def _auto_transform_date_columns(self, message: str) -> str:
        # a hack to cast a string 'Date' column to 'timestamp' since the 'str' type wouldn't work for
        # SQL query for 'date time' and user at this point has no knowledge of the data types
        # of loaded data we are executing multiple tools in one request!!
        if self._df is None:
            return f'{{"success": "False", "agent_message":"Required Dataframe is null."}}'

        sql_query_str = self._geo_llm.get_sql_code(self._df, message)
        if sql_query_str is None or sql_query_str == '':
            return f'{{"success": "False", "agent_message":"There is no SQL query string."}}'

        sql_query_str = sql_query_str.replace("\n", " ")
        print(f'_auto_transform_date_columns -> {sql_query_str}')
        # any column name contains 'Date', 'time', or 'timestamp'.
        # also needs to get one sample record from DataFrame to attach to
        # decoded_message to help LLM to come up with right SQL query!
        possible_date_time_col_names = ["Date", "date", "time", "Time", "Timestamp", "timestamp"]
        items = sql_query_str.split(" ")
        print(items)
        head = self._df.take(1)[0]
        for name, dtype in self._df.dtypes:
            if name in items:
                for x in possible_date_time_col_names:
                    if x in name:
                        sample_value = head[name]
                        if dtype == 'string':
                            message = (f'please transform a string {name} column to a timestamp column. '
                                       f'Here is the sample value: {sample_value}')
                            print(message)
                            return self._execute_transform_df(message)
                        elif dtype == 'int':
                            message = (f'please transform an integer {name} column to a timestamp column.'
                                       f'Here is the sample value: {sample_value}')
                            print(message)
                            return self._execute_transform_df(message)

        return (f'{{"success": "True", "agent_message":"{sql_query_str} '
                f'does not contain any possible date/time column."}}')

    def chat2(self, message: Message, server_url: str) -> str:
        decoded_message = urllib.parse.unquote(message.message)
        print(decoded_message)
        tools_str = self._geo_llm.select_tool2(decoded_message)
        print(f'chat2 tools: {tools_str}')
        tools = json.loads(tools_str)

        if tools is None or len(tools['tools']) == 0:
            return (f'{{"success": "False", "agent_message":"Could not find a tool matching your description. '
                    f'Please refine your message and try again!"}}')

        pattern = r'"(.*?)"'
        final_response = ''
        for tool in tools['tools']:
            print(f'tool -> {tool}')
            match tool:
                case 'create_dataframe_from_s3':
                    response = self._execute_create_dataframe(decoded_message)
                    response_json = json.loads(response)
                    if response_json['success'] == 'False':
                        return response
                    else:
                        final_response = response

                case 'transform_dataframe':
                    message = (f'please extract sentences that may require SQL query from the following input:\n'
                               f'{decoded_message}')
                    extract_message = self._geo_llm.extract_info(message)
                    extract_message = re.findall(pattern, extract_message)
                    if len(extract_message) > 0:
                        extract_message = extract_message[0]
                    print(f'extracted SQL query message: {extract_message}')

                    # in case the transformation is on a possible date column but
                    # the column type is a 'str' or 'int'
                    auto_date_transform_response = self._auto_transform_date_columns(extract_message)
                    print(auto_date_transform_response)
                    auto_date_transform_response_json = json.loads(auto_date_transform_response)
                    if auto_date_transform_response_json['success'] == 'False':
                        return auto_date_transform_response

                    response = self._execute_transform_df(extract_message)
                    response_json = json.loads(response)
                    if response_json['success'] == 'False':
                        return response
                    else:
                        final_response = response

                case 'create_square_bins':
                    final_response = self._execute_create_geo_bins(decoded_message, server_url)

        return final_response

    def chat(self, message: Message, server_url: str) -> str:
        decoded_message = urllib.parse.unquote(message.message)
        print(decoded_message)
        tool = self._geo_llm.select_tool(decoded_message)
        print(tool)
        match tool:
            case 'create_dataframe_from_s3':
                return self._execute_create_dataframe(decoded_message, True)

            case 'transform_dataframe':
                return self._execute_transform_df(decoded_message)

            case 'create_square_bins':
                return self._execute_create_geo_bins(decoded_message, server_url)

            case _:
                return (f'{{"success": "False", "agent_message":"Could not find a tool matching your description. '
                        f'Please refine your message and try again!"}}')
