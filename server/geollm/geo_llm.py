from json import JSONDecodeError

import pandas as pd  # noqa: F401

import contextlib
import io
import os
import re
import json

from urllib.parse import urlparse
import requests
import tiktoken
from tiktoken import Encoding
from bs4 import BeautifulSoup

from langchain import BasePromptTemplate, GoogleSearchAPIWrapper, LLMChain
from langchain.base_language import BaseLanguageModel
from langchain.chat_models import ChatOpenAI

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import *

from .ai_utils import AIUtils
from .cache import Cache, LLMChainWithCache, SKIP_CACHE_TAGS, SearchToolWithCache
from .code_logger import CodeLogger
from .prompts import EXPLAIN_DF_PROMPT, PLOT_PROMPT, SEARCH_PROMPT, SQL_PROMPT
from .prompts import TRANSFORM_PROMPT, UDF_PROMPT, VERIFY_PROMPT, DATA_ANALYSIS_PROMPT
from .prompts import TOOL_SELECTION_PROMPT, GEOBINS_PLOT_PROMPT, INFO_EXTRACTION_PROMPT
from .prompts import MULTI_TOOLS_SELECTION_PROMPT, CREATE_GEO_BINS_PROMPT
from .temp_view_utils import random_view_name, replace_view_name

# AIUtils
# Cache
# CodeLogger
# SKIP_CACHE_TAGS, LLMChainWithCache
# (
#     EXPLAIN_DF_PROMPT,
#     PLOT_PROMPT,
#     SEARCH_PROMPT,
#     SQL_PROMPT,
#     TRANSFORM_PROMPT,
#     UDF_PROMPT,
#     VERIFY_PROMPT,
#     DATA_ANALYSIS_PROMPT,
#     TOOL_SELECTION_PROMPT,
#     GEOBINS_PLOT_PROMPT,
#     CREATE_GEOBINS_PROMPT,
# )
# SearchToolWithCache
# random_view_name, replace_view_name


geo_function_templates = {
    "create_square_bins":
        """
import geofunctions as S
import pyspark.sql.functions as F
import geopandas as gp

# this template requires following 5 variables to be included before executing
# lon_column, lat_column, cell_size, max_count and output_file

# Perform Spatial Binning
df = (
    df.select(S.st_lontoq(lon_column, cell_size), S.st_lattor(lat_column, cell_size))
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
    """
}


def _check_lat_lon_cols(df: DataFrame, col_json_str: str, cache: bool = True) -> str | None:
    # first check if the lat/lon columns extracted columns from Dataframe
    column_json = json.loads(col_json_str)
    lat_col = column_json['lat_column']
    lon_col = column_json['lon_column']
    lat_found = False
    lon_found = False
    for name, dtype in df.dtypes:
        if name == lat_col and (dtype == 'double' or dtype == 'float'):
            lat_found = True
        if name == lon_col and (dtype == 'double' or dtype == 'float'):
            lon_found = True
    if lat_found and lon_found:
        return col_json_str

    print(f'try to direct extracting x/y columns from column names and types!')
    # extracted lat/lon columns not found from Dataframe based on column name and type
    x_col_candidate = None
    x_col_candidate_priority = 10000000
    y_col_candidate = None
    y_col_candidate_priority = 10000000

    for col_name, col_type in df.dtypes:
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


class GeoLLM:
    _HTTP_HEADER = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
                      " (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    def __init__(
            self,
            llm: Optional[BaseLanguageModel] = None,
            web_search_tool: Optional[Callable[[str], str]] = None,
            spark_session: Optional[SparkSession] = None,
            enable_cache: bool = True,
            cache_file_format: str = "json",
            cache_file_location: Optional[str] = None,
            encoding: Optional[Encoding] = None,
            max_tokens_of_web_content: int = 3000,
            verbose: bool = True,
    ) -> None:
        """
        Initialize the SparkAI object with the provided parameters.

        :param llm: LLM instance for selecting web search result
                                 and writing the ingestion SQL query.
        :param web_search_tool: optional function to perform web search,
                                Google search will be used if not provided
        :param spark_session: optional SparkSession, a new one will be created if not provided
        :param encoding: optional Encoding, cl100k_base will be used if not provided
        :param max_tokens_of_web_content: maximum tokens of web content after encoding
        """
        self._spark = spark_session or SparkSession.builder.getOrCreate()
        if llm is None:
            llm = ChatOpenAI(model_name="gpt-4", temperature=0.0)
        self._llm = llm
        self._web_search_tool = web_search_tool or self._default_web_search_tool
        if enable_cache:
            self._enable_cache = enable_cache
            if cache_file_location is not None:
                # if there is parameter setting for it, use the parameter
                self._cache_file_location = cache_file_location
            elif "AI_CACHE_FILE_LOCATION" in os.environ:
                # otherwise read from env variable AI_CACHE_FILE_LOCATION
                self._cache_file_location = os.environ["AI_CACHE_FILE_LOCATION"]
            else:
                # use default value "spark_ai_cache.json"
                self._cache_file_location = "spark_ai_cache.json"
            self._cache = Cache(
                cache_file_location=self._cache_file_location,
                file_format=cache_file_format,
            )
            self._web_search_tool = SearchToolWithCache(
                self._web_search_tool, self._cache
            ).search
        else:
            self._cache = None
            self._enable_cache = False

        self._encoding = encoding or tiktoken.get_encoding("cl100k_base")
        self._max_tokens_of_web_content = max_tokens_of_web_content
        self._search_llm_chain = self._create_llm_chain(prompt=SEARCH_PROMPT)
        self._sql_llm_chain = self._create_llm_chain(prompt=SQL_PROMPT)
        self._data_analysis_llm_chain = self._create_llm_chain(prompt=DATA_ANALYSIS_PROMPT)
        self._tool_selection_llm_chain = self._create_llm_chain(prompt=TOOL_SELECTION_PROMPT)
        self._tools_selection_llm_chain = self._create_llm_chain(prompt=MULTI_TOOLS_SELECTION_PROMPT)
        self._explain_chain = self._create_llm_chain(prompt=EXPLAIN_DF_PROMPT)
        self._transform_chain = self._create_llm_chain(prompt=TRANSFORM_PROMPT)
        self._plot_chain = self._create_llm_chain(prompt=PLOT_PROMPT)
        self._geo_bins_plot_chain = self._create_llm_chain(prompt=GEOBINS_PLOT_PROMPT)
        # self._create_geo_bins_chain = self._create_llm_chain(prompt=CREATE_GEOBINS_PROMPT)
        self._create_geo_bins_chain = self._create_llm_chain(prompt=CREATE_GEO_BINS_PROMPT)
        self._info_extraction_chain = self._create_llm_chain(prompt=INFO_EXTRACTION_PROMPT)
        self._verify_chain = self._create_llm_chain(prompt=VERIFY_PROMPT)
        self._udf_chain = self._create_llm_chain(prompt=UDF_PROMPT)
        self._verbose = verbose
        if verbose:
            self._logger = CodeLogger("spark_ai")

    def _create_llm_chain(self, prompt: BasePromptTemplate):
        # print(f'_create_llm_chain._cache: {self._cache}')
        if self._cache is None:
            return LLMChain(llm=self._llm, prompt=prompt)

        return LLMChainWithCache(llm=self._llm, prompt=prompt, cache=self._cache)

    @staticmethod
    def _extract_view_name(query: str) -> str:
        """
        Extract the view name from the provided SQL query.

        :param query: SQL query as a string
        :return: view name as a string
        """
        pattern = r"^CREATE(?: OR REPLACE)? TEMP VIEW (\S+)"
        match = re.search(pattern, query, re.IGNORECASE)
        if not match:
            raise ValueError(
                f"The provided query: '{query}' is not valid for creating a temporary view. "
                "Expected pattern: 'CREATE TEMP VIEW [VIEW_NAME] ...'"
            )
        return match.group(1)

    @staticmethod
    def _generate_search_prompt(columns: Optional[List[str]]) -> str:
        return (
            f"The best search results should contain as many as possible of these info: {','.join(columns)}"
            if columns is not None and len(columns) > 0
            else ""
        )

    @staticmethod
    def _generate_sql_prompt(columns: Optional[List[str]]) -> str:
        return (
            f"The result view MUST contain following columns: {columns}"
            if columns is not None and len(columns) > 0
            else ""
        )

    @staticmethod
    def _default_web_search_tool(desc: str) -> str:
        search_wrapper = GoogleSearchAPIWrapper()
        return str(search_wrapper.results(query=desc, num_results=10))

    @staticmethod
    def _is_http_or_https_url(s: str):
        result = urlparse(s)  # Parse the URL
        # Check if the scheme is 'http' or 'https'
        return result.scheme in ["http", "https"]

    @staticmethod
    def _extract_code_blocks(text) -> List[str]:
        code_block_pattern = re.compile(r"```(.*?)```", re.DOTALL)
        code_blocks = re.findall(code_block_pattern, text)
        if code_blocks:
            # If there are code blocks, strip them and remove language
            # specifiers.
            extracted_blocks = []
            for block in code_blocks:
                block = block.strip()
                if block.startswith("python"):
                    block = block.replace("python\n", "", 1)
                elif block.startswith("sql"):
                    block = block.replace("sql\n", "", 1)
                extracted_blocks.append(block)
            return extracted_blocks
        else:
            # If there are no code blocks, treat the whole text as a single
            # block of code.
            return [text]

    def log(self, message: str) -> None:
        if self._verbose:
            self._logger.log(message)

    def _trim_text_from_end(self, text: str, max_tokens: int) -> str:
        """
        Trim text from the end based on the maximum number of tokens allowed.

        :param text: text to trim
        :param max_tokens: maximum tokens allowed
        :return: trimmed text
        """
        tokens = list(self._encoding.encode(text))
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        return self._encoding.decode(tokens)

    def _get_url_from_search_tool(
            self, desc: str, columns: Optional[List[str]], cache: bool
    ) -> str:
        search_result = self._web_search_tool(desc)
        search_columns_hint = self._generate_search_prompt(columns)
        # Run the LLM chain to pick the best search result
        tags = self._get_tags(cache)
        return self._search_llm_chain.run(
            tags=tags,
            query=desc,
            search_results=search_result,
            columns={search_columns_hint},
        )

    def _create_dataframe_with_llm(
            self, text: str, desc: str, columns: Optional[List[str]], cache: bool
    ) -> DataFrame:
        clean_text = " ".join(text.split())
        web_content = self._trim_text_from_end(
            clean_text, self._max_tokens_of_web_content
        )

        sql_columns_hint = self._generate_sql_prompt(columns)

        # Run the LLM chain to get an ingestion SQL query
        tags = self._get_tags(cache)
        temp_view_name = random_view_name()
        llm_result = self._sql_llm_chain.run(
            tags=tags,
            query=desc,
            web_content=web_content,
            view_name=temp_view_name,
            columns=sql_columns_hint,
        )
        sql_query = self._extract_code_blocks(llm_result)[0]
        # The actual view name used in the SQL query may be different from the
        # temp view name because of caching.
        view_name = self._extract_view_name(sql_query)
        formatted_sql_query = CodeLogger.colorize_code(sql_query, "sql")
        self.log(f"SQL query for the ingestion:\n{formatted_sql_query}")
        self.log(f"Storing data into temp view: {view_name}\n")
        self._spark.sql(sql_query)
        return self._spark.table(view_name)

    @staticmethod
    def _get_df_schema(df: DataFrame) -> str:
        return "\n".join([f"{name}: {dtype}" for name, dtype in df.dtypes])

    @staticmethod
    def _trim_hash_id(analyzed_plan):
        # Pattern to find strings like #59 or #2021
        pattern = r"#\d+"

        # Remove matching patterns
        trimmed_plan = re.sub(pattern, "", analyzed_plan)

        return trimmed_plan

    @staticmethod
    def _parse_explain_string(df: DataFrame) -> str:
        """
        Helper function to parse the content of the extended explain
        string to extract the analyzed logical plan. As Spark does not provide
        access to the logical plane without accessing the query execution object
        directly, the value is extracted from the explain text representation.

        :param df: The dataframe to extract the logical plan from.
        :return: The analyzed logical plan.
        """
        with contextlib.redirect_stdout(io.StringIO()) as f:
            df.explain(extended=True)
        explain = f.getvalue()
        splits = explain.split("\n")
        # The two index operations will fail if Spark changes the textual
        # plan representation.
        begin = splits.index("== Analyzed Logical Plan ==")
        end = splits.index("== Optimized Logical Plan ==")
        # The analyzed logical plan starts two lines after the section marker.
        # The first line is the output schema.
        return "\n".join(splits[begin + 2: end])

    def _get_df_explain(self, df: DataFrame, cache: bool) -> str:
        raw_analyzed_str = self._parse_explain_string(df)
        tags = self._get_tags(cache)
        return self._explain_chain.run(
            tags=tags, input=self._trim_hash_id(raw_analyzed_str)
        )

    def _get_tags(self, cache: bool) -> Optional[List[str]]:
        if self._enable_cache and not cache:
            return SKIP_CACHE_TAGS
        return None

    @staticmethod
    def _get_col_info_json(col_name_type_list) -> str:
        field_info_json = ''
        for col_name_type in col_name_type_list:
            name_field = '"name":"{}",'
            type_field = '"type":"{}"'
            col_name = name_field.format(col_name_type[0])
            col_type = type_field.format(col_name_type[1])
            field_info_json = f'{field_info_json}{{{col_name}{col_type}}},'
        return f'[{field_info_json[:-1]}]'

    @staticmethod
    def _get_sample_data_json(sample_data_list) -> str:
        samples = ''
        for idx in range(len(sample_data_list)):
            samples = f'{samples}{json.dumps(sample_data_list[0].asDict())},'
        return f'[{samples[:-1]}]'

    def create_df(
            self, desc: str, columns: Optional[List[str]] = None, cache: bool = True
    ) -> DataFrame | None:
        """
        Create a Spark DataFrame by querying an LLM from web search result.

        :param desc: the description of the result DataFrame, which will be used for
                     web searching
        :param columns: the expected column names in the result DataFrame
        :param cache: If `True`, fetches cached data, if available. If `False`, retrieves fresh data and updates cache.

        :return: a Spark DataFrame
        """
        url = desc.strip()  # Remove leading and trailing whitespace
        is_url = self._is_http_or_https_url(url)
        # If the input is not a valid URL, use search tool to get the dataset.
        if not is_url:
            url = self._get_url_from_search_tool(desc, columns, cache)

        self.log(f"Parsing URL: {url}\n")
        try:
            response = requests.get(url, headers=self._HTTP_HEADER)
            response.raise_for_status()
        except requests.exceptions.HTTPError as http_err:
            self.log(f"HTTP error occurred: {http_err}")
            return None
        except Exception as err:
            self.log(f"Other error occurred: {err}")
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        # add url and page content to cache
        if cache:
            if self._cache.lookup(key=url):
                page_content = self._cache.lookup(key=url)
            else:
                page_content = soup.get_text()
                self._cache.update(key=url, val=page_content)
        else:
            page_content = soup.get_text()

        # If the input is a URL link, use the title of web page as the
        # dataset's description.
        if is_url:
            desc = soup.title.string
        return self._create_dataframe_with_llm(page_content, desc, columns, cache)

    def select_tool(self, desc: str, cache: bool = True) -> str | None:
        """
        Select a tool from a list of candidates.

        :param cache:
        :param desc: desc for a candidate tool.

        """
        instruction = f"The purpose of the request: {desc}" if desc is not None else ""
        tags = self._get_tags(cache)
        tool_list = ["transform_df", "create_square_bins", "plot_df_geo_bins", "analyze_s3_dataset"]

        response = None
        loop_count = 0
        print(f'select_tool() message: {instruction}')

        while response is None and loop_count < 3:
            response = self._tool_selection_llm_chain.run(
                tags=tags,
                instruction=instruction,
            )
            loop_count += 1

            if response is not None:
                print(f'select_tool-> response: {response}')
                for tool in tool_list:
                    if response.find(tool) != -1:
                        return tool
                response = None  # try again

        if response is None:
            raise Exception("Could not find a tool fitting the description", "")

    def _get_tools_from_response(self, response: str) -> dict | None:
        # look for json within the response
        pattern = r'\{.*?\}'
        json_pattern = re.compile(pattern, re.DOTALL)
        json_matches = re.findall(json_pattern, response)
        print(f'{len(json_matches)} => {json_matches}')
        tools = None
        if len(json_matches) > 0:
            candidate_tools = json_matches[0]
            tools = self.extract_tools_from_candidates(candidate_tools)
            if tools is None:
                if 'tools' in candidate_tools:
                    tools_index = candidate_tools.index('tools')
                    tools_array_start = candidate_tools.index('[', tools_index)
                    tools_array_end = candidate_tools.index(']', tools_index)
                    if tools_index < tools_array_start < tools_array_end:
                        tools_json_str = \
                            f'{{"tools":[{candidate_tools[tools_array_start + 1:tools_array_end]}]}}'
                        print(f'new tools-json-string: {tools_json_str}')
                        tools = self.extract_tools_from_candidates(tools_json_str)
        return tools

    @staticmethod
    def extract_tools_from_candidates(tools_json_str: str) -> dict | None:
        try:
            match_tools = json.loads(tools_json_str)
            print(f'select_tools-> response: {match_tools}')
            if match_tools is not None and len(match_tools['tools']) > 0:
                return match_tools
            else:
                return None
        except JSONDecodeError as e:
            return None

    def select_tool2(self, desc: str, cache: bool = True) -> str | None:
        """
        Select one or more tools from a list of candidates.

        :param cache:
        :param desc: desc for one or more candidate tools.

        """
        instruction = f"{desc}" if desc is not None else ""
        tags = self._get_tags(cache)
        print(f'select_tools() message: {instruction}')

        # testing single tool call
        # print("<-----------------")
        # single_response = self._tool_selection_llm_chain(desc)
        # print(single_response)
        # print("------------------>")

        response = None
        loop_count = 0
        while response is None and loop_count < 3:
            response = self._tools_selection_llm_chain.run(
                tags=tags,
                instruction=instruction,
            )
            loop_count += 1
            if response is not None:
                tools = self._get_tools_from_response(response)
                if tools is None:
                    response = None
                else:
                    response = json.dumps(tools)

        if response is None:
            self.log("Could not find tools fitting the description")

        return response

    def analyze_sample_data(self, data_samples: str, cache: bool = True) -> str | None:
        """
        Analyze the data items based on the given sample dataset.

        :param data_samples:  a few data samples.
        :param cache:

        """
        instruction = f'Please analyze the data based on these samples {data_samples}' \
            if data_samples is not None else ""
        tags = self._get_tags(cache)
        print(f'analyze_sample_data() message: {instruction}')

        response = None
        loop_count = 0

        while response is None and loop_count < 3:
            response = self._data_analysis_llm_chain.run(
                tags=tags,
                instruction=instruction,
            )
            loop_count += 1

            if response is not None:
                cleaned_response = (os.linesep.join([s for s in response.splitlines() if s])
                                    .replace("\n", "")).replace("\t", "")
                print(f'analyze_sample_data, cleaned_response => {cleaned_response}')
                try:
                    json.loads(cleaned_response)
                    return cleaned_response
                except JSONDecodeError as e:
                    # invalid JSON return, re-try
                    response = None

        return None

    def create_s3_df(self, s3_url, header: bool = True,
                     infer_schema: bool = True, cache: bool = True) -> DataFrame | None:
        if s3_url is None or not s3_url.startswith("s3a://"):
            return None

        try:
            df = self._spark \
                .read.format("csv") \
                .option("header", header) \
                .option("inferSchema", infer_schema) \
                .load(s3_url)
            # replace spaces in the column name with  '_'
            return df.select([col(c).alias(c.replace(' ', '_')) for c in df.columns])
        except Exception as e:
            return None

    def analyze_s3_dataset(self, df: DataFrame, number_samples: int = 3, cache: bool = True) -> str:
        col_info_json = self._get_col_info_json(df.dtypes)
        samples_json = self._get_sample_data_json(df.take(number_samples))

        response = self.analyze_sample_data(samples_json)
        success = True
        error_msg = ''
        if response is None:
            success = False
            response = '{}'
            error_msg = "Failed to get preliminary analysis results from LLM after 3 tries."

        response = response.replace("\\n", "").replace('\\', '')
        print(response)
        print("================ analyze_s3_dataset above === ")
        return (f'{{"success":"{success}","error_msg":"{error_msg}","col_data_types":{col_info_json},'
                f'"samples":{samples_json},"col_descriptions":{response}}}')

    def get_sql_code(self, df: DataFrame, desc: str, cache: bool = True) -> str:
        schema_str = self._get_df_schema(df)
        tags = self._get_tags(cache)
        llm_result = self._transform_chain.run(
            tags=tags, view_name='', columns=schema_str, desc=desc
        )
        return self._extract_code_blocks(llm_result)[0]

    def transform_df(self, df: DataFrame, desc: str, cache: bool = True) -> DataFrame:
        """
        This method applies a transformation to a provided Spark DataFrame,
        the specifics of which are determined by the 'desc' parameter.

        :param df: The Spark DataFrame that is to be transformed.
        :param desc: A natural language string that outlines the specific transformation to be applied on the DataFrame.
        :param cache: If `True`, fetches cached data, if available. If `False`, retrieves fresh data and updates cache.

        :return: Returns a new Spark DataFrame that is the result of applying the specified transformation
                 on the input DataFrame.
        """
        # get first record as an example and append to the 'desc'
        desc = f'{desc}. Sample: {df.take(1)[0]}.'

        temp_view_name = random_view_name()
        create_temp_view_code = CodeLogger.colorize_code(
            f'df.createOrReplaceTempView("{temp_view_name}")', "python"
        )
        self.log(f"Creating temp view for the transform:\n{create_temp_view_code}")
        df.createOrReplaceTempView(temp_view_name)
        schema_str = self._get_df_schema(df)

        print(f'transform_df() columns: {schema_str}\n desc: {desc}')

        tags = self._get_tags(cache)
        llm_result = self._transform_chain.run(
            tags=tags, view_name=temp_view_name, columns=schema_str, desc=desc
        )
        sql_query_from_response = self._extract_code_blocks(llm_result)[0]
        # Replace the temp view name in case the view name is from the cache.
        sql_query = replace_view_name(sql_query_from_response, temp_view_name)
        formatted_sql_query = CodeLogger.colorize_code(sql_query, "sql")
        self.log(f"SQL query for the transform:\n{formatted_sql_query}")
        return self._spark.sql(sql_query)

    def explain_df(self, df: DataFrame, cache: bool = True) -> str:
        """
        This method generates a natural language explanation of the SQL plan of the input Spark DataFrame.

        :param df: The Spark DataFrame to be explained.
        :param cache: If `True`, fetches cached data, if available. If `False`, retrieves fresh data and updates cache.

        :return: A string explanation of the DataFrame's SQL plan, detailing what the DataFrame is intended to retrieve.
        """
        explain_result = self._get_df_explain(df, cache)
        # If there is code block in the explain result, ignore it.
        if "```" in explain_result:
            summary = explain_result.split("```")[-1]
            return summary.strip()
        else:
            return explain_result

    def plot_df(
            self, df: DataFrame, desc: Optional[str] = None, cache: bool = True
    ) -> None:
        instruction = f"The purpose of the plot: {desc}" if desc is not None else ""
        tags = self._get_tags(cache)
        response = self._plot_chain.run(
            tags=tags,
            columns=self._get_df_schema(df),
            explain=self._get_df_explain(df, cache),
            instruction=instruction,
        )
        self.log(response)
        codeblocks = self._extract_code_blocks(response)
        code = "\n".join(codeblocks)
        try:
            exec(compile(code, "plot_df-CodeGen", "exec"))
        except Exception as e:
            raise Exception("Could not evaluate Python code", e)

    def plot_df_geo_bins(
            self, df: DataFrame, desc: Optional[str] = None, cache: bool = True
    ) -> None:
        instruction = f"The purpose of the plot: {desc}" if desc is not None else ""
        tags = self._get_tags(cache)

        response = None
        loop_count = 0

        while response is None and loop_count < 3:
            loop_count += 1
            response = self._geo_bins_plot_chain.run(
                tags=tags,
                columns=self._get_df_schema(df),
                explain=self._get_df_explain(df, cache),
                instruction=instruction,
            )
            self.log(response)
            codeblocks = self._extract_code_blocks(response)
            code = "\n".join(codeblocks)
            try:
                exec(compile(code, "plot_df_geo_bins-CodeGen", "exec"))
            except Exception as e:
                if loop_count < 3:
                    response = None
                else:
                    raise Exception("Could not evaluate Python code", e)

    def create_square_bins(
            self, df: DataFrame, output_file: str, desc: Optional[str] = None, cache: bool = True
    ) -> None:
        # assert DataFrame is not None
        if df is None:
            raise Exception("Could not perform geo-binning!", "The required PySpark DataFrame is not defined!")

        instruction = f"The purpose of the action: {desc}" if desc is not None else ""
        tags = self._get_tags(cache)

        response = None
        loop_count = 0

        while response is None and loop_count < 3:
            loop_count += 1
            response = self._create_geo_bins_chain.run(
                tags=tags,
                columns=self._get_df_schema(df),
                explain=self._get_df_explain(df, cache),
                instruction=instruction,
            )
            self.log(response)
            print(f'create_square_bins: {response}')
            pattern = r'\{.*?\}'
            json_pattern = re.compile(pattern, re.DOTALL)
            response_json_array = re.findall(json_pattern, response)
            print(f'create_square_bins: response_json_array => {response_json_array}')
            if len(response_json_array) > 0:
                # codeblocks = self._extract_code_blocks(response)
                # code = "\n".join(codeblocks)
                # print(code)
                # check lon_column and lat_column names against Dataframe
                parameters_json = _check_lat_lon_cols(df, response_json_array[0])
                print(f'parameter json => {parameters_json}')
                code_json = json.loads(parameters_json)
                python_code = ''
                for item in code_json:
                    if type(code_json[item]) is str:
                        python_code = f'{python_code}\n{item} = "{code_json[item]}"'
                    else:
                        python_code = f'{python_code}\n{item} = {code_json[item]}'

                python_code = (f'{python_code}\noutput_file = "{output_file}"\n'
                               f'{geo_function_templates["create_square_bins"]}')
                print(python_code)
                try:
                    exec(compile(python_code, "create_square_bins-CodeGen", "exec"))
                except Exception as e:
                    self.log(python_code)
                    if loop_count < 3:
                        response = None
                    else:
                        raise Exception("Could not evaluate Python code in create_square_bins after 3 tries!", e)
            else:
                response = None

    def extract_info(self, desc: str) -> str:
        return self._info_extraction_chain.run(
            instruction=desc
        )

    def verify_df(self, df: DataFrame, desc: str, cache: bool = True) -> None:
        """
        This method creates and runs test cases for the provided PySpark dataframe transformation function.

        :param df: The Spark DataFrame to be verified
        :param desc: A description of the expectation to be verified
        :param cache: If `True`, fetches cached data, if available. If `False`, retrieves fresh data and updates cache.
        """
        tags = self._get_tags(cache)
        llm_output = self._verify_chain.run(tags=tags, df=df, desc=desc)

        codeblocks = self._extract_code_blocks(llm_output)
        llm_output = "\n".join(codeblocks)

        self.log(f"LLM Output:\n{llm_output}")

        formatted_code = CodeLogger.colorize_code(llm_output, "python")
        self.log(f"Generated code:\n{formatted_code}")

        locals_ = {}
        try:
            exec(compile(llm_output, "verify_df-CodeGen", "exec"), {"df": df}, locals_)
        except Exception as e:
            raise Exception("Could not evaluate Python code", e)
        self.log(f"\nResult: {locals_['result']}")

    def udf(self, func: Callable) -> Callable:
        from inspect import signature

        desc = func.__doc__
        func_signature = str(signature(func))
        input_args_types = func_signature.split("->")[0].strip()
        return_type = func_signature.split("->")[1].strip()
        udf_name = func.__name__

        code = self._udf_chain.run(
            input_args_types=input_args_types,
            desc=desc,
            return_type=return_type,
            udf_name=udf_name,
        )

        formatted_code = CodeLogger.colorize_code(code, "python")
        self.log(f"Creating following Python UDF:\n{formatted_code}")

        locals_ = {}
        try:
            exec(compile(code, "udf-CodeGen", "exec"), globals(), locals_)
        except Exception as e:
            raise Exception("Could not evaluate Python code", e)
        return locals_[udf_name]

    def activate(self):
        """
        Activates AI utility functions for Spark DataFrame.
        """
        DataFrame.ai = AIUtils(self)
        # Patch the Spark Connect DataFrame as well.
        try:
            from pyspark.sql.connect.dataframe import DataFrame as CDataFrame

            CDataFrame.ai = AIUtils(self)
        except ImportError:
            self.log(
                "The pyspark.sql.connect.dataframe module could not be imported. "
                "This might be due to your PySpark version being below 3.4."
            )

    def commit(self):
        """
        Commit the staging in-memory cache into persistent cache, if cache is enabled.
        """
        if self._cache is not None:
            self._cache.commit()
