from typing import Dict, Callable
from typing import Any, List, Optional

from langchain import LLMChain
from langchain.callbacks.manager import Callbacks

from .file_cache import JsonCache, FileCache, SQLiteCacheWrapper
from .temp_view_utils import canonize_string


# JsonCache, SQLiteCacheWrapper, FileCache


class Cache:
    """
    This class provides an interface for a simple in-memory and persistent cache system.
    It keeps an in-memory staging cache, which gets updated through the `update` method and
    can be persisted through the `commit` method. Cache lookup is first performed on the
    in-memory staging cache, and if not found, it is performed on the persistent cache.

    Attributes:
        _staging_updates: A dictionary to keep track of the in-memory staging updates.
        _file_cache: An instance of either JsonCache or SQLiteCacheWrapper that acts as
        the persistent cache.
    """

    def __init__(
            self, cache_file_location: str = ".pyspark_ai.json", file_format: str = "json"
    ):
        """
        Initializes a new instance of the Cache class.

        Args:
            cache_file_location (str, optional): The path to the cache file for the JsonCache or
            SQLiteCacheWrapper. Defaults to ".pyspark_ai.json".
            file_format (str, optional): The format of the file to use for the cache.
            Defaults to "json".
        """
        self._staging_updates: Dict[str, str] = {}
        if file_format == "json":
            self._file_cache: FileCache = JsonCache(cache_file_location)
        else:
            self._file_cache = SQLiteCacheWrapper(cache_file_location)

    def lookup(self, key: str) -> Optional[str]:
        """
        Performs a lookup in the cache using the given key.

        Args:
            key (str): The key string for the lookup.

        Returns:
            Optional[str]: The cached text corresponding to the key, if available. Otherwise, None.
        """
        # First look in the staging cache
        staging_result = self._staging_updates.get(key)
        if staging_result is not None:
            return staging_result
        # If not found in staging cache, look in the persistent cache
        return self._file_cache.lookup(key)

    def update(self, key: str, val: str) -> None:
        """
        Updates the staging cache with the given key and value.

        Args:
            key (str): The key string.
            val (str): The value to be cached.
        """
        self._staging_updates[key] = val

    def clear(self) -> None:
        """
        Clears both the in-memory staging cache and the persistent cache.
        """
        self._file_cache.clear()
        self._staging_updates = {}

    def commit(self) -> None:
        """
        Commits all the staged updates to the persistent cache.
        """
        self._file_cache.commit_staging_cache(self._staging_updates)
        self._staging_updates = {}


# canonize_string
SKIP_CACHE_TAGS = ["SKIP_CACHE"]


class LLMChainWithCache(LLMChain):
    cache: Cache

    @staticmethod
    def _sort_and_stringify(*args: Any) -> str:
        # Convert all arguments to strings, then sort them
        sorted_args = sorted(str(arg) for arg in args)
        # Join all the sorted, stringified arguments with a space
        result = " ".join(sorted_args)
        return result

    def run(
            self,
            *args: Any,
            callbacks: Callbacks = None,
            tags: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> str:
        assert not args, "The chain expected no arguments"
        prompt_str = canonize_string(self.prompt.format_prompt(**kwargs).to_string())
        use_cache = tags != SKIP_CACHE_TAGS
        cached_result = self.cache.lookup(prompt_str) if use_cache else None
        if cached_result is not None:
            return cached_result
        result = super().run(*args, callbacks=callbacks, tags=tags, **kwargs)
        if use_cache:
            self.cache.update(prompt_str, result)
        return result


# Cache


class SearchToolWithCache:
    def __init__(self, web_search_tool: Callable[[str], str], cache: Cache):
        self.web_search_tool = web_search_tool
        self.cache = cache

    def search(self, query: str) -> str:
        # Try to get the result from the cache
        key = f"web_search:{query}"
        cached_result = self.cache.lookup(key)
        if cached_result is not None:
            return cached_result

        # If the result was not in the cache, use the web_search_tool
        result = self.web_search_tool(query)

        # Update the cache with the new result
        self.cache.update(key, result)

        return result
