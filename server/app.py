##
# pip install python-multipart
#
from pydantic import BaseModel
from fastapi import HTTPException, FastAPI, Response, Depends, Form
from uuid import UUID, uuid4

from fastapi_sessions.backends.implementations import InMemoryBackend
from fastapi_sessions.session_verifier import SessionVerifier
from fastapi_sessions.frontends.implementations import SessionCookie, CookieParameters

from geollm.geo_agent import GeoAgent
from geollm.geo_llm import GeoLLM

import glob
import os
from pyspark.sql import SparkSession
from langchain.chat_models import AzureChatOpenAI
import time
from typing import Annotated


class SessionData(BaseModel):
    username: str


cookie_params = CookieParameters()

# Uses UUID
cookie = SessionCookie(
    cookie_name="cookie",
    identifier="general_verifier",
    auto_error=True,
    secret_key="DONOTUSE",
    cookie_params=cookie_params,
)
backend = InMemoryBackend[UUID, SessionData]()


class BasicVerifier(SessionVerifier[UUID, SessionData]):
    def __init__(
            self,
            *,
            identifier: str,
            auto_error: bool,
            backend: InMemoryBackend[UUID, SessionData],
            auth_http_exception: HTTPException,
    ):
        self._identifier = identifier
        self._auto_error = auto_error
        self._backend = backend
        self._auth_http_exception = auth_http_exception

    @property
    def identifier(self):
        return self._identifier

    @property
    def backend(self):
        return self._backend

    @property
    def auto_error(self):
        return self._auto_error

    @property
    def auth_http_exception(self):
        return self._auth_http_exception

    def verify_session(self, model: SessionData) -> bool:
        """If the session exists, it is valid"""
        return True


verifier = BasicVerifier(
    identifier="general_verifier",
    auto_error=True,
    backend=backend,
    auth_http_exception=HTTPException(status_code=403, detail="invalid session"),
)


def create_geo_llm() -> GeoLLM:
    spark_jars = glob.glob(os.path.join(".", "lib", "*.jar"))[0]

    aws_access_key = os.environ["AWS_ACCESS_KEY"]
    aws_secret_key = os.environ["AWS_SECRET_KEY"]

    spark = (
        SparkSession.builder.appName("SpatialBin")
        .config("spark.driver.memory", "64G")
        .config("spark.executor.memory", "64G")
        .config("spark.memory.offHeap.enabled", True)
        .config("spark.memory.offHeap.size", "64G")
        .config("spark.ui.enabled", False)
        .config("spark.hadoop.fs.s3a.access.key", aws_access_key)
        .config("spark.hadoop.fs.s3a.secret.key", aws_secret_key)
        .config('spark.hadoop.fs.s3a.aws.credentials.provider', 'org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider')
        .config("spark.jars", spark_jars)
        .getOrCreate()
    )

    llm = AzureChatOpenAI(
        openai_api_base="https://esri-openai-dev.openai.azure.com",
        openai_api_version="2023-05-15",
        deployment_name="esri-chatgpt-dev",
        openai_api_key="974995b34b294be196324349ac32e5b2",
        openai_api_type="azure",
    )

    geo_llm = GeoLLM(llm, spark_session=spark.getActiveSession(), enable_cache=False, verbose=True)
    geo_llm.activate()
    return geo_llm


agent_cache = {}
# create a singleton for GeoLLM object, which includes a SparkSession that
# will be shared among all GeoAgents!
geoLLM = create_geo_llm()

app = FastAPI()


@app.post("/chat/", dependencies=[Depends(cookie)])
async def chat(message: Annotated[str, Form()], response: Response,
               session_id: UUID = Depends(cookie)
               ) -> str:
    agent = await get_geo_agent(session_id, response)
    if agent:
        print(f'message from user: {message}')
        print(f'selected agent: {agent}, # of agents: {len(agent_cache)}')
        return agent.chat(message)
    else:
        return f'{{"success":"False", "error_msg":"No valid GeoAgent is available!"}}'


async def get_geo_agent(session_id: UUID, response: Response) -> GeoAgent:
    if session_id is None:
        session_id = uuid4()
        return await create_agent_and_session(session_id, response)
    else:
        try:
            print(f'session id: {session_id}')
            agent = agent_cache[session_id]
            return agent
        except KeyError as err:
            # maybe session expired, re-create session
            # delete old session
            try:
                await backend.delete(session_id)
                return await create_agent_and_session(session_id, response)
            except KeyError as e:
                session_id = uuid4()
                return await create_agent_and_session(session_id, response)


async def create_agent_and_session(session_id: UUID, response: Response) -> GeoAgent:
    agent = GeoAgent(geoLLM)
    agent_cache[session_id] = agent
    name = f'geo_agent_{time.time()}'
    data = SessionData(username=name)
    await backend.create(session_id, data)
    cookie.attach_to_response(response, session_id)
    print(f'data.username: {data.username}')
    return agent


@app.post("/create_session/{name}")
async def create_session(name: str, response: Response):
    session_id = uuid4()
    await create_agent_and_session(session_id, response)
    return f"created session for {name}"

#
# @app.get("/whoami", dependencies=[Depends(cookie)])
# async def whoami(session_data: SessionData = Depends(verifier)):
#     return session_data


@app.get("/check_session", dependencies=[Depends(cookie)])
async def check_session(session_data: SessionData = Depends(verifier), session_id: UUID = Depends(cookie)):
    print(session_id)
    session_info = await backend.read(session_id)
    print(f'session type: {type(session_info)}')
    print(f'session name: {session_info.username}')
    agent = agent_cache[session_id]
    print(f'geo agent: {agent}')

    return session_data


@app.post("/start_new_session")
async def del_session(response: Response, session_id: UUID = Depends(cookie)):
    try:
        del agent_cache[session_id]
        await backend.delete(session_id)
    except KeyError as err:
        # do nothing
        print(err)

    cookie.delete_from_response(response)
    return "now you are ready to start a new session!"

