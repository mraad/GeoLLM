### GeoLLM 

This is a demo app for showing customers on using OpenAI GPT APIs to analyze 
user's input message and then perform a spatial analysis using some pre-defined 
datasets such as Chicago Crime dataset. 

To start the app server, you just simply run the following command from a terminal 
window: 

`uvicorn app:app`

though you need to define the following environment variables first: 

#### AWS S3 bucket access parameters
```
export AWS_ACCESS_KEY=xxxxxx
export AWS_SECRET_KEY=xxxxxxxxx
```

#### AzureChatOpenAI and OpenAI required parameters
```
export openai_api_base=https://xxxx.openai.azure.com/
export openai_api_version=2023-07-01-preview
export deployment_name=esri-gpt4-dev
export openai_api_key=xxxxx
export openai_api_type=azure
export llm_temperature=0
```

#### Use OpenAI function call or not
`export USE_OPENAI_FUNCTION_CALL=true`
