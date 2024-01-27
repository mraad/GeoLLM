# GeoLLM

Demo of [ReAct](https://python.langchain.com/docs/modules/agents/agent_types/react) pseudo geospatial logic with [Ollama](https://ollama.ai/).

## Create Conda Env

```shell
conda deactivate
conda remove --yes --all --name geollm
git clone https://github.com/mraad/GeoLLM.git
cd GeoLLM
conda env create
conda activate geollm
```

### Ollama

[Download](https://ollama.ai/download) and start ollama:
```shell
ollama server
```

Pull [mistral](https://ollama.ai/library/mistral) model:
```shell
ollama pull mistral
```

Start jupyter lab:
```shell
jupyter lab
```

### References:

- https://ollama.ai/
- https://python.langchain.com/docs/modules/agents/agent_types/react
- https://learnlangchain.streamlit.app/
- https://docs.featureform.com/llms-embeddings-and-vector-databases/building-a-chatbot-with-openai-and-a-vector-database
- https://github.com/RoboCoachTechnologies/GPT-Synthesizer
- https://github.com/facebookresearch/codellama
- https://levelup.gitconnected.com/you-can-now-build-a-chatbot-to-talk-to-your-internal-knowledge-base-b6066cacf2d5
