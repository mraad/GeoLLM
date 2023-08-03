# GeoLLM

## Create Conda Env using [mamba](https://github.com/mamba-org/mamba)

```shell
mamba update -n base -c conda-forge conda mamba

mamba deactivate
mamba remove --yes --all --name geollm
git clone https://github.com/mraad/GeoLLM.git
cd GeoLLM
mamba env create
mamba activate geollm
pip install lib/geofunctions-0.4-py3-none-any.whl
```
