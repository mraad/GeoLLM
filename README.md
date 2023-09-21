# GeoLLM

## Create Conda Env using [mamba](https://github.com/mamba-org/mamba).

```shell
mamba update -n base -c conda-forge conda mamba

mamba deactivate
mamba remove --yes --all --name geollm
git clone https://github.com/mraad/GeoLLM.git
cd GeoLLM
mamba env create
mamba activate geollm
pip install lib/geofunctions-0.5-py3-none-any.whl
```

### Add 2 additional jar files to the geollm environment 

To run the demo, we need to access CSV files from AWS S3 buckets. Download these two jar files, hadoop-aws-3.3.4.jar 
and aws-java-sdk-bundle-1.12.262.jar and put them into the following folder (assume you installed mamba in your home folder):
```
~/mambaforge/envs/geollm/lib/python3.10/site-packages/pyspark/jars
```

### References:

- https://learnlangchain.streamlit.app/
