# Register kernel scpec

When jupyter notebook or jupyter-lab does not detect your virtual environment, you must register it.

- Create your virtual python environment: `python3 -m venv .venv`
- Activate your environment on Linux: `source .venv/bin/activate`
- Install `ipykernel`: `pip install ipykernel`
- Register your environment as a kernel spec:

```bash
python -m ipykernel install --user --name=myenv --display-name="Python (myenv)"
```