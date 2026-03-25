FROM python:3.12-slim
RUN pip install --no-cache-dir \
    numpy scipy pandas scikit-learn matplotlib \
    statsmodels seaborn networkx sympy
WORKDIR /work
# NOTE: requests is intentionally excluded — sandbox has --network none
