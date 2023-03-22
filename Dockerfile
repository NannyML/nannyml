#  Author:  Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

## Thank you Michael Oliver
## https://github.com/michaeloliverx/python-poetry-docker-example/blob/master/docker/Dockerfile
## Thank you Baptiste Maingret
## https://github.com/bmaingret/coach-planner/blob/main/docker/Dockerfile

ARG APP_NAME=nannyml
ARG APP_PATH=/opt/$APP_NAME
ARG PYTHON_VERSION=3.10.0-slim-bullseye
ARG POETRY_VERSION=1.3.2

#
# Stage: staging
#
FROM python:$PYTHON_VERSION as staging
ARG APP_NAME
ARG APP_PATH
ARG POETRY_VERSION

ENV \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1
ENV \
    POETRY_VERSION=$POETRY_VERSION \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    curl \
    build-essential

# Install Poetry - respects $POETRY_VERSION & $POETRY_HOME
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python
ENV PATH="$POETRY_HOME/bin:$PATH"

# Import our project files
WORKDIR $APP_PATH
COPY ./poetry.lock ./pyproject.toml ./
COPY . .

#
# Stage: development
#
FROM staging as development
ARG APP_NAME
ARG APP_PATH

ENV POETRY_INSTALLER_MAX_WORKERS=10

# Install project in editable mode and with development dependencies
WORKDIR $APP_PATH
RUN poetry install

ENTRYPOINT ["poetry", "run"]
CMD ["nml"]

#
# Stage: build
#
FROM staging as build
ARG APP_PATH

WORKDIR $APP_PATH
RUN poetry build --format wheel
RUN poetry export --format requirements.txt --output constraints.txt --without-hashes

#
# Stage: production
#
FROM python:$PYTHON_VERSION as production
ARG APP_NAME
ARG APP_PATH

ENV \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1

ENV \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

# LightGBM dependency
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    libgomp1

# Get build artifact wheel and install it respecting dependency versions
WORKDIR $APP_PATH
COPY --from=build $APP_PATH/dist/*.whl ./
COPY --from=build $APP_PATH/constraints.txt ./
RUN pip install ./$APP_NAME*.whl --constraint constraints.txt

RUN useradd --create-home --shell /bin/bash $APP_NAME
WORKDIR /home/$APP_NAME
USER $APP_NAME

# export APP_NAME as environment variable for the CMD
ENV APP_NAME=$APP_NAME

CMD ["sh", "-c", "$APP_NAME"]
