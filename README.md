# Environment Setup

## Python

Install [`python3.11`](https://www.python.org/downloads/release/python-3117/) _if you do not have it installed_.

## Dependency Management

### Option 1: `Virtualenv`

Virtualenv is pre-installed with Python 3.3 and later.

### Option 2: `Conda`

https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html

## Environment Setup

### Option 1: `Virtualenv`

#### 1. Create Virtual Environment

```shell
python3.11 -m venv venv
```

#### 2. Activate Virtual Environment

##### On macOS and Linux:

```shell
source venv/bin/activate
```

##### On Windows:

```shell
.\venv\Scripts\activate
```

#### 3. Install Dependencies

```shell
pip install -r requirements.txt
```

#### 4. Install PyTorch

https://pytorch.org/get-started/locally/

### Option 2: `Conda`

#### 1. Create Conda Environment

```shell
conda env create -f environment.yml
```

#### 2. Activate Conda Environment

```shell
conda activate dream-heist
```

#### 3. Install PyTorch

https://pytorch.org/get-started/locally/

[//]: # (# If you do not have conda installed, please install it from the following link:)

[//]: # (   https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

[//]: # ()

[//]: # (# How to install the env?)

[//]: # (    Use this code in command line: conda env create -f environment.yml)

[//]: # ()

[//]: # (# Install pytorch manually)

[//]: # (    Go to pytorch website and select the right version for your system and run the command in the command line.)

[//]: # ()

[//]: # (# How to activate the env?)

[//]: # (    Use this code in command line: conda activate dream-heist)

[//]: # ()

[//]: # (# How to update the environment.yml?)

[//]: # (    Use this code in command line: conda env export | grep -v "^prefix: " > environment.yml)

[//]: # ()

[//]: # (# Note requirements.txt has been removed and replaced with environment.yml)
