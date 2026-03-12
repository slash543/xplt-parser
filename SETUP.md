# Setup

## 1. Create a virtual environment

```bash
python3 -m venv .venv
```

## 2. Install dependencies

```bash
.venv/bin/pip install -r requirements.txt
```

On Windows:
```bash
.venv\Scripts\pip install -r requirements.txt
```

## 3. Register the kernel with VS Code

```bash
.venv/bin/python -m ipykernel install --user --name xplt-parser --display-name "xplt-parser"
```

On Windows:
```bash
.venv\Scripts\python -m ipykernel install --user --name xplt-parser --display-name "xplt-parser"
```

## 4. Open the notebook in VS Code

1. Open `xplt_explorer.ipynb`
2. Click **Select Kernel** (top-right corner)
3. Choose **xplt-parser** from the list

> Place `sample.feb` and `sample.xplt` in the same folder as the notebook before running.
