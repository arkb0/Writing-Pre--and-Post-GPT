## Background

Scripts converted from the notebooks - with minor tweaks.

## Run instructions

### Expected Directory Structure

```
(project root)/
    :-- data/
    |   '-- (subtree by term, assignment, and student - omitted for brevity)
    :-- 1/
    |   '-- Pre_and_Post_GPT_Writing_Styles_01_Expert.py
    '-- 2/
        '-- Pre_and_Post_GPT_Writing_Styles_02_ML.py
```

### The Easy Way

From the project root:

```bash
chmod +x test.sh run.sh
./test.sh   # runs demo mode checks
./run.sh    # runs full analysis
```

#### For the Curious...

Both the `.sh` scripts create a venv and activate it.

`test.sh`: Runs checks (print architecture, print assignment)
`run.sh`: Runs all tests on the real data.

***

### Manual Runs

Assumption: The venv is created and activated. All commands below use relative paths runnable from the root.

**1 - Demo mode, print architecture**
```bash
python 1/Pre_and_Post_GPT_Writing_Styles_01_Expert.py --use-fake-data --print-architecture
python 2/Pre_and_Post_GPT_Writing_Styles_02_ML.py     --use-fake-data --print-architecture
```
Expected: synthetic folders are created, ASCII tree is printed, `[info] SP21 has no Assignment_4` (or similar) appears for whichever semesters lost the coin flip, analysis runs, results written to their respective output dirs.

**2 - Demo mode, print assignments**
```bash
python 1/Pre_and_Post_GPT_Writing_Styles_01_Expert.py --use-fake-data --print-assignments
python 2/Pre_and_Post_GPT_Writing_Styles_02_ML.py     --use-fake-data --print-assignments
```
Expected: for each of the 7 synthetic semesters, the semester code, first assignment name, and extracted PDF text are printed to the terminal, results written to their respective output dirs.

**3 - Demo mode, silent**
```bash
python 1/Pre_and_Post_GPT_Writing_Styles_01_Expert.py --use-fake-data
python 2/Pre_and_Post_GPT_Writing_Styles_02_ML.py     --use-fake-data
```
Expected: no diagnostic output beyond ingestion progress lines; all results written to their respective output dirs.

**4 - Real data, print architecture**
```bash
python 1/Pre_and_Post_GPT_Writing_Styles_01_Expert.py --print-architecture
python 2/Pre_and_Post_GPT_Writing_Styles_02_ML.py     --print-architecture
```
Expected: reads from `(project_root)/data/`, prints the directory tree, runs full analyses, writes to their respective output dirs.
