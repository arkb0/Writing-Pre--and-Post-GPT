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

From the project root, type/paste in order:

```bash
chmod +x test.sh run.sh # Give yourself execute permissons
./test.sh   # runs demo mode checks
./run.sh    # runs full analysis
```

**Note for Windows Users**: Git Bash must be run as admin because Windows handles permissions to temp folders differently 💀. It's a feature, not a bug (promise).

#### For the Curious...

Both the `.sh` scripts create a venv and activate it.

`test.sh`: Runs checks (print architecture, print assignment)
`run.sh`: Runs all tests on the real data.

***

### Checkpointing

The scripts now persist intermediate (default: every 100) results to `(project root)/output_confidential/`. This effectively allows the scripts to 'resume' after an interrupt, timeout, or (unlikely) crash.

The `output_confidential` must **not** be deleted between runs. It should not need to be returned with the actual outputs because it may leak PII.

***

### Manual Runs

#### Suggested Running Sequence

From the project root, type/paste in order:

```bash
python3 -m venv venv      # Create a virtual environment (venv)
source venv/bin/activate  # Activate the venv

python 1/Pre_and_Post_GPT_Writing_Styles_01_Expert.py --use-fake-data --print-architecture --print-assignments  # Demo run + print architecture and assignment text (to separate files) using the fake data (can do the same for 2/Pre_and_Post_GPT_Writing_Styles_02_ML.py but they use the same directory)

python 1/Pre_and_Post_GPT_Writing_Styles_01_Expert.py  # Real-data run of the first script
python 2/Pre_and_Post_GPT_Writing_Styles_02_ML.py      # Real-data run of the second script
```

***

#### Reference: The Full Version

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

**4 - Real data, debug only, no analysis**
```bash
python 1/Pre_and_Post_GPT_Writing_Styles_01_Expert.py --print-architecture --print-assignments --skip-analysis
python 2/Pre_and_Post_GPT_Writing_Styles_02_ML.py     --print-architecture --print-assignments --skip-analysis
```
Expected: reads from `(project_root)/data/`, prints the directory tree and first assignment name and text, writes to their respective output dirs.

**5 - Real data, print architecture**
```bash
python 1/Pre_and_Post_GPT_Writing_Styles_01_Expert.py --print-architecture
python 2/Pre_and_Post_GPT_Writing_Styles_02_ML.py     --print-architecture
```
Expected: reads from `(project_root)/data/`, prints the directory tree, runs full analyses, writes to their respective output dirs.
