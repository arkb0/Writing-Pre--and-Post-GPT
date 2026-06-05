from __future__ import annotations

import argparse
import subprocess
import sys
import os
import re
import random
import string
import textwrap
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

# =============================================================================
# 1. Dependency installation
# =============================================================================

def _pip(*args):
    import threading
    import itertools

    done  = threading.Event()
    frames = itertools.cycle(['-', '\\', '|', '/'])

    def _spin():
        while not done.wait(0.1):
            print(f'\r  installing... {next(frames)}', end='', flush=True)

    t = threading.Thread(target=_spin, daemon=True)
    t.start()
    try:
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', '--quiet', *args],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    finally:
        done.set()
        t.join()
        print('\r  installing... done.   ')

_pip(
    'nltk>=3.8', 'spacy>=3.7', 'numpy>=1.26', 'pandas>=2.1',
    'matplotlib>=3.8', 'seaborn>=0.13', 'scikit-learn>=1.4', 'scipy>=1.12',
    'pdfplumber>=0.10', 'fpdf2>=2.7',
)

import nltk
for resource in ['punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'stopwords']:
    nltk.download(resource, quiet=True)

try:
    import en_core_web_sm
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'])

# =============================================================================
# 2. Remaining imports (after installation)
# =============================================================================

import matplotlib
matplotlib.use('Agg')  # non-interactive backend; no display required
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import pdfplumber
import seaborn as sns
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
from scipy.spatial.distance import cosine
from scipy.stats import mannwhitneyu
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =============================================================================
# 3. CLI arguments
# =============================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='AI-Impact Essay Analysis Toolkit — 01 Expert Metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''\
            Examples
            --------
            # 1. Demo mode, print architecture
            python Pre_and_Post_GPT_Writing_Styles_01_Expert.py --use-fake-data --print-architecture

            # 2. Demo mode, print assignments
            python Pre_and_Post_GPT_Writing_Styles_01_Expert.py --use-fake-data --print-assignments

            # 3. Demo mode, silent (just write output)
            python Pre_and_Post_GPT_Writing_Styles_01_Expert.py --use-fake-data

            # 4. Real data, print architecture
            python Pre_and_Post_GPT_Writing_Styles_01_Expert.py --print-architecture
        '''),
    )
    parser.add_argument(
        '--use-fake-data', '-f',
        action='store_true',
        help='Generate and use synthetic data instead of reading from data/.',
    )
    parser.add_argument(
        '--print-architecture', '-a',
        action='store_true',
        help='Print the directory tree of the data root after ingestion.',
    )
    parser.add_argument(
        '--print-assignments', '-p',
        action='store_true',
        help='Print the first assignment (first student, last submission for the student if they have multiple pdfs) from each semester.',
    )
    return parser.parse_args()

ARGS = _parse_args()

# =============================================================================
# 4. Output directory
# =============================================================================

OUTPUT_DIR = Path('output_1_em')
OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# 5. Configuration
# =============================================================================

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

COURSES: list[tuple[str, str]] = [
    ('CS6460', 'Educational Technology'),
]

ASSIGNMENTS: list[str] = [
    'Assignment_1',
    'Assignment_2',
    'Assignment_3',
    'Assignment_4',   # not every semester will have this; handled gracefully
    'Project_Paper',
    'Qualifier_Question',
]

CHATGPT_CUTOFF_YYYYMM: int = 202208
MAX_CHARS_PER_PAGE: int    = 50_000

# =============================================================================
# 6. Data model
# =============================================================================

@dataclass(frozen=True)
class Semester:
    yyyymm:    int
    code:      str
    canvas_id: str
    course:    str
    path:      Path

    @property
    def era(self) -> str:
        return 'Pre-ChatGPT' if self.yyyymm <= CHATGPT_CUTOFF_YYYYMM else 'Post-ChatGPT'

    @property
    def label(self) -> str:
        return self.code


@dataclass
class EssayRecord:
    semester:     Semester
    assignment:   str
    student_id:   str
    student_name: str
    pdf_path:     Path
    text:         str = field(default='', repr=False)

    @property
    def uid(self) -> str:
        return f'{self.semester.canvas_id}__{self.assignment}__{self.student_id}'

# =============================================================================
# 7. Folder parsing
# =============================================================================

_FOLDER_RE = re.compile(
    r'^(?P<yyyymm>\d{6})'
    r'-(?P<course_slug>.+?)'
    r'(?:_\((?P<code>(?:SP|SU|FA)\d{2})\))?'
    r'-(?P<canvas_id>\d+)$'
)


def _match_course(course_slug: str) -> str | None:
    slug_lower = course_slug.lower()
    for substring, label in COURSES:
        if substring.lower() in slug_lower:
            return label
    return None


def _match_assignment(folder_name: str) -> str | None:
    name_lower = folder_name.lower()
    if not ASSIGNMENTS:
        _, _, rest = folder_name.partition('_')
        return rest or folder_name
    for assignment in ASSIGNMENTS:
        if assignment.lower() in name_lower:
            return assignment
    return None


def discover_semesters(data_root: Path) -> list[Semester]:
    semesters: list[Semester] = []
    for entry in sorted(data_root.iterdir()):
        if not entry.is_dir():
            continue
        m = _FOLDER_RE.match(entry.name)
        if not m:
            continue
        course_label = _match_course(m.group('course_slug'))
        if course_label is None:
            continue
        semesters.append(Semester(
            yyyymm    = int(m.group('yyyymm')),
            code      = m.group('code'),
            canvas_id = m.group('canvas_id'),
            course    = course_label,
            path      = entry,
        ))
    return sorted(semesters, key=lambda s: (s.yyyymm, s.canvas_id))


def iter_essay_records(semesters: list[Semester]) -> Iterator[EssayRecord]:
    """
    Yield one EssayRecord per PDF found under
    semester/submissions/<assignment>/<student>/<file>.pdf.

    For every configured assignment that has no matching folder inside a
    semester, a clear CLI message is printed and iteration continues.
    """
    sub_re = re.compile(r'^\d+_')

    for sem in semesters:
        submissions_dir = sem.path / 'submissions'
        if not submissions_dir.is_dir():
            print(f'  [warn] no submissions/ folder in {sem.path.name}')
            continue

        # Build a map of assignment_name -> folder for fast lookup
        found_assign_dirs: dict[str, Path] = {}
        for d in submissions_dir.iterdir():
            if not d.is_dir():
                continue
            matched = _match_assignment(d.name)
            if matched:
                found_assign_dirs[matched] = d

        # Report configured assignments that are absent for this semester
        for configured_assignment in ASSIGNMENTS:
            if configured_assignment not in found_assign_dirs:
                print(
                    f'  [info] {sem.code} has no {configured_assignment}'
                )

        # Yield records for assignments that do exist
        for assignment, assign_dir in sorted(found_assign_dirs.items()):
            for student_dir in sorted(assign_dir.iterdir()):
                if not student_dir.is_dir() or not sub_re.match(student_dir.name):
                    continue
                student_id, _, student_name = student_dir.name.partition('_')
                pdfs = list(student_dir.glob('*.pdf'))
                if not pdfs:
                    print(f'  [warn] no PDF in {student_dir}')
                    continue
                if len(pdfs) > 1:
                    print(
                        f'  [warn] multiple PDFs in {student_dir}; '
                        f'using the last alphabetically' # Change the message here if needed
                    )
                yield EssayRecord(
                    semester     = sem,
                    assignment   = assignment,
                    student_id   = student_id,
                    student_name = student_name,
                    pdf_path     = sorted(pdfs)[-1], # 0 for first, -1 for last
                )

# =============================================================================
# 8. PDF extraction
# =============================================================================

def extract_text_from_pdf(path: Path, *, max_chars: int = MAX_CHARS_PER_PAGE) -> str:
    try:
        pages: list[str] = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                raw = page.extract_text(x_tolerance=2, y_tolerance=2) or ''
                pages.append(raw[:max_chars])
        full = '\n\n'.join(pages)
        full = re.sub(r'\n{3,}', '\n\n', full)
        return full.strip()
    except Exception as exc:
        print(f'  [warn] could not extract {path.name}: {exc}')
        return ''


def load_records(records: list[EssayRecord]) -> list[EssayRecord]:
    loaded: list[EssayRecord] = []
    for rec in records:
        text = extract_text_from_pdf(rec.pdf_path)
        if not text:
            print(f'  [warn] empty text for {rec.uid}; skipping')
            continue
        rec.text = text
        loaded.append(rec)
    return loaded

# =============================================================================
# 9. Demo mode (synthetic data)
# =============================================================================

_PRE_AI_TEMPLATE = textwrap.dedent('''\
    {topic} is an important subject in educational technology.
    There are several reasons why this matters. First, {reason1}.
    Second, {reason2}. I think this shows that {conclusion}.

    Robespierre is an interesting figure because he seemed to believe in
    what he was doing. Whether that was a good thing is still debated.

    In the end, {final}.
''')

_POST_AI_TEMPLATE = textwrap.dedent('''\
    {topic} represents a pivotal and multifaceted dimension of educational
    technology. It is important to note that this transformative landscape
    encompasses several crucial aspects. Furthermore, {reason1}.

    Moreover, {reason2}. Navigating the complex intersection of theory and
    practice, scholars have fostered a comprehensive understanding of the
    paradigm. Additionally, this robust framework underscores the vibrant
    ecosystem of modern pedagogy.

    In conclusion, {topic} is not merely a concept but a testament to the
    broader human endeavour for knowledge. Its intricate tapestry of causes
    and consequences continues to illuminate our understanding of {final}.
''')

_FILLERS = {
    'topic':      ['personalised learning', 'AI in classrooms', 'peer assessment',
                   'gamification', 'adaptive systems'],
    'reason1':    ['students learn better with feedback', 'technology reduces barriers',
                   'engagement metrics improve outcomes'],
    'reason2':    ['the evidence supports early intervention', 'costs decrease over time',
                   'teachers report higher satisfaction'],
    'conclusion': ['this field is worth further study', 'policy should reflect these findings',
                   'more research is needed'],
    'final':      ['education will continue to evolve', 'the future remains uncertain',
                   'collaboration is key'],
}

_FAKE_STUDENTS = [
    ('88001', 'Amal_Mohsin'),
    ('88002', 'Aqsa_Ambreen'),
    ('88003', 'Hiba_Aamir'),
    ('88004', 'Nayab_Khizar'),
    ('88005', 'Noor_Mirza'),
]


def _random_fill(template: str, rng: random.Random) -> str:
    return template.format(**{k: rng.choice(v) for k, v in _FILLERS.items()})


def _write_synthetic_pdf(text: str, path: Path) -> None:
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Helvetica', size=6)
    pdf.set_auto_page_break(auto=True, margin=15)
    for line in text.splitlines():
        if line.strip():
            pdf.multi_cell(w=0, h=3, text=line)
            pdf.ln(2)
    pdf.output(str(path))


def build_demo_corpus(root: Path, *, random_seed: int = 42) -> None:
    """
    Construct a synthetic dataset under *root*.

    Per-assignment student count: randomly 3–5 (reproducible via random_seed).
    Assignment_4 presence: decided per-semester by a coin flip so that the
    testing team can observe both the 'has Assignment_4' and
    '{term} has no Assignment_4' paths in a single run.
    """
    rng = random.Random(random_seed)

    fake_semesters = [
        ('202102', 'SP21', '100001'),
        ('202105', 'SU21', '100002'),
        ('202108', 'FA21', '100003'),
        ('202202', 'SP22', '100004'),
        ('202208', 'FA22', '100005'),
        ('202302', 'SP23', '100006'),
        ('202308', 'FA23', '100007'),
    ]

    # Base assignments that every semester always receives
    base_assignments = [a for a in ASSIGNMENTS if a != 'Assignment_4']
    course_slug      = COURSES[0][0] if COURSES else 'CS6460'

    for yyyymm, code, canvas_id in fake_semesters:
        era_cutoff = int(yyyymm) <= CHATGPT_CUTOFF_YYYYMM
        template   = _PRE_AI_TEMPLATE if era_cutoff else _POST_AI_TEMPLATE

        # Coin flip: does this semester have Assignment_4?
        semester_assignments = base_assignments.copy()
        has_a4 = rng.random() < 0.5
        if has_a4:
            semester_assignments.append('Assignment_4')

        sem_dir = root / (
            f'{yyyymm}-{course_slug}_Educational_Technology_({code})-{canvas_id}'
        )

        for i, assignment in enumerate(semester_assignments):
            assign_prefix = str(900000 + i)
            assign_dir    = sem_dir / 'submissions' / f'{assign_prefix}_{assignment}'

            # Per-assignment student count: 3–5
            n_students   = rng.randint(3, 5)
            chosen_students = _FAKE_STUDENTS[:n_students]

            for sid, sname in chosen_students:
                student_dir = assign_dir / f'{sid}_{sname}'
                student_dir.mkdir(parents=True, exist_ok=True)
                pdf_path = student_dir / 'submission.pdf'
                if not pdf_path.exists():
                    body = _random_fill(template, rng)
                    body = (
                        body
                        + (' ' + rng.choice(list(_FILLERS['conclusion'])))
                        * rng.randint(0, 3)
                    )
                    _write_synthetic_pdf(body, pdf_path)

    print(f'[demo] Synthetic dataset written to: {root}')

# =============================================================================
# 10. Corpus assembly
# =============================================================================

LabelledCorpus = dict  # dict[str, list[str]]


def assemble_corpus(
    records: list[EssayRecord],
    *,
    group_by: str = 'era',
) -> LabelledCorpus:
    corpus: dict[str, list[str]] = {}
    for rec in records:
        match group_by:
            case 'era':
                key = rec.semester.era
            case 'semester':
                key = rec.semester.label
            case 'assignment':
                key = rec.assignment
            case _:
                raise ValueError(
                    f"group_by must be 'era', 'semester', or 'assignment'; "
                    f"got '{group_by}'"
                )
        corpus.setdefault(key, []).append(rec.text)
    return corpus


def summarise_corpus(records: list[EssayRecord]) -> pd.DataFrame:
    rows = [
        {
            'semester':   rec.semester.code,
            'yyyymm':     rec.semester.yyyymm,
            'era':        rec.semester.era,
            'assignment': rec.assignment,
            'student_id': rec.student_id,
            'word_count': len(rec.text.split()),
            'uid':        rec.uid,
        }
        for rec in records
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    print(
        df.groupby(['era', 'semester', 'assignment'])
          .size()
          .rename('n_essays')
          .to_string()
    )
    return df

# =============================================================================
# 11. Diagnostic helpers
# =============================================================================

def print_architecture_terminal(data_root: Path) -> None:
    """
    Print an ASCII directory tree of *data_root*.
    Only folder and file names are shown; file contents are not read.
    """
    print()
    print('DATA ARCHITECTURE')
    print('=' * 60)
    print(data_root.resolve())

    def _walk(path: Path, prefix: str = '') -> None:
        entries = sorted(path.iterdir())
        for i, entry in enumerate(entries):
            connector = '|-- '
            extension = '|   ' if i < len(entries) - 1 else '    '
            print(f'{prefix}{connector}{entry.name}')
            if entry.is_dir():
                _walk(entry, prefix + extension)

    _walk(data_root)
    print()


def print_architecture(data_root: Path) -> None:
    out_path = OUTPUT_DIR / 'structure.txt'
    lines: list[str] = []
    lines.append(f'[DIR]  {data_root.resolve()}/')

    def _walk(path: Path, prefix: str = '') -> None:
        entries = sorted(path.iterdir())
        for i, entry in enumerate(entries):
            connector  = '|-- '
            extension  = '|   ' if i < len(entries) - 1 else '    '
            tag        = '[DIR] ' if entry.is_dir() else '[FILE]'
            suffix     = '/' if entry.is_dir() else ''
            lines.append(f'{prefix}{connector}{tag}  {entry.name}{suffix}')
            if entry.is_dir():
                _walk(entry, prefix + extension)

    _walk(data_root)
    out_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Done! Check {out_path.resolve()}')


def print_assignments_terminal(
    semesters:  list[Semester],
    records:    list[EssayRecord],
) -> None:
    """
    For each semester, print the first assignment's first student submission. (Last submission for the student if they have multiple pdfs)
    Shows: semester code, assignment name, and extracted text content.
    """
    # Index records by (semester_canvas_id, assignment) for fast lookup
    record_index: dict[tuple[str, str], EssayRecord] = {}
    for rec in records:
        key = (rec.semester.canvas_id, rec.assignment)
        if key not in record_index:
            record_index[key] = rec

    print()
    print('ASSIGNMENT CONTENTS (first assignment, first student per semester, last submission for the student if they have multiple pdfs)')
    print('=' * 60)

    for sem in semesters:
        submissions_dir = sem.path / 'submissions'
        if not submissions_dir.is_dir():
            continue

        # Find the first assignment folder alphabetically
        assign_dirs = sorted(
            d for d in submissions_dir.iterdir() if d.is_dir()
        )
        if not assign_dirs:
            print(f'\n[{sem.code}] No assignment folders found.')
            continue

        first_assign_dir  = assign_dirs[0]
        assignment_name   = _match_assignment(first_assign_dir.name) or first_assign_dir.name

        key = (sem.canvas_id, assignment_name)
        rec = record_index.get(key)

        print(f'\nSemester   : {sem.code}  ({sem.era})')
        print(f'Assignment : {assignment_name}')
        print('-' * 60)

        if rec is None or not rec.text:
            print('  [no text available for this assignment]')
        else:
            # Wrap text for readable terminal output
            wrapped = textwrap.fill(rec.text, width=72, initial_indent='  ',
                                    subsequent_indent='  ')
            print(wrapped)

        print()


def print_assignments(
    semesters: list[Semester],
    records:   list[EssayRecord],
) -> None:
    out_path = OUTPUT_DIR / 'assignments.txt'
    lines: list[str] = []
    lines.append('ASSIGNMENT CONTENTS (first assignment, first student per semester, last submission for the student if they have multiple pdfs)')
    lines.append('=' * 60)

    record_index: dict[tuple[str, str], EssayRecord] = {}
    for rec in records:
        key = (rec.semester.canvas_id, rec.assignment)
        if key not in record_index:
            record_index[key] = rec

    for sem in semesters:
        submissions_dir = sem.path / 'submissions'
        if not submissions_dir.is_dir():
            continue

        assign_dirs = sorted(
            d for d in submissions_dir.iterdir() if d.is_dir()
        )
        if not assign_dirs:
            lines.append(f'\n[{sem.code}] No assignment folders found.')
            continue

        first_assign_dir = assign_dirs[0]
        assignment_name  = (
            _match_assignment(first_assign_dir.name) or first_assign_dir.name
        )
        key = (sem.canvas_id, assignment_name)
        rec = record_index.get(key)

        lines.append('')
        lines.append(f'Semester   : {sem.code}  ({sem.era})')
        lines.append(f'Assignment : {assignment_name}')
        lines.append('-' * 60)

        if rec is None or not rec.text:
            lines.append('  [no text available for this assignment]')
        else:
            wrapped = textwrap.fill(
                rec.text, width=72,
                initial_indent='  ', subsequent_indent='  ',
            )
            lines.append(wrapped)

        lines.append('')

    out_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Done! Check {out_path.resolve()}')

# =============================================================================
# 12. Ingestion entry point
# =============================================================================

def run_ingestion(demo_mode: bool) -> tuple[list[Semester], list[EssayRecord], pd.DataFrame, LabelledCorpus]:
    """
    Resolve data root, discover semesters, load records, assemble corpus.
    Returns early with empty structures if the data root does not exist
    (vacuous success for no-data + no-print runs).
    """
    if demo_mode:
        print('[demo] Demo mode ON -- generating synthetic data.')
        _demo_root = Path(tempfile.mkdtemp(prefix='essay_demo_'))
        build_demo_corpus(_demo_root)
        data_root = _demo_root
    else:
        data_root = Path(DATA_ROOT)
        if not data_root.exists():
            print(
                f'[info] Data root "{data_root}" not found. '
                f'Skipping ingestion (use --use-fake-data for a dry run).'
            )
            return [], [], pd.DataFrame(), {}

    print(f'[info] Loading from: {data_root}')

    semesters   = discover_semesters(data_root)
    print(f'[info] Found {len(semesters)} matching semester folder(s).')

    raw_records = list(iter_essay_records(semesters))
    print(f'[info] Discovered {len(raw_records)} submission(s).')

    records     = load_records(raw_records)
    print(f'[info] Loaded {len(records)} submission(s) with non-empty text.')

    summary_df  = summarise_corpus(records)
    summary_df.to_csv(OUTPUT_DIR / 'corpus_summary.csv', index=False)

    corpus = assemble_corpus(records, group_by='era')
    print(f'[info] Corpus groups: { {k: len(v) for k, v in corpus.items()} }')

    return semesters, records, summary_df, corpus

# =============================================================================
# 13. NLP setup
# =============================================================================

for _resource in ('punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'stopwords'):
    try:
        nltk.data.find(f'tokenizers/{_resource}')
    except LookupError:
        nltk.download(_resource, quiet=True)

_nlp = spacy.load('en_core_web_sm')

UAF = {
    'blue':   '#403F84',
    'sky':    '#2F9FD9',
    'yellow': '#F8D727',
    'green':  '#005C45',
}
_PALETTE_TWO  = [UAF['blue'], UAF['sky']]
_PALETTE_FOUR = [UAF['blue'], UAF['sky'], UAF['yellow'], UAF['green']]

sns.set_theme(style='whitegrid', palette=sns.color_palette(_PALETTE_FOUR))
plt.rcParams.update({'figure.dpi': 120, 'font.size': 11})

# =============================================================================
# 14. EssayMetrics dataclass
# =============================================================================

@dataclass
class EssayMetrics:
    label:                          str
    colon_count:                    int   = 0
    semicolon_count:                int   = 0
    colon_semicolon_ratio:          float = 0.0
    sentence_lengths:               list[int]   = field(default_factory=list)
    mean_sentence_length:           float = 0.0
    sentence_length_std:            float = 0.0
    hedge_frequency:                float = 0.0
    discourse_marker_frequency:     float = 0.0
    ttr:                            float = 0.0
    gptism_frequency:               float = 0.0
    gptism_counts:                  dict[str, int] = field(default_factory=dict)
    paragraph_lengths:              list[int]   = field(default_factory=list)
    paragraph_length_std:           float = 0.0
    signpost_frequency:             float = 0.0
    nuance_trap_frequency:          float = 0.0
    intro_outro_similarity:         float = 0.0
    mean_intra_paragraph_similarity: float = 0.0
    intra_paragraph_similarities:   list[float] = field(default_factory=list)

# =============================================================================
# 15. Vocabulary lists
# =============================================================================

HEDGE_WORDS: frozenset[str] = frozenset({
    'arguably', 'potentially', 'largely', 'generally', 'typically',
    'usually', 'often', 'somewhat', 'perhaps', 'possibly', 'likely',
    'seemingly', 'apparently', 'presumably', 'to some extent',
    'it is important to note', 'it should be noted', 'it is worth noting',
    'one could argue', 'one might argue',
})

DISCOURSE_MARKERS: frozenset[str] = frozenset({
    'furthermore', 'moreover', 'additionally', 'in conclusion',
    'in summary', 'to summarise', 'firstly', 'secondly', 'thirdly',
    'lastly', 'finally', 'in addition', 'nevertheless', 'nonetheless',
    'consequently', 'therefore', 'thus', 'hence',
})

GPTISMS: dict[str, list[str]] = {
    'verbs':      ['delve', 'underscore', 'foster', 'enhance', 'navigate',
                   'utilise', 'leverage', 'facilitate', 'illuminate', 'embody'],
    'adjectives': ['pivotal', 'multifaceted', 'transformative', 'vibrant',
                   'comprehensive', 'nuanced', 'robust', 'dynamic', 'crucial',
                   'intricate', 'profound'],
    'nouns':      ['landscape', 'testament', 'tapestry', 'intersection',
                   'framework', 'paradigm', 'ecosystem', 'dimension', 'realm'],
}

_ALL_GPTISMS: frozenset[str] = frozenset(
    w for words in GPTISMS.values() for w in words
)

# =============================================================================
# 16. Preprocessing helpers
# =============================================================================

def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]


def tokenise_words(text: str, lowercase: bool = True) -> list[str]:
    tokens = word_tokenize(text.lower() if lowercase else text)
    return [t for t in tokens if t not in string.punctuation]


def count_phrase_occurrences(text: str, phrases: frozenset[str]) -> int:
    text_lower = text.lower()
    total = 0
    for phrase in phrases:
        if ' ' in phrase:
            total += text_lower.count(phrase)
        else:
            total += len(re.findall(rf'\b{re.escape(phrase)}\b', text_lower))
    return total

# =============================================================================
# 17. Metric computation
# =============================================================================

def compute_punctuation_metrics(text: str, metrics: EssayMetrics) -> None:
    metrics.colon_count     = text.count(':')
    metrics.semicolon_count = text.count(';')
    metrics.colon_semicolon_ratio = (
        metrics.colon_count / (metrics.semicolon_count + 1)
    )


def compute_sentence_metrics(text: str, metrics: EssayMetrics) -> None:
    sentences = sent_tokenize(text)
    lengths   = [len(word_tokenize(s)) for s in sentences]
    metrics.sentence_lengths     = lengths
    if lengths:
        metrics.mean_sentence_length = float(np.mean(lengths))
        metrics.sentence_length_std  = float(np.std(lengths))


def compute_hedge_metrics(
    text: str, metrics: EssayMetrics, word_count: int,
) -> None:
    hedge_hits     = count_phrase_occurrences(text, HEDGE_WORDS)
    discourse_hits = count_phrase_occurrences(text, DISCOURSE_MARKERS)
    per_100 = 100 / max(word_count, 1)
    metrics.hedge_frequency            = hedge_hits    * per_100
    metrics.discourse_marker_frequency = discourse_hits * per_100


def compute_lexical_metrics(
    tokens: list[str], text: str, metrics: EssayMetrics,
) -> None:
    if tokens:
        metrics.ttr = len(set(tokens)) / len(tokens)
    gptism_counts: dict[str, int] = {}
    total_gptisms = 0
    for word in _ALL_GPTISMS:
        n = len(re.findall(rf'\b{re.escape(word)}\b', text.lower()))
        if n:
            gptism_counts[word] = n
            total_gptisms += n
    metrics.gptism_counts    = gptism_counts
    metrics.gptism_frequency = total_gptisms * (100 / max(len(tokens), 1))


def compute_structural_metrics(
    text: str,
    paragraphs: list[str],
    metrics: EssayMetrics,
    word_count: int,
) -> None:
    para_lengths = [len(word_tokenize(p)) for p in paragraphs]
    metrics.paragraph_lengths    = para_lengths
    metrics.paragraph_length_std = (
        float(np.std(para_lengths)) if para_lengths else 0.0
    )
    signpost_hits = count_phrase_occurrences(text, DISCOURSE_MARKERS)
    metrics.signpost_frequency = signpost_hits * (100 / max(word_count, 1))
    nuance_pattern = re.compile(
        r'\b(while|although|even though|despite|whereas)\b', re.IGNORECASE,
    )
    metrics.nuance_trap_frequency = (
        len(nuance_pattern.findall(text)) * (100 / max(word_count, 1))
    )
    if len(paragraphs) >= 2:
        metrics.intro_outro_similarity = _paragraph_cosine_similarity(
            paragraphs[0], paragraphs[-1]
        )


def _paragraph_cosine_similarity(para_a: str, para_b: str) -> float:
    vectoriser = TfidfVectorizer().fit([para_a, para_b])
    vecs = vectoriser.transform([para_a, para_b])
    return float(cosine_similarity(vecs[0], vecs[1])[0, 0])


def compute_semantic_redundancy(
    paragraphs: list[str], metrics: EssayMetrics,
) -> None:
    all_sims: list[float] = []
    for para in paragraphs:
        sents = sent_tokenize(para)
        if len(sents) < 2:
            continue
        try:
            vectoriser = TfidfVectorizer().fit(sents)
            vecs = vectoriser.transform(sents).toarray()
        except ValueError:
            continue
        n    = len(sents)
        sims = [
            float(1 - cosine(vecs[i], vecs[j]))
            for i in range(n)
            for j in range(i + 1, n)
            if np.any(vecs[i]) and np.any(vecs[j])
        ]
        if sims:
            all_sims.append(float(np.mean(sims)))
    metrics.intra_paragraph_similarities    = all_sims
    metrics.mean_intra_paragraph_similarity = (
        float(np.mean(all_sims)) if all_sims else 0.0
    )

# =============================================================================
# 18. Orchestrator
# =============================================================================

def analyse_essay(text: str, label: str = 'essay') -> EssayMetrics:
    metrics    = EssayMetrics(label=label)
    paragraphs = split_paragraphs(text)
    tokens     = tokenise_words(text)
    word_count = len(tokens)
    compute_punctuation_metrics(text, metrics)
    compute_sentence_metrics(text, metrics)
    compute_hedge_metrics(text, metrics, word_count)
    compute_lexical_metrics(tokens, text, metrics)
    compute_structural_metrics(text, paragraphs, metrics, word_count)
    compute_semantic_redundancy(paragraphs, metrics)
    return metrics


def metrics_to_series(m: EssayMetrics) -> pd.Series:
    return pd.Series({
        'label':                      m.label,
        'colon_semicolon_ratio':      m.colon_semicolon_ratio,
        'mean_sentence_length':       m.mean_sentence_length,
        'sentence_length_std':        m.sentence_length_std,
        'hedge_frequency':            m.hedge_frequency,
        'discourse_marker_frequency': m.discourse_marker_frequency,
        'ttr':                        m.ttr,
        'gptism_frequency':           m.gptism_frequency,
        'paragraph_length_std':       m.paragraph_length_std,
        'signpost_frequency':         m.signpost_frequency,
        'nuance_trap_frequency':      m.nuance_trap_frequency,
        'intro_outro_similarity':     m.intro_outro_similarity,
        'mean_intra_para_similarity': m.mean_intra_paragraph_similarity,
    })

# =============================================================================
# 19. Comparison functions
# =============================================================================

def compare_texts(
    text1: str,
    text2: str,
    label1: str = 'Text 1',
    label2: str = 'Text 2',
) -> tuple[EssayMetrics, EssayMetrics, pd.DataFrame]:
    m1 = analyse_essay(text1, label=label1)
    m2 = analyse_essay(text2, label=label2)
    s1 = metrics_to_series(m1)
    s2 = metrics_to_series(m2)
    numeric_keys = [k for k in s1.index if k != 'label']
    comparison   = pd.DataFrame({
        label1: s1[numeric_keys].astype(float),
        label2: s2[numeric_keys].astype(float),
    })
    comparison['delta_2_minus_1'] = comparison[label2] - comparison[label1]
    comparison['delta_pct'] = (
        comparison['delta_2_minus_1']
        / comparison[label1].replace(0, np.nan) * 100
    ).round(1)
    return m1, m2, comparison


def compare_corpora(
    corpus1: list[str],
    corpus2: list[str],
    label1:  str = 'Corpus 1',
    label2:  str = 'Corpus 2',
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows1 = [metrics_to_series(analyse_essay(t, label=label1)) for t in corpus1]
    rows2 = [metrics_to_series(analyse_essay(t, label=label2)) for t in corpus2]
    df    = pd.DataFrame(rows1 + rows2)
    numeric_cols = df.columns.difference(['label'])
    df[numeric_cols] = df[numeric_cols].astype(float)
    summary_rows = []
    for col in numeric_cols:
        g1 = df.loc[df['label'] == label1, col].dropna()
        g2 = df.loc[df['label'] == label2, col].dropna()
        stat, p = (
            mannwhitneyu(g1, g2, alternative='two-sided')
            if (len(g1) > 1 and len(g2) > 1)
            else (np.nan, np.nan)
        )
        summary_rows.append({
            'metric':          col,
            f'{label1}_mean':  g1.mean(),
            f'{label1}_std':   g1.std(),
            f'{label2}_mean':  g2.mean(),
            f'{label2}_std':   g2.std(),
            'mann_whitney_u':  stat,
            'p_value':         p,
            'significant_p05': p < 0.05 if not np.isnan(p) else False,
        })
    summary = pd.DataFrame(summary_rows).set_index('metric')
    return df, summary

# =============================================================================
# 20. Visualisation
# =============================================================================

def _bar_comparison(
    ax: plt.Axes,
    labels: list[str],
    values: list[float],
    metric_name: str,
    colours: Optional[list[str]] = None,
) -> None:
    colours = colours or _PALETTE_TWO
    bars = ax.bar(labels, values, color=colours[:len(labels)], width=0.5)
    ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=9)
    ax.set_title(metric_name.replace('_', ' ').title(), fontsize=10)
    ax.set_ylabel('Score')


def _sentence_length_histogram(
    ax: plt.Axes, m1: EssayMetrics, m2: EssayMetrics,
) -> None:
    bins = np.linspace(
        0,
        max(
            max(m1.sentence_lengths, default=1),
            max(m2.sentence_lengths, default=1),
        ) + 5,
        20,
    )
    ax.hist(m1.sentence_lengths, bins=bins, alpha=0.7,
            label=m1.label, color=_PALETTE_TWO[0])
    ax.hist(m2.sentence_lengths, bins=bins, alpha=0.7,
            label=m2.label, color=_PALETTE_TWO[1])
    ax.set_title('Sentence Length Distribution (Burstiness)')
    ax.set_xlabel('Words per sentence')
    ax.set_ylabel('Count')
    ax.legend()


_PAIRWISE_SCALAR_METRICS: list[str] = [
    'colon_semicolon_ratio',
    'mean_sentence_length',
    'sentence_length_std',
    'hedge_frequency',
    'discourse_marker_frequency',
    'ttr',
    'gptism_frequency',
    'paragraph_length_std',
    'intro_outro_similarity',
    'mean_intra_paragraph_similarity',
]


def plot_pairwise(
    m1: EssayMetrics, m2: EssayMetrics, figsize: tuple = (18, 14),
) -> plt.Figure:
    n_scalar   = len(_PAIRWISE_SCALAR_METRICS)
    n_bar_rows = (n_scalar + 1) // 2
    total_rows = n_bar_rows + 2
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    fig.suptitle(
        f'Essay Metric Comparison: {m1.label} vs {m2.label}',
        fontsize=14, fontweight='bold',
    )
    gs = gridspec.GridSpec(total_rows, 2, figure=fig)
    for idx, metric in enumerate(_PAIRWISE_SCALAR_METRICS):
        row, col = divmod(idx, 2)
        ax = fig.add_subplot(gs[row, col])
        _bar_comparison(
            ax,
            labels=[m1.label, m2.label],
            values=[getattr(m1, metric), getattr(m2, metric)],
            metric_name=metric,
        )
    ax_hist = fig.add_subplot(gs[n_bar_rows, :])
    _sentence_length_histogram(ax_hist, m1, m2)
    ax_gpt = fig.add_subplot(gs[n_bar_rows + 1, :])
    _plot_gptism_breakdown(ax_gpt, m1, m2)
    return fig


def _plot_gptism_breakdown(
    ax: plt.Axes, m1: EssayMetrics, m2: EssayMetrics,
) -> None:
    all_words = sorted(set(m1.gptism_counts) | set(m2.gptism_counts))
    if not all_words:
        ax.set_visible(False)
        return
    x       = np.arange(len(all_words))
    width   = 0.35
    counts1 = [m1.gptism_counts.get(w, 0) for w in all_words]
    counts2 = [m2.gptism_counts.get(w, 0) for w in all_words]
    ax.bar(x - width / 2, counts1, width, label=m1.label, color=_PALETTE_TWO[0])
    ax.bar(x + width / 2, counts2, width, label=m2.label, color=_PALETTE_TWO[1])
    ax.set_xticks(x)
    ax.set_xticklabels(all_words, rotation=35, ha='right', fontsize=9)
    ax.set_title('GPT-ism Word Counts')
    ax.set_ylabel('Count')
    ax.legend()


def plot_corpus_summary(
    summary: pd.DataFrame,
    label1:  str   = 'Corpus 1',
    label2:  str   = 'Corpus 2',
    figsize: tuple = (14, 10),
) -> plt.Figure:
    metrics = summary.index.tolist()
    means1  = summary[f'{label1}_mean'].values
    means2  = summary[f'{label2}_mean'].values
    stds1   = summary[f'{label1}_std'].values
    stds2   = summary[f'{label2}_std'].values
    sig     = summary['significant_p05'].values
    y       = np.arange(len(metrics))
    height  = 0.35
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.barh(y + height / 2, means1, height, xerr=stds1, label=label1,
            color=_PALETTE_TWO[0], alpha=0.9, capsize=3)
    ax.barh(y - height / 2, means2, height, xerr=stds2, label=label2,
            color=_PALETTE_TWO[1], alpha=0.9, capsize=3)
    for i, (metric, is_sig) in enumerate(zip(metrics, sig)):
        if is_sig:
            ax.text(
                max(means1[i] + stds1[i], means2[i] + stds2[i]) * 1.02,
                i, '* p<.05', va='center', fontsize=8, color='#c00000',
            )
    ax.set_yticks(y)
    ax.set_yticklabels([m.replace('_', ' ').title() for m in metrics])
    ax.set_xlabel('Mean score')
    ax.set_title(
        f'Corpus Metric Comparison: {label1} vs {label2}', fontweight='bold',
    )
    ax.legend()
    ax.invert_yaxis()
    return fig


def plot_sentence_burstiness_corpora(
    df:      pd.DataFrame,
    metric:  str   = 'sentence_length_std',
    label1:  str   = 'Corpus 1',
    label2:  str   = 'Corpus 2',
    figsize: tuple = (8, 5),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    sns.boxplot(data=df, x='label', y=metric, ax=ax, order=[label1, label2],
                palette=_PALETTE_TWO, width=0.4)
    sns.stripplot(data=df, x='label', y=metric, ax=ax, order=[label1, label2],
                  color='black', alpha=0.5, size=4, jitter=True)
    ax.set_title(
        f'{metric.replace("_", " ").title()} by Corpus', fontweight='bold',
    )
    ax.set_xlabel('')
    ax.set_ylabel(metric.replace('_', ' ').title())
    return fig

# =============================================================================
# 21. Fixed essay examples
# =============================================================================

_PRE_AI_ESSAY = '''
The French Revolution fundamentally changed European politics. People were angry
about inequality and bread prices. The king didn't really understand what was
happening until it was too late. There were a lot of different causes - economic,
social, and political - but they all kind of built up together.

Robespierre is an interesting figure because he seemed to believe in what he was
doing even when it got violent. I think that's what makes him different from
someone who was just power-hungry.

In the end, the Revolution opened the door for Napoleon. Whether that was a good
thing or not is still debated.
'''

_POST_AI_ESSAY = '''
The French Revolution was a pivotal moment in European history that fundamentally
transformed the political landscape of the continent. It is important to note that
this transformative event was multifaceted in its causes; economic hardship,
social inequality, and political instability all played crucial roles.

Furthermore, the Revolution fostered a new paradigm of governance that would
underscore subsequent democratic movements. Moreover, figures such as Robespierre
navigated the complex intersection of idealism and pragmatism, leaving a
comprehensive legacy that continues to resonate.

In conclusion, the French Revolution was not merely a national event but a
testament to the broader human struggle for liberty. Its vibrant tapestry of
causes and consequences continues to shape our understanding of modern democracy.
'''

# =============================================================================
# 22. Main
# =============================================================================

def main() -> None:
    # ------------------------------------------------------------------ ingest
    semesters, records, summary_df, corpus = run_ingestion(ARGS.use_fake_data)

    # ----------------------------------------------------------- diagnostics
    if ARGS.print_architecture:
        # Resolve the data root identically to run_ingestion
        if ARGS.use_fake_data:
            # build_demo_corpus already ran inside run_ingestion;
            # re-derive the root from the first semester's parent^2
            if semesters:
                data_root = semesters[0].path.parent.parent
            else:
                print('[info] No semesters found; nothing to show.')
                data_root = None
        else:
            data_root = Path(DATA_ROOT) if Path(DATA_ROOT).exists() else None

        if data_root:
            print_architecture(data_root)

    if ARGS.print_assignments:
        if semesters and records:
            print_assignments(semesters, records)
        else:
            print('[info] No data loaded; --print-assignments has nothing to show.')

    # --------------------------------------------------------- guard: no data
    if not records:
        print('[info] No records loaded. Exiting without analysis.')
        return

    # ------------------------------------------------------- pairwise analysis
    m1, m2, comparison_df = compare_texts(
        _PRE_AI_ESSAY, _POST_AI_ESSAY,
        label1='Pre-ChatGPT', label2='Post-ChatGPT',
    )

    print('=' * 60)
    print('PAIRWISE METRIC COMPARISON')
    print('=' * 60)
    print(comparison_df.to_string(float_format='{:.3f}'.format))

    comparison_df.to_csv(OUTPUT_DIR / 'pairwise_comparison.csv')

    fig_pair = plot_pairwise(m1, m2)
    fig_pair.savefig(
        OUTPUT_DIR / 'pairwise_dashboard.png', dpi=120, bbox_inches='tight',
    )
    plt.close(fig_pair)

    # --------------------------------------------------------- corpus analysis
    pre_corpus  = corpus.get('Pre-ChatGPT',  [])
    post_corpus = corpus.get('Post-ChatGPT', [])

    corpus_df, summary_stats_df = compare_corpora(
        pre_corpus, post_corpus,
        label1='Pre-ChatGPT', label2='Post-ChatGPT',
    )

    print('\n' + '=' * 60)
    print('CORPUS-LEVEL SUMMARY')
    print('=' * 60)
    print(summary_stats_df.to_string(float_format='{:.3f}'.format))

    corpus_df.to_csv(OUTPUT_DIR / 'corpus_per_essay_metrics.csv', index=False)
    summary_stats_df.to_csv(OUTPUT_DIR / 'corpus_summary_stats.csv')

    fig_corpus = plot_corpus_summary(
        summary_stats_df, label1='Pre-ChatGPT', label2='Post-ChatGPT',
    )
    fig_corpus.savefig(
        OUTPUT_DIR / 'corpus_summary.png', dpi=120, bbox_inches='tight',
    )
    plt.close(fig_corpus)

    fig_burst = plot_sentence_burstiness_corpora(
        corpus_df, metric='sentence_length_std',
        label1='Pre-ChatGPT', label2='Post-ChatGPT',
    )
    fig_burst.savefig(
        OUTPUT_DIR / 'burstiness.png', dpi=120, bbox_inches='tight',
    )
    plt.close(fig_burst)

    print(f'\n[done] All results written to {OUTPUT_DIR.resolve()}')


if __name__ == '__main__':
    main()
