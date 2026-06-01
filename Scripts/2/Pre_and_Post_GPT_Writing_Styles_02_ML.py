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
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Protocol

# =============================================================================
# 1. Dependency installation
# =============================================================================

def _pip(*args):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', *args])

_pip(
    'nltk>=3.8', 'spacy>=3.7', 'numpy>=1.26', 'pandas>=2.1',
    'matplotlib>=3.8', 'seaborn>=0.13', 'scikit-learn>=1.4', 'scipy>=1.12',
    'umap-learn>=0.5', 'sentence-transformers>=2.7', 'shap>=0.45',
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdfplumber
import seaborn as sns
import shap
import spacy
import umap
from nltk.corpus import stopwords as _nltk_stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore', category=FutureWarning)

# =============================================================================
# 3. CLI arguments
# =============================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='AI-Impact Essay Analysis Toolkit — 02 ML',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''\
            Examples
            --------
            # 1. Demo mode, print architecture
            python Pre_and_Post_GPT_Writing_Styles_02_ML.py --use-fake-data --print-architecture

            # 2. Demo mode, print assignments
            python Pre_and_Post_GPT_Writing_Styles_02_ML.py --use-fake-data --print-assignments

            # 3. Demo mode, silent (just write output)
            python Pre_and_Post_GPT_Writing_Styles_02_ML.py --use-fake-data

            # 4. Real data, print architecture
            python Pre_and_Post_GPT_Writing_Styles_02_ML.py --print-architecture
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
        help='Print the first assignment (first student) from each semester.',
    )
    return parser.parse_args()

ARGS = _parse_args()

# =============================================================================
# 4. Output directory
# =============================================================================

OUTPUT_DIR = Path('output_2_ml')
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
    r'^(?P<yyyymm>\d{6})-(?P<course_slug>.+?)_\((?P<code>[A-Z]{2}\d{2})\)-(?P<canvas_id>\d+)$'
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

        found_assign_dirs: dict[str, Path] = {}
        for d in submissions_dir.iterdir():
            if not d.is_dir():
                continue
            matched = _match_assignment(d.name)
            if matched:
                found_assign_dirs[matched] = d

        for configured_assignment in ASSIGNMENTS:
            if configured_assignment not in found_assign_dirs:
                print(
                    f'  [info] {sem.code} has no {configured_assignment}'
                )

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
                        f'using first alphabetically'
                    )
                yield EssayRecord(
                    semester     = sem,
                    assignment   = assignment,
                    student_id   = student_id,
                    student_name = student_name,
                    pdf_path     = sorted(pdfs)[0],
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
    ('88001', 'Fatima_Khan'),
    ('88002', 'Misbah_Nazir'),
    ('88003', 'Qurratulain_Sultan'),
    ('88004', 'Sawaira_Jutt'),
    ('88005', 'Umm_Hafsa'),
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

    Per-assignment student count: randomly 3-5 (reproducible via random_seed).
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

    base_assignments = [a for a in ASSIGNMENTS if a != 'Assignment_4']
    course_slug      = COURSES[0][0] if COURSES else 'CS6460'

    for yyyymm, code, canvas_id in fake_semesters:
        era_cutoff = int(yyyymm) <= CHATGPT_CUTOFF_YYYYMM
        template   = _PRE_AI_TEMPLATE if era_cutoff else _POST_AI_TEMPLATE

        semester_assignments = base_assignments.copy()
        if rng.random() < 0.5:
            semester_assignments.append('Assignment_4')

        sem_dir = root / (
            f'{yyyymm}-{course_slug}_Educational_Technology_({code})-{canvas_id}'
        )

        for i, assignment in enumerate(semester_assignments):
            assign_prefix = str(900000 + i)
            assign_dir    = sem_dir / 'submissions' / f'{assign_prefix}_{assignment}'

            n_students      = rng.randint(3, 5)
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

def print_architecture(data_root: Path) -> None:
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


def print_assignments(
    semesters: list[Semester],
    records:   list[EssayRecord],
) -> None:
    """
    For each semester, print the first assignment's first student submission.
    Shows: semester code, assignment name, and extracted text content.
    """
    record_index: dict[tuple[str, str], EssayRecord] = {}
    for rec in records:
        key = (rec.semester.canvas_id, rec.assignment)
        if key not in record_index:
            record_index[key] = rec

    print()
    print('ASSIGNMENT CONTENTS (first assignment, first student per semester)')
    print('=' * 60)

    for sem in semesters:
        submissions_dir = sem.path / 'submissions'
        if not submissions_dir.is_dir():
            continue
        assign_dirs = sorted(
            d for d in submissions_dir.iterdir() if d.is_dir()
        )
        if not assign_dirs:
            print(f'\n[{sem.code}] No assignment folders found.')
            continue

        first_assign_dir = assign_dirs[0]
        assignment_name  = (
            _match_assignment(first_assign_dir.name) or first_assign_dir.name
        )
        key = (sem.canvas_id, assignment_name)
        rec = record_index.get(key)

        print(f'\nSemester   : {sem.code}  ({sem.era})')
        print(f'Assignment : {assignment_name}')
        print('-' * 60)

        if rec is None or not rec.text:
            print('  [no text available for this assignment]')
        else:
            wrapped = textwrap.fill(
                rec.text, width=72,
                initial_indent='  ', subsequent_indent='  ',
            )
            print(wrapped)

        print()

# =============================================================================
# 12. Ingestion entry point
# =============================================================================

def run_ingestion(
    demo_mode: bool,
) -> tuple[list[Semester], list[EssayRecord], pd.DataFrame, dict]:
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
# 13. NLP models & theme
# =============================================================================

UAF = {
    'blue':   '#403F84',
    'sky':    '#2F9FD9',
    'yellow': '#F8D727',
    'green':  '#005C45',
}
_PALETTE_TWO  = [UAF['sky'],  UAF['yellow']]
_PALETTE_FOUR = [UAF['blue'], UAF['sky'], UAF['yellow'], UAF['green']]

sns.set_theme(style='whitegrid', palette=_PALETTE_FOUR)
plt.rcParams.update({
    'figure.dpi':       130,
    'font.size':        11,
    'axes.titlesize':   12,
    'axes.labelsize':   11,
    'legend.fontsize':  10,
    'figure.facecolor': 'white',
})

_NLP   = spacy.load('en_core_web_sm')
_SBERT = SentenceTransformer('all-MiniLM-L6-v2')

# =============================================================================
# 14. Shared types & protocols
# =============================================================================

@dataclass(frozen=True)
class DiscoveredMetric:
    name:           str
    importance:     float
    direction:      str
    description:    str
    example_values: dict[str, float] = field(default_factory=dict)


class FeatureExtractor(Protocol):
    name: str
    def extract(self, text: str) -> dict[str, float]: ...

# =============================================================================
# 15. Text preprocessing utilities
# =============================================================================

def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]


def tokenise_words(text: str, *, lowercase: bool = True) -> list[str]:
    tokens = word_tokenize(text.lower() if lowercase else text)
    return [t for t in tokens if t not in string.punctuation]


def embed_sentences(sentences: list[str]) -> np.ndarray:
    if not sentences:
        return np.zeros((0, 384), dtype=np.float32)
    return _SBERT.encode(
        sentences, show_progress_bar=False, normalize_embeddings=True,
    )


def embed_text(text: str) -> np.ndarray:
    sents = sent_tokenize(text)
    vecs  = embed_sentences(sents)
    return vecs.mean(axis=0) if len(vecs) else np.zeros(384, dtype=np.float32)

# =============================================================================
# 16. Feature extractors
# =============================================================================

@dataclass
class SurfaceExtractor:
    name: str = 'surface'

    def extract(self, text: str) -> dict[str, float]:
        tokens    = tokenise_words(text)
        sents     = sent_tokenize(text)
        words     = len(tokens)
        sent_lens = [len(word_tokenize(s)) for s in sents]
        avg_word_len = float(np.mean([len(t) for t in tokens])) if tokens else 0.0
        return {
            'word_count':          float(words),
            'sentence_count':      float(len(sents)),
            'avg_sentence_length': float(np.mean(sent_lens)) if sent_lens else 0.0,
            'std_sentence_length': float(np.std(sent_lens))  if sent_lens else 0.0,
            'avg_word_length':     avg_word_len,
            'char_word_ratio':     float(len(text.replace(' ', '')) / max(words, 1)),
            'colon_per_100w':      text.count(':')  * 100 / max(words, 1),
            'semicolon_per_100w':  text.count(';')  * 100 / max(words, 1),
            'exclaim_per_100w':    text.count('!')  * 100 / max(words, 1),
            'question_per_100w':   text.count('?')  * 100 / max(words, 1),
            'comma_per_100w':      text.count(',')  * 100 / max(words, 1),
            'dash_per_100w':       (
                text.count('-') + text.count('\u2013') + text.count('\u2014')
            ) * 100 / max(words, 1),
            'bracket_per_100w':    (
                text.count('(') + text.count('[')
            ) * 100 / max(words, 1),
        }


@dataclass
class LexicalExtractor:
    name:        str = 'lexical'
    window_size: int = 50

    def extract(self, text: str) -> dict[str, float]:
        tokens = tokenise_words(text)
        n      = len(tokens)
        if n == 0:
            return {k: 0.0 for k in [
                'ttr', 'mattr', 'hapax_ratio', 'top10_coverage',
                'stopword_ratio', 'content_word_ratio', 'long_word_ratio',
            ]}
        stop  = frozenset(_nltk_stopwords.words('english'))
        freq  = pd.Series(tokens).value_counts()
        hapax = int((freq == 1).sum())
        top10 = freq.head(10).sum()
        mattr_scores = [
            len(set(tokens[i:i + self.window_size])) / self.window_size
            for i in range(0, max(1, n - self.window_size + 1))
        ]
        content = [t for t in tokens if t not in stop and t.isalpha()]
        return {
            'ttr':                len(set(tokens)) / n,
            'mattr':              float(np.mean(mattr_scores)),
            'hapax_ratio':        hapax / n,
            'top10_coverage':     top10 / n,
            'stopword_ratio':     sum(1 for t in tokens if t in stop) / n,
            'content_word_ratio': len(content) / n,
            'long_word_ratio':    sum(1 for t in tokens if len(t) >= 7) / n,
        }


@dataclass
class SyntacticExtractor:
    name: str = 'syntactic'

    def extract(self, text: str) -> dict[str, float]:
        doc    = _NLP(text[:100_000])
        tags   = [t.pos_ for t in doc if not t.is_space]
        n      = max(len(tags), 1)
        counts = pd.Series(tags).value_counts()

        def ratio(pos: str) -> float:
            return counts.get(pos, 0) / n

        passive_count = sum(
            1 for token in doc
            if token.dep_ == 'auxpass'
            or (token.lemma_ == 'be' and token.head.tag_ == 'VBN')
        )
        depths     = [len(list(t.ancestors)) for t in doc if not t.is_space]
        mean_depth = float(np.mean(depths)) if depths else 0.0
        return {
            'pos_noun_ratio':   ratio('NOUN'),
            'pos_verb_ratio':   ratio('VERB'),
            'pos_adj_ratio':    ratio('ADJ'),
            'pos_adv_ratio':    ratio('ADV'),
            'pos_pron_ratio':   ratio('PRON'),
            'pos_propn_ratio':  ratio('PROPN'),
            'pos_conj_ratio':   ratio('CCONJ') + ratio('SCONJ'),
            'passive_per_100w': passive_count * 100 / n,
            'mean_parse_depth': mean_depth,
            'ner_density':      len(doc.ents) * 100 / n,
        }


@dataclass
class SemanticExtractor:
    name: str = 'semantic'

    def extract(self, text: str) -> dict[str, float]:
        paras     = split_paragraphs(text)
        sents     = sent_tokenize(text)
        sent_vecs = embed_sentences(sents)
        para_vecs = embed_sentences(paras) if paras else np.zeros((0, 384))
        seq_coh   = self._sequential_similarity(sent_vecs)
        spread    = self._mean_pairwise_distance(sent_vecs)
        intro_outro = (
            float(para_vecs[0] @ para_vecs[-1])
            if len(para_vecs) >= 2 else 0.0
        )
        para_redundancies = self._intra_para_redundancy(paras)
        lex_ttr = LexicalExtractor().extract(text)['ttr']
        sem_div = (
            1.0 - float(np.mean([
                float(sent_vecs[i] @ sent_vecs[j])
                for i in range(len(sent_vecs))
                for j in range(i + 1, len(sent_vecs))
            ]))
            if len(sent_vecs) > 1 else 0.0
        )
        return {
            'sequential_coherence':       seq_coh,
            'semantic_spread':            spread,
            'intro_outro_similarity':     intro_outro,
            'mean_intra_para_redundancy': (
                float(np.mean(para_redundancies)) if para_redundancies else 0.0
            ),
            'max_intra_para_redundancy':  (
                float(np.max(para_redundancies))  if para_redundancies else 0.0
            ),
            'lex_sem_diversity_gap':      lex_ttr - (1.0 - sem_div),
        }

    @staticmethod
    def _sequential_similarity(vecs: np.ndarray) -> float:
        if len(vecs) < 2:
            return 0.0
        return float(np.mean([vecs[i] @ vecs[i + 1] for i in range(len(vecs) - 1)]))

    @staticmethod
    def _mean_pairwise_distance(vecs: np.ndarray) -> float:
        if len(vecs) < 2:
            return 0.0
        sims = [
            float(vecs[i] @ vecs[j])
            for i in range(len(vecs))
            for j in range(i + 1, len(vecs))
        ]
        return 1.0 - float(np.mean(sims))

    @staticmethod
    def _intra_para_redundancy(paras: list[str]) -> list[float]:
        scores = []
        for para in paras:
            sents = sent_tokenize(para)
            if len(sents) < 2:
                continue
            vecs = embed_sentences(sents)
            sims = [
                float(vecs[i] @ vecs[j])
                for i in range(len(vecs))
                for j in range(i + 1, len(vecs))
            ]
            if sims:
                scores.append(float(np.mean(sims)))
        return scores


@dataclass
class StructuralExtractor:
    name: str = 'structural'
    _OPENERS: frozenset[str] = frozenset({
        'furthermore', 'moreover', 'additionally', 'in conclusion',
        'in summary', 'firstly', 'secondly', 'thirdly', 'finally',
        'in addition', 'consequently', 'therefore', 'to conclude',
        'to summarise', 'it is worth noting', 'it is important',
        'in this essay', 'this essay will', 'as discussed',
    })

    def extract(self, text: str) -> dict[str, float]:
        paras     = split_paragraphs(text)
        tokens    = tokenise_words(text)
        n_w       = max(len(tokens), 1)
        sents     = sent_tokenize(text)
        para_lens = [len(word_tokenize(p)) for p in paras]
        cv_paras  = (
            float(np.std(para_lens) / np.mean(para_lens))
            if len(para_lens) > 1 and np.mean(para_lens) > 0
            else 0.0
        )
        opener_hits = sum(
            1 for s in sents
            if any(s.lower().strip().startswith(op) for op in self._OPENERS)
        )
        concession_pattern = re.compile(
            r'\b(while|although|even though|whereas|despite)\b', re.IGNORECASE,
        )
        first_person = re.compile(
            r'\b(i|me|my|myself|we|our|us)\b', re.IGNORECASE,
        )
        return {
            'para_count':            float(len(paras)),
            'para_cv':               cv_paras,
            'opener_signpost_ratio': opener_hits / max(len(sents), 1),
            'concession_per_100w':   (
                len(concession_pattern.findall(text)) * 100 / n_w
            ),
            'first_person_per_100w': (
                len(first_person.findall(text)) * 100 / n_w
            ),
            'para_uniformity':       1.0 - cv_paras,
        }


ALL_EXTRACTORS: list = [
    SurfaceExtractor(),
    LexicalExtractor(),
    SyntacticExtractor(),
    SemanticExtractor(),
    StructuralExtractor(),
]

# =============================================================================
# 17. Feature extraction pipeline
# =============================================================================

def extract_features(
    text: str, extractors: list | None = None,
) -> dict[str, float]:
    extractors = extractors or ALL_EXTRACTORS
    features: dict[str, float] = {}
    for extractor in extractors:
        raw = extractor.extract(text)
        features.update({f'{extractor.name}__{k}': v for k, v in raw.items()})
    return features


def build_feature_matrix(
    corpus: dict, extractors: list | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, essays in corpus.items():
        for i, essay in enumerate(essays):
            try:
                feats = extract_features(essay, extractors)
            except Exception as exc:
                print(f'  [warn] extractor failed on {label}[{i}]: {exc}')
                feats = {}
            rows.append({'label': label, 'essay_index': i, **feats})
    df        = pd.DataFrame(rows)
    meta_cols = {'label', 'essay_index'}
    num_cols  = [c for c in df.columns if c not in meta_cols]
    df[num_cols] = df[num_cols].astype(float)
    return df

# =============================================================================
# 18. Anomaly detection
# =============================================================================

def detect_anomalies(
    feature_df:    pd.DataFrame,
    *,
    contamination: float = 0.15,
    random_state:  int   = 42,
) -> pd.Series:
    num_cols = feature_df.select_dtypes(include='number').columns.tolist()
    X        = feature_df[num_cols].fillna(0).values
    scaler   = StandardScaler()
    X_sc     = scaler.fit_transform(X)
    iso      = IsolationForest(
        contamination=contamination, random_state=random_state, n_jobs=-1,
    )
    iso.fit(X_sc)
    return pd.Series(
        iso.decision_function(X_sc), index=feature_df.index, name='anomaly_score',
    )

# =============================================================================
# 19. Supervised feature importance
# =============================================================================

def rank_features_by_importance(
    feature_df:   pd.DataFrame,
    label_col:    str = 'label',
    *,
    n_estimators: int = 300,
    random_state: int = 42,
) -> tuple:
    meta_cols = {label_col, 'essay_index'}
    num_cols  = [c for c in feature_df.columns if c not in meta_cols]
    X         = feature_df[num_cols].fillna(0).values
    y         = feature_df[label_col].values
    scaler    = StandardScaler()
    X_sc      = scaler.fit_transform(X)
    rf        = RandomForestClassifier(
        n_estimators=n_estimators, max_features='sqrt',
        random_state=random_state, n_jobs=-1,
    )
    rf.fit(X_sc, y)
    importance_df = pd.DataFrame({
        'feature':    num_cols,
        'importance': rf.feature_importances_,
        'std':        np.std(
            [tree.feature_importances_ for tree in rf.estimators_], axis=0,
        ),
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    return importance_df, rf, scaler, num_cols


def compute_shap_values(
    rf:            RandomForestClassifier,
    X_sc:          np.ndarray,
    feature_names: list[str],
) -> shap.Explanation:
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer(X_sc)

    # Explicitly map the text names to the SHAP explanation object
    shap_values.feature_names = feature_names
    
    return shap_values

# =============================================================================
# 20. Metric suggestion engine
# =============================================================================

def suggest_metrics(
    feature_df:     pd.DataFrame,
    importance_df:  pd.DataFrame,
    label_col:      str   = 'label',
    top_n:          int   = 10,
    *,
    min_importance: float = 0.01,
) -> list[DiscoveredMetric]:
    labels    = feature_df[label_col].unique().tolist()
    top_feats = importance_df[
        importance_df['importance'] >= min_importance
    ].head(top_n)
    suggestions: list[DiscoveredMetric] = []
    for _, row in top_feats.iterrows():
        feat  = row['feature']
        imp   = float(row['importance'])
        group_means: dict[str, float] = {
            lbl: float(feature_df.loc[feature_df[label_col] == lbl, feat].mean())
            for lbl in labels
        }
        if len(labels) == 2:
            l1, l2    = labels
            direction = (
                f'higher_in_{_slug(l1)}' if group_means[l1] > group_means[l2]
                else f'higher_in_{_slug(l2)}'
            )
        else:
            direction = 'unclear'
        suggestions.append(DiscoveredMetric(
            name           = feat,
            importance     = imp,
            direction      = direction,
            description    = _describe_feature(feat),
            example_values = group_means,
        ))
    return suggestions


def _slug(s: str) -> str:
    return re.sub(r'\W+', '_', s).strip('_').lower()


_FEATURE_DESCRIPTIONS: dict[str, str] = {
    'surface__word_count':                  'Total word count of the essay.',
    'surface__avg_sentence_length':         'Mean number of words per sentence.',
    'surface__std_sentence_length':         'Variability in sentence length (burstiness).',
    'surface__colon_per_100w':              'Colon frequency per 100 words.',
    'surface__semicolon_per_100w':          'Semicolon frequency per 100 words.',
    'surface__comma_per_100w':              'Comma frequency per 100 words.',
    'surface__dash_per_100w':               'Dash frequency per 100 words.',
    'surface__avg_word_length':             'Mean character count per word.',
    'lexical__ttr':                         'Type-token ratio (raw vocabulary richness).',
    'lexical__mattr':                       'Moving-average TTR (length-corrected richness).',
    'lexical__hapax_ratio':                 'Fraction of words appearing only once.',
    'lexical__long_word_ratio':             'Fraction of words >= 7 characters.',
    'lexical__stopword_ratio':              'Fraction of tokens that are stopwords.',
    'lexical__content_word_ratio':          'Fraction of tokens that are content words.',
    'syntactic__pos_adj_ratio':             'Proportion of adjectives in the text.',
    'syntactic__pos_adv_ratio':             'Proportion of adverbs.',
    'syntactic__pos_noun_ratio':            'Proportion of nouns.',
    'syntactic__pos_verb_ratio':            'Proportion of verbs.',
    'syntactic__pos_pron_ratio':            'Proportion of pronouns (first-person signal).',
    'syntactic__passive_per_100w':          'Passive voice constructions per 100 words.',
    'syntactic__mean_parse_depth':          'Mean syntactic parse depth.',
    'syntactic__ner_density':               'Named entity density per 100 tokens.',
    'semantic__sequential_coherence':       'Mean cosine similarity between adjacent sentences.',
    'semantic__semantic_spread':            'Mean pairwise sentence distance (topical diversity).',
    'semantic__intro_outro_similarity':     'Semantic similarity between first and last paragraph.',
    'semantic__mean_intra_para_redundancy': 'Average semantic redundancy within paragraphs.',
    'semantic__lex_sem_diversity_gap':      'Gap between lexical and semantic diversity (AI signal).',
    'structural__para_uniformity':          'How uniform paragraph lengths are (high -> template-like).',
    'structural__opener_signpost_ratio':    'Fraction of sentences starting with discourse markers.',
    'structural__first_person_per_100w':    'First-person pronoun rate (student voice).',
    'structural__concession_per_100w':      'Concessive constructions per 100 words.',
}


def _describe_feature(feature_name: str) -> str:
    if feature_name in _FEATURE_DESCRIPTIONS:
        return _FEATURE_DESCRIPTIONS[feature_name]
    _, _, short = feature_name.partition('__')
    return short.replace('_', ' ').capitalize() + '.'

# =============================================================================
# 21. Dimensionality reduction
# =============================================================================

def reduce_dimensions(
    feature_df:   pd.DataFrame,
    method:       str = 'umap',
    *,
    n_components: int = 2,
    random_state: int = 42,
) -> np.ndarray:
    meta_cols = {'label', 'essay_index'}
    num_cols  = [c for c in feature_df.columns if c not in meta_cols]
    X         = feature_df[num_cols].fillna(0).values
    scaler    = StandardScaler()
    X_sc      = scaler.fit_transform(X)
    match method:
        case 'umap':
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=min(15, len(X_sc) - 1),
                min_dist=0.1,
                random_state=random_state,
            )
        case 'tsne':
            reducer = TSNE(
                n_components=n_components,
                perplexity=min(30, len(X_sc) - 1),
                random_state=random_state,
            )
        case 'pca':
            reducer = PCA(n_components=n_components, random_state=random_state)
        case _:
            raise ValueError(
                f'Unknown method "{method}". Choose "umap", "tsne", or "pca".'
            )
    return reducer.fit_transform(X_sc)

# =============================================================================
# 22. Visualisation
# =============================================================================

_LABEL_COLOURS: dict[str, str] = {}


def _colour_for(label: str, palette: list[str] = _PALETTE_FOUR) -> str:
    if label not in _LABEL_COLOURS:
        _LABEL_COLOURS[label] = palette[len(_LABEL_COLOURS) % len(palette)]
    return _LABEL_COLOURS[label]


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n:         int   = 20,
    figsize:       tuple = (10, 7),
) -> plt.Figure:
    df     = importance_df.head(top_n).copy().sort_values('importance')
    labels = [_prettify(f) for f in df['feature']]
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    bars = ax.barh(
        labels, df['importance'], xerr=df['std'],
        color=UAF['blue'], ecolor=UAF['sky'], capsize=3, alpha=0.9,
    )
    ax.set_xlabel('Gini Importance')
    ax.set_title(
        f'Top {top_n} Discriminative Features (Random Forest)', fontweight='bold',
    )
    for bar, val in zip(bars, df['importance']):
        ax.text(
            val + 0.001, bar.get_y() + bar.get_height() / 2,
            f'{val:.3f}', va='center', fontsize=8,
        )
    return fig


def plot_embedding(
    feature_df: pd.DataFrame,
    coords:     np.ndarray,
    label_col:  str   = 'label',
    method:     str   = 'UMAP',
    figsize:    tuple = (8, 6),
) -> plt.Figure:
    labels  = feature_df[label_col].values
    markers = ['o', 's', '^', 'D', 'v', 'P']
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    for i, lbl in enumerate(pd.unique(labels)):
        mask = labels == lbl
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            label=lbl, color=_colour_for(lbl),
            marker=markers[i % len(markers)],
            s=80, alpha=0.85, edgecolors='white', linewidths=0.5,
        )
    ax.set_xlabel(f'{method} 1')
    ax.set_ylabel(f'{method} 2')
    ax.set_title(f'Essay Embedding Space ({method})', fontweight='bold')
    ax.legend(framealpha=0.9)
    return fig


def plot_metric_suggestions(
    suggestions: list[DiscoveredMetric],
    figsize:     tuple = (12, 7),
) -> plt.Figure:
    if not suggestions:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No suggestions to display.', ha='center', va='center')
        return fig
    n          = len(suggestions)
    names      = [_prettify(s.name) for s in suggestions]
    all_labels = list(suggestions[0].example_values.keys())
    n_groups   = len(all_labels)
    width      = 0.8 / n_groups
    x          = np.arange(n)
    fig, axes  = plt.subplots(
        2, 1, figsize=figsize, constrained_layout=True,
        gridspec_kw={'height_ratios': [3, 1]},
    )
    ax = axes[0]
    for gi, lbl in enumerate(all_labels):
        vals   = [s.example_values.get(lbl, 0.0) for s in suggestions]
        offset = (gi - (n_groups - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=lbl, color=_colour_for(lbl), alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Mean feature value')
    ax.set_title('Suggested Metrics: Per-Group Means', fontweight='bold')
    ax.legend()
    ax2 = axes[1]
    ax2.bar(
        x, [s.importance for s in suggestions],
        color=UAF['yellow'], edgecolor=UAF['blue'], linewidth=0.8,
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=35, ha='right', fontsize=9)
    ax2.set_ylabel('Importance')
    ax2.set_title('Feature Importance', fontweight='bold')
    return fig


def plot_shap_summary(
    shap_vals:     shap.Explanation,
    feature_names: list[str],
    max_display:   int = 15,
) -> plt.Figure:
    fig, _ = plt.subplots(figsize=(10, 6), constrained_layout=True)
    shap.plots.beeswarm(
        shap_vals[:, :, 1] if shap_vals.values.ndim == 3 else shap_vals,
        max_display=max_display,
        show=False,
        color_bar=True,
        plot_size=None,
    )
    plt.title('SHAP Feature Impact (Post-ChatGPT direction)', fontweight='bold')
    return fig


def plot_anomaly_scores(
    feature_df:     pd.DataFrame,
    anomaly_scores: pd.Series,
    label_col:      str   = 'label',
    figsize:        tuple = (9, 5),
) -> plt.Figure:
    plot_df = feature_df[[label_col]].copy()
    plot_df['anomaly_score'] = anomaly_scores.values
    order   = plot_df[label_col].unique().tolist()
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    sns.boxplot(
        data=plot_df, x=label_col, y='anomaly_score', ax=ax,
        order=order, palette={lbl: _colour_for(lbl) for lbl in order},
        width=0.4, fliersize=0,
    )
    sns.stripplot(
        data=plot_df, x=label_col, y='anomaly_score', ax=ax,
        order=order, color='black', alpha=0.5, size=5, jitter=True,
    )
    ax.axhline(0, linestyle='--', color='grey', linewidth=0.8)
    ax.set_ylabel('Anomaly Score (higher = more typical)')
    ax.set_xlabel('')
    ax.set_title('Essay Anomaly Scores by Group', fontweight='bold')
    return fig


def _prettify(feature_name: str) -> str:
    _, _, short = feature_name.partition('__')
    return short.replace('_', ' ').title()

# =============================================================================
# 23. Metric report
# =============================================================================

def print_metric_report(suggestions: list[DiscoveredMetric]) -> None:
    sep = '=' * 64
    print(sep)
    print('ML-DISCOVERED METRIC SUGGESTIONS')
    print(sep)
    for rank, metric in enumerate(suggestions, start=1):
        print(f'\n#{rank:>2}  {metric.name}')
        print(f'     Importance : {metric.importance:.4f}')
        print(f'     Direction  : {metric.direction}')
        print(f'     Description: {metric.description}')
        if metric.example_values:
            for lbl, val in metric.example_values.items():
                print(f'     [{lbl}]: {val:.4f}')
    print(f'\n{sep}\n')


def save_metric_report(suggestions: list[DiscoveredMetric], path: Path) -> None:
    rows = [
        {
            'rank':        rank,
            'name':        m.name,
            'importance':  m.importance,
            'direction':   m.direction,
            'description': m.description,
            **{
                f'mean_{_slug(lbl)}': val
                for lbl, val in m.example_values.items()
            },
        }
        for rank, m in enumerate(suggestions, start=1)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)

# =============================================================================
# 24. Full pipeline orchestrator
# =============================================================================

def run_ml_discovery(
    corpus:       dict,
    *,
    top_n:        int  = 10,
    reduction:    str  = 'umap',
    show_shap:    bool = True,
    random_state: int  = 42,
) -> dict[str, Any]:
    print('[1/5] Extracting features ...')
    feature_df = build_feature_matrix(corpus)
    print(f'      {len(feature_df)} essays x {len(feature_df.columns) - 2} features')

    print('[2/5] Detecting anomalies ...')
    anomaly_scores = detect_anomalies(feature_df, random_state=random_state)
    fig_anom = plot_anomaly_scores(feature_df, anomaly_scores)
    fig_anom.savefig(
        OUTPUT_DIR / 'anomaly_scores.png', dpi=130, bbox_inches='tight',
    )
    plt.close(fig_anom)

    print('[3/5] Ranking features (Random Forest) ...')
    importance_df, rf, scaler, num_cols = rank_features_by_importance(
        feature_df, random_state=random_state,
    )
    importance_df.to_csv(OUTPUT_DIR / 'feature_importance.csv', index=False)

    fig_imp = plot_feature_importance(importance_df, top_n=min(20, len(importance_df)))
    fig_imp.savefig(
        OUTPUT_DIR / 'feature_importance.png', dpi=130, bbox_inches='tight',
    )
    plt.close(fig_imp)

    shap_vals = None
    if show_shap and len(feature_df) >= 4:
        print('[3b]  Computing SHAP values ...')
        X_sc      = scaler.transform(feature_df[num_cols].fillna(0).values)
        shap_vals = compute_shap_values(rf, X_sc, num_cols)
        fig_shap  = plot_shap_summary(shap_vals, num_cols)
        fig_shap.savefig(
            OUTPUT_DIR / 'shap_summary.png', dpi=130, bbox_inches='tight',
        )
        plt.close(fig_shap)

    print('[4/5] Generating metric suggestions ...')
    suggestions = suggest_metrics(feature_df, importance_df, top_n=top_n)
    print_metric_report(suggestions)
    save_metric_report(suggestions, OUTPUT_DIR / 'metric_suggestions.csv')

    fig_sugg = plot_metric_suggestions(suggestions)
    fig_sugg.savefig(
        OUTPUT_DIR / 'metric_suggestions.png', dpi=130, bbox_inches='tight',
    )
    plt.close(fig_sugg)

    print('[5/5] Projecting to embedding space ...')
    coords    = reduce_dimensions(
        feature_df, method=reduction, random_state=random_state,
    )
    fig_embed = plot_embedding(feature_df, coords, method=reduction.upper())
    fig_embed.savefig(
        OUTPUT_DIR / 'embedding.png', dpi=130, bbox_inches='tight',
    )
    plt.close(fig_embed)

    feature_df.to_csv(OUTPUT_DIR / 'feature_matrix.csv', index=False)

    print(f'[done] All results written to {OUTPUT_DIR.resolve()}')
    return {
        'feature_df':     feature_df,
        'anomaly_scores': anomaly_scores,
        'importance_df':  importance_df,
        'suggestions':    suggestions,
        'shap_vals':      shap_vals,
        'coords':         coords,
        'rf':             rf,
    }

# =============================================================================
# 25. Main
# =============================================================================

def main() -> None:
    # ------------------------------------------------------------------ ingest
    semesters, records, summary_df, corpus = run_ingestion(ARGS.use_fake_data)

    # ----------------------------------------------------------- diagnostics
    if ARGS.print_architecture:
        if ARGS.use_fake_data:
            data_root = semesters[0].path.parent.parent if semesters else None
        else:
            data_root = Path(DATA_ROOT) if Path(DATA_ROOT).exists() else None

        if data_root:
            print_architecture(data_root)
        else:
            print('[info] No data root available; --print-architecture skipped.')

    if ARGS.print_assignments:
        if semesters and records:
            print_assignments(semesters, records)
        else:
            print('[info] No data loaded; --print-assignments has nothing to show.')

    # --------------------------------------------------------- guard: no data
    if not records:
        print('[info] No records loaded. Exiting without analysis.')
        return

    # ------------------------------------------------------- ML pipeline
    run_ml_discovery(corpus, top_n=10, reduction='umap')


if __name__ == '__main__':
    main()
