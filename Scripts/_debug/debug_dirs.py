from __future__ import annotations

import os
import re
import textwrap
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# =============================================================================
# 4. Output directory
# =============================================================================

OUTPUT_DIR = Path('output_0_debug')
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
        submissions_dir = next((d for d in sem.path.iterdir() if d.is_dir() and d.name.lower() == 'submissions'), None)
        if not submissions_dir:
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

    for dirpath, _, filenames in os.walk(data_root):
        depth = dirpath.replace(str(data_root), '').count(os.sep)
        prefix = '|   ' * depth
        dirname = os.path.basename(dirpath) or str(data_root)
        print(f'{prefix}|-- {dirname}')
        for f in sorted(filenames):
            print(f'{prefix}|   |-- {f}')

    print()


def print_architecture(data_root: Path) -> None:
    out_path = OUTPUT_DIR / 'structure.txt'
    lines: list[str] = []
    lines.append(f'[DIR]  {data_root.parent.resolve()}/')

    for dirpath, _, filenames in os.walk(data_root):
        depth = dirpath.replace(str(data_root), '').count(os.sep)
        prefix = '|   ' * depth
        dirname = os.path.basename(dirpath) or str(data_root)
        lines.append(f'{prefix}|-- [DIR]  {dirname}/')
        for f in sorted(filenames):
            lines.append(f'{prefix}|   |-- [FILE] {f}')

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
        submissions_dir = next((d for d in sem.path.iterdir() if d.is_dir() and d.name.lower() == 'submissions'), None)
        if not submissions_dir:
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
        submissions_dir = next((d for d in sem.path.iterdir() if d.is_dir() and d.name.lower() == 'submissions'), None)
        if not submissions_dir:
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
# 22. Main
# =============================================================================

def main() -> None:
    # ----------------------------------------------------------- diagnostics
    data_root = Path(DATA_ROOT) if Path(DATA_ROOT).exists() else None
    if data_root:
        print_architecture(data_root)

    
if __name__ == '__main__':
    main()
