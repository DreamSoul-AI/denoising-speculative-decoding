"""Relative links, including spaced angle targets, resolve in repo-owned Markdown."""

import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
LINK = re.compile(r'\]\((?:<([^>]+)>|([^\s)]+))\)')
# These four byte-pinned upstream texts retain DEM-only references documented in their README.
PINNED_STANDARDS = {
    'docs/research/standards/研究计划规范 Research Plan Standard.md',
    'docs/research/standards/研究计划模板 Research Plan Template.md',
    'docs/research/standards/research_plan_standard_en.md',
    'docs/research/standards/research_plan_template_en.md',
}


def test_relative_markdown_links_resolve():
    files = subprocess.check_output(['git', '-c', 'safe.directory=' + str(ROOT), 'ls-files', '-z', '*.md'], cwd=ROOT, text=True, encoding='utf-8').rstrip('\0').split('\0')
    broken = []
    for name in files:
        if name.startswith(('src/rpipe/', 'src/spec_mtp/structure/algorithm/reference/')):
            continue
        text = (ROOT / name).read_text(encoding='utf-8')
        for match in LINK.finditer(text):
            raw = match[1] or match[2]
            if raw.startswith(('http://', 'https://', 'mailto:', '#', 'data:', 'file:')):
                continue
            target = unquote(raw.partition('#')[0])
            if name in PINNED_STANDARDS and target in {'../文献调研/文献调研规范 Literature Review Standard.md', '../../开发/通用调研规范 General Research Standard.md', '../../../../../研发/研究/校准 Calibration/校准 Calibration 文献调研 Literature Review 大纲.md'}:
                continue
            if not (ROOT / name).parent.joinpath(target).exists():
                broken.append(f'{name} -> {target}')
    assert not broken, broken






def test_first_party_docs_have_one_canonical_chinese_copy():
    assert {p.name for p in (ROOT / 'docs').glob('*.md')} == {
        '去噪推测解码 通用调研.md', '去噪推测解码 架构设计.md',
        '去噪推测解码 代码结构设计.md', '去噪推测解码 研究计划.md'}
