"""Read-only display/CUDA classification with synthetic process/XML data."""
import pytest

from studies._shared.structure import gpu_admission as admission


@pytest.mark.parametrize('name', ['Figma.exe', 'chrome.exe', 'msedge.exe', 'firefox.exe', 'dwm.exe',
    'explorer.exe', 'WINWORD.EXE', 'EXCEL.EXE', 'POWERPNT.EXE', 'RayLink.exe', 'RayLinkCapturer.exe',
    'ToDesk.exe', 'SunloginClient.exe', 'Parsec.exe', 'TeamViewer.exe'])
@pytest.mark.parametrize('kind', ['G', 'C+G'])
def test_recognized_display_process_is_not_a_blocker(name, kind):
    result = admission.classify(dict(pid=1, process_name='C:\\Apps\\' + name, type=kind,
                             gpu_memory_mib=None, cuda_memory_mib=None))
    assert result['category'] == 'display' and not result['blocking']


@pytest.mark.parametrize('name,kind,memory,blocked', [('mystery.exe', 'C', None, False),
    ('mystery.exe', 'C', 0., False), ('mystery.exe', 'C', 1024., True),
    ('mystery.exe', 'C+G', None, False), ('python.exe', 'C', None, True),
    ('python.exe', 'C', 0., False), ('Figma.exe', 'C', 1024., True)])
def test_only_confirmed_cuda_or_known_runtime_context_blocks(name, kind, memory, blocked):
    row = dict(pid=42, process_name=name, type=kind, cuda_memory_mib=memory)
    assert admission.classify(row)['blocking'] is blocked
    assert not admission.classify(row, {42})['blocking']


def test_mixed_display_context_is_not_assigned_fake_cuda_bytes():
    text = '<nvidia_smi_log><gpu><fb_memory_usage><free>12000 MiB</free><used>12000 MiB</used></fb_memory_usage><utilization><gpu_util>1 %</gpu_util></utilization><processes>'
    for pid, kind, name, memory in [(1, 'C+G', 'RayLinkCapturer.exe', 'N/A'), (2, 'C', 'unknown.exe', '2048 MiB'), (3, 'C', 'python.exe', 'N/A'), (4, 'C+G', 'unknown-ui.exe', '500 MiB')]:
        text += f'<process_info><pid>{pid}</pid><type>{kind}</type><process_name>{name}</process_name><used_memory>{memory}</used_memory></process_info>'
    text += '</processes></gpu></nvidia_smi_log>'
    result = admission.parse_xml(text, {3})
    assert [r['pid'] for r in result['foreign_gpu_processes']] == [2]
    assert result['classified_processes'][3]['gpu_memory_mib'] == 500
    assert result['classified_processes'][3]['cuda_memory_mib'] is None
    assert admission.memory_mib('N/A') is None and admission.memory_mib('1 GiB') == 1024
