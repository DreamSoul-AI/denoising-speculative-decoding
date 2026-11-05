"""Native Study phase and explicit saved-count analysis entry."""
from studies._shared.flow.process import run
from studies.output_space_4b_recheck_v2.flow.process.run import analyze, main

__all__ = ['run', 'analyze']


if __name__ == '__main__':
    raise SystemExit(main(study_name='output_space_4b_input_control'))
