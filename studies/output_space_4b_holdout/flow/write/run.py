from studies._shared.flow.write import run
__all__ = ['run']


def main(argv=None):
    from studies._shared.flow.write.run import paired_report_main
    return paired_report_main('output_space_4b_holdout', argv)


if __name__ == '__main__':
    raise SystemExit(main())
