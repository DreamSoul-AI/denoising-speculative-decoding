"""Preserve the normal write phase and render explicitly bound historical CPU counts."""
from studies._shared.flow.write import run
__all__=['run']


def write_plot(input_path,input_sha256,output_root,*,repository=None,commit=None,input_git_sha256=None):
    import json
    from pathlib import Path
    from spec_mtp.structure.artifact import layout,paths,provenance
    from studies.smoke.structure import validation as protocol
    from studies._shared.flow.write.run import publish_new_completion
    supplied=(repository is not None,commit is not None,input_git_sha256 is not None)
    if any(supplied) and not all(supplied): raise ValueError('historical Git binding requires repository, full commit and SHA256')
    alias=Path(input_path).absolute();primary=alias.resolve(strict=True)
    checkout=Path(__file__).resolve().parents[4]
    roots=[checkout,primary.parent]
    if repository is not None: roots.append(Path(repository).resolve(strict=True))
    output=layout.validate_study_root(paths.validate_output_root(output_root,*roots))
    if output.exists() or output.is_symlink(): raise FileExistsError('historical smoke plot requires a new output root')
    if provenance.file_sha256(primary) != input_sha256: raise ValueError('historical smoke input differs from actual SHA256')
    files=[Path(__file__),Path(protocol.__file__),Path(publish_new_completion.__code__.co_filename),
           Path(layout.__file__),Path(paths.__file__),Path(provenance.__file__)]
    sources={str(p.resolve()):provenance.file_sha256(p) for p in files}
    raw=primary.read_bytes();data=json.loads(raw);historical=None
    if repository is not None:
        member='studies/smoke/docs/delivery_validation.json'
        archived=provenance.read_historical_bytes(repository,commit,member,input_git_sha256)
        if data != json.loads(archived): raise ValueError('actual CPU report differs from its fixed Git original')
        historical={'commit':commit,'path':member,'sha256':input_git_sha256}
    def verify():
        if (alias.resolve(strict=True) != primary or provenance.file_sha256(primary) != input_sha256
                or any(provenance.file_sha256(p) != sha for p,sha in sources.items())):
            raise ValueError('historical smoke input or figure source changed')
    verify();figure=protocol.historical_cpu_figure(data)
    try:
        verify();output.mkdir(parents=True,exist_ok=False)
        from spec_mtp.structure.system.logger import Logger
        Logger(output/'assets', managed=True).info('saved CPU/934dac3 smoke counts; no model or tensor execution')
        image=output/'tiny_acceptance.png'
        with image.open('xb') as stream:figure.savefig(stream,format='png',dpi=160)
        verify()
    except BaseException as error:
        try:
            figure.clear()
        except BaseException as cleanup:
            setattr(error, 'figure_cleanup_error', cleanup)
        raise
    figure.clear()
    record={'schema':1,'scope':'historical CPU saved scalars only; no GPU comparison or new execution',
        'implementation_commit':protocol.IMPLEMENTATION_COMMIT,
        'runtime_input':{'path':str(primary),'sha256':input_sha256},'historical_input':historical,
        'sources':sources,'image':{'path':image.name,'sha256':provenance.file_sha256(image)}}
    payload=json.dumps(record,indent=2,allow_nan=False)+'\n'
    verify()
    publish_new_completion(output/'plot_complete.json',payload)
    return image


def main(argv=None):
    import argparse
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('input','output-root'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--input-sha256',required=True)
    parser.add_argument('--repository',type=Path);parser.add_argument('--commit');parser.add_argument('--input-git-sha256')
    args=parser.parse_args(argv)
    write_plot(args.input,args.input_sha256,args.output_root,repository=args.repository,commit=args.commit,input_git_sha256=args.input_git_sha256)
    return 0


if __name__=='__main__':raise SystemExit(main())
