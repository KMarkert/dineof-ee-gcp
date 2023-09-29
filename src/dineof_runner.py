import chevron
import subprocess
from pathlib import Path
import logging

FILEDIR = Path(__file__).parent.resolve()

def write_dineof_config(outfile, dineof_in, variable, min_modes=10, max_modes=20, land_mask='land_mask', reconstruction=0):

    template = FILEDIR / 'dineof_config_template.init'

    ncv = max_modes + 5

    data = dict(
        dineof_in=dineof_in,
        variable=variable,
        min_modes=min_modes, 
        max_modes=max_modes, 
        land_mask=land_mask,
        reconstruction=reconstruction,
        ncv = ncv
    )

    with open(str(template), 'r') as f:
        outconfig = chevron.render(f, data)

    with open(outfile,"w") as f:
        f.write(outconfig)

    return

def exec_dineof(dineof_in, variable, **kwargs):

    config_path = f'dineof_{variable}.init'

    write_dineof_config(config_path, dineof_in, variable, **kwargs)

    cmd = f'dineof {config_path}'

    proc = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    out, err = proc.communicate()

    if out is not None:
        print(out.decode("utf-8"))
    
    if err is not None:
        print('Error:',err.decode("utf-8"))

    return

    

