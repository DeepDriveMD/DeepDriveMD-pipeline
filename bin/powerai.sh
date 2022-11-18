module load gcc/7.3.1
. /etc/profile.d/conda.sh
conda activate /usr/workspace/cv_ddmd/conda3/powerai

export IBM_POWERAI_LICENSE_ACCEPT=yes
module use /usr/workspace/cv_ddmd/software1/modules
module load adios2/2.8.1a

export WORKSPACE=/usr/workspace/$USER
export SHARED=/usr/workspace/cv_ddmd/$USER
export GPFS1=/p/gpfs1/$USER
export SB=$GPFS1/radical.pilot.sandbox

export MyDir=$1
export MyDir1=$GPFS1/DDMD/$MyDir/entk_cvae_md
export MyDir2=$MyDir1/analysis

export PYTHONPATH=$MyDir1/Outlier_search:$PYTHONPATH

export PATH=$HOME/bin:$HOME/spack/bin:$PATH

shopt -s direxpand

alias wks='cd $WORKSPACE'
alias gpfs='cd $GPFS1'

alias sb='cd $SB'
alias sb1='cd $SB/$(ls -d $MyDir1/re.* | xargs basename)/pilot.0000'

alias cd1='cd $MyDir1'
alias cd2='cd $MyDir2'
alias cd3='cd $SHARED'

alias bu='bkill -u $USER'

alias topU='top -U $USER'
alias bj='bjobs'
alias current='cd /g/g15/yakushin/tmp2/tmp/DeepDriveMD-pipeline/'
alias sim='cd /g/g15/yakushin/tmp2/tmp/DeepDriveMD-pipeline/deepdrivemd/sim/openmm_stream'
alias agg='cd /g/g15/yakushin/tmp2/tmp/DeepDriveMD-pipeline/deepdrivemd/aggregation/stream'
alias ml='cd /g/g15/yakushin/tmp2/tmp/DeepDriveMD-pipeline/deepdrivemd/models/keras_cvae_stream'
alias outliers='cd /g/g15/yakushin/tmp2/tmp/DeepDriveMD-pipeline/deepdrivemd/agents/stream'
alias outputs_ls='echo "/p/gpfs1/yakushin/Outputs"; ls -ltr /p/gpfs1/yakushin/Outputs'
alias outputs='cd /p/gpfs1/yakushin/Outputs'

function mcd1()
{
    cd $GPFS1/DDMD/$1/entk_cvae_md
}

function mcd2()
{
    cd $GPFS1/DDMD/$1/entk_cvae_md/analysis
}

function msb1()
{
    cd $SB/$(ls -d $GPFS1/DDMD/$1/entk_cvae_md/re.* | xargs basename)/pilot.0000
}


function em
{
    emacs -nw +$2:1 $1
}

function logs()
{
    cd /p/gpfs1/yakushin/radical.pilot.sandbox
    cd $(ls -tr | tail -1)
    cd pilot*
}

function logs1()
{
    cd /p/gpfs1/yakushin/radical.pilot.sandbox
    cd $1/pilot*
}
