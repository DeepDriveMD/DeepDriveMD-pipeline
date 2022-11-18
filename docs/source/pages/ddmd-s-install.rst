=====================================================================
How to install software dependencies for DeepDriveMD-S on Lassen@LLNL
=====================================================================

#. Set up the environment for Anaconda::

     module load gcc/7.3.1
     . /etc/profile.d/conda.sh
     export IBM_POWERAI_LICENSE_ACCEPT=yes

#. Add the following lines to ``$HOME/.condarc`` so that the packages are downloaded not into ``$HOME`` but into ``$CONDA2``::

     pkgs_dirs:
        - /usr/workspace/cv_ddmd/conda2/pkgs


#. Follow IBM's documentation to install powerai environment (``https://www.ibm.com/docs/en/wmlce/1.7.0?topic=installing-mldl-frameworks``) with some modifications::

     conda config --prepend channels https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda/
     export CONDA2=/usr/workspace/cv_ddmd/conda2
     mkdir $CONDA2
     conda create -p $CONDA2/powerai
     conda activate $CONDA2/powerai
     conda install powerai
     conda install powerai-rapids

   The last two steps can easily take hours before you are even asked to answer y/n.

	       

#. Install ADIOS::
     
     git clone git@github.com:ornladios/ADIOS2.git
     cd ADIOS2
     mkdir build; cd build
     cmake -DADIOS2_USE_MPI=OFF -DADIOS2_USE_CUDA=OFF -DCMAKE_INSTALL_PREFIX=/usr/workspace/cv_ddmd/software1/ADIOS2a  ..
     make -j20
     make install
     export PYTHONPATH=/usr/workspace/cv_ddmd/software1/ADIOS2a/lib/python3.7/site-packages:$PYTHONPATH
     export PATH=/usr/workspace/cv_ddmd/software1/ADIOS2_060522/bin:$PATH
     export LD_LIBRARY_PATH=/usr/workspace/cv_ddmd/software1/ADIOS2a/lib64:$LD_LIBRARY_PATH
      
#. Assuming that you have configured  `spack <https://spack.io/>`_ , install swig::

     spack install swig

#. Install openmm::
     
     module load cuda/10.2.89
     module load cmake/3.23.1
     spack load swig
     git clone https://github.com/pandegroup/openmm.git
     cd openmm
     mkdir build; cd build
     cmake -DCMAKE_INSTALL_PREFIX=/usr/workspace/cv_ddmd/software1/OPENMMa -DSWIG_EXECUTABLE=`which swig` ..
     make -j20
     make install
     make PythonInstall
     python -m openmm.testInstallation

   The last statement runs tests.

#. Miscellaneous dependencies::

     pip install pathos
     pip install mdanalysis==1.0.0
     pip install opentsne
     pip install radical.entk==1.6.7 radical.gtod==1.6.7 radical.pilot==1.6.7 radical.saga==1.6.10 radical.utils==1.6.7
     pip install torchsummary
     
   I had trouble with the latest versions of Radical.

#. Install mdlearn::

     git clone https://github.com/ramanathanlab/mdlearn
     cd mdlearn
     git checkout develop
     pip install -e .

#. Install MD-Tools::

     git clone git@github.com:braceal/MD-tools.git
     cd MD-tools
     pip install -e .

#. Install DeepDriveMD::

     mkdir /usr/workspace/cv_ddmd/$USER
     cd /usr/workspace/cv_ddmd/$USER
     git clone git@github.com:DeepDriveMD/DeepDriveMD-pipeline.git
     cd DeepDriveMD-pipeline
     pip install -e .
