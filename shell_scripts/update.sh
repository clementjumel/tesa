#!/bin/zsh

rsync -r -e "ssh -p 2222" -a  ~/Desktop/master_thesis/shell_scripts \
                              ~/Desktop/master_thesis/modeling \
                              ~/Desktop/master_thesis/fairseq \
                              ~/Desktop/master_thesis/toolbox \
                              ~/Desktop/master_thesis/run_model.py \
                          jumelcle@login.mila.quebec:master_thesis

echo "Done."; echo
