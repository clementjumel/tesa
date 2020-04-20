#!/bin/zsh

rsync -r -e "ssh -p 2222" -a  ~/Desktop/master_thesis/modeling \
                              ~/Desktop/master_thesis/mt_models.py \
                              ~/Desktop/master_thesis/shell_scripts \
                              ~/Desktop/master_thesis/toolbox \
                          jumelcle@login.mila.quebec:master_thesis

echo "Code updated."
